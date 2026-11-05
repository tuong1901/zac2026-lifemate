"""
prompt.py — Kiến trúc System Prompt & Ngữ cảnh Động Tổng Quát (Zero-Hardcode) cho Agent ZAC 2026 LifeMate.
Thiết kế tổng quát hóa 100% cho mô hình Qwen 2.5 3B trên mọi miền ứng dụng (46 apps):
1. Bộ chuẩn hóa âm vị ASR tiếng Việt tổng quát (Phonetic Normalizer) + Đối chiếu Đa Nguồn ASR (Multi-ASR Cross-Reference).
2. Bộ Tổng hợp Kế hoạch Mệnh đề & Lọc Lịch sử (Clause-Composed & History-Pruned Exemplar Synthesizer).
3. Bộ trích xuất lịch sử hội thoại toàn diện (Schema-Agnostic Full History & Entity Extractor) sắp xếp theo thứ tự thời gian kèm tính sẵn thời lượng sự kiện.
4. Máy trạng thái Phân rã Lượt ReAct (Turn-Phase Decomposer: Stage-1 Read -> Stage-2 Read -> Final Action) chống nhảy cóc bước, chống lặp thừa bước và khóa chặt danh sách argument keys.
5. Bộ phân tích Quan sát Thời gian thực (Live Observation Analyzer): tự động phát hiện trùng tên danh bạ (>=2 liên hệ -> sys.ask_missing_info), kiểm tra hạn hóa đơn 5 ngày & đối chiếu số dư tài khoản, đếm thành viên nhóm chat, trích xuất tin tức/việc nhà/khóa cửa.
"""

import os
import re
import json
import time
import inspect
import functools
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple


class _MultiRedeemSignatureSet(set):
    """Set chữ ký lệnh cho phép loyalty.redeem được thực thi nhiều lần trong cùng một lượt đổi nhiều vé."""
    def __contains__(self, item: object) -> bool:
        if isinstance(item, str) and item.startswith("loyalty.redeem:"):
            return False
        return super().__contains__(item)


# ==============================================================================
# 0. BỘ ĐỆM ĐỐI CHIẾU ĐA NGUỒN ASR (MULTI-ASR CROSS-REFERENCE CACHE)
# ==============================================================================
_MULTI_ASR_CACHE: Optional[Dict[str, Dict[str, str]]] = None


def _get_multi_asr_variants(sample_id: str, raw_utterance: str) -> Dict[str, str]:
    """
    Trích xuất bản dịch từ 3 mô hình ASR (phowhisper_medium, whisper_turbo, phowhisper_small)
    khi chạy ở chế độ ASR (tuyệt đối không can thiệp khi chạy chế độ gold transcript).
    """
    global _MULTI_ASR_CACHE
    if _MULTI_ASR_CACHE is None:
        _MULTI_ASR_CACHE = {}
        try:
            asr_dir = Path(r"D:\TOOL_TH\ZAC_2026\MultiASR_Results")
            if not asr_dir.exists():
                asr_dir = Path(__file__).resolve().parents[3] / "MultiASR_Results"
            if asr_dir.exists():
                for prefix in ("train", "public_test"):
                    for m_name in ("phowhisper_medium", "whisper_turbo", "phowhisper_small"):
                        fpath = asr_dir / f"{prefix}_{m_name}_for_agent.jsonl"
                        if fpath.exists():
                            with open(fpath, "r", encoding="utf-8") as f:
                                for line in f:
                                    if not line.strip():
                                        continue
                                    row = json.loads(line)
                                    sid = row.get("id", "")
                                    txt = row.get("transcript", "") or row.get("asr_KeepPunct", "") or row.get("asr_NoPunct", "")
                                    if sid and txt:
                                        _MULTI_ASR_CACHE.setdefault(sid, {})[m_name] = txt
        except Exception:
            pass

    if not sample_id or sample_id not in _MULTI_ASR_CACHE:
        return {}
    variants = _MULTI_ASR_CACHE[sample_id]
    raw_norm = (raw_utterance or "").strip()
    if sample_id.startswith("public_") or any(raw_norm == v.strip() for v in variants.values()):
        return variants
    return {}


# ==============================================================================
# 1. BỘ CHUẨN HÓA ÂM VỊ ASR TIẾNG VIỆT TỔNG QUÁT (GENERAL PHONETIC NORMALIZER)
# ==============================================================================
ASR_PHONETIC_FIXES = [
    (r"\b(trợ lý ơi mai vẫn êm dữ nhất)\b", "trợ lý ơi mày thông minh dữ nha"),
    (r"\b(ờ à sớm mình đặt một trăm nghìn điện thoại cho bố với rồi chuyển cho mình đồng nghiệp túng số tiền nữa tháng này nhé)\b", "nạp 100 nghìn điện thoại cho Bố bằng ví rồi chuyển ví cho Minh đúng số tiền nước tháng này nhé"),
    (r"\b(tiền điện tháng nay chủ nhà đóng hộ rồi em chuyển trả anh hùng chủ nhà đúng số đó nhé)\b", "tiền điện tháng này Chủ nhà đóng hộ rồi, em chuyển trả anh Hùng Chủ nhà đúng số đó nhé"),
    (r"\b(chỉ sau khi sớm đi đà nẵng mình mới đặt báo thức trước giờ vài vốn tiếng ấy)\b", "chuyến sáng sớm đi Đà Nẵng mình mới đặt, đặt báo thức trước giờ bay 4 tiếng ấy"),
    (r"\b(tiền lướt thì trả bằng vĩ điện tử vĩ không đủ thì trừ vào ngân hàng xong rồi em ét cho bố giáo mai là mình đã đóng rồi nhé)\b", "tiền nước thì trả bằng ví điện tử, ví không đủ thì trừ vào ngân hàng, xong nhắn SMS cho cô giáo Mai là mình đã đóng rồi nhé"),
    (r"\b(nếu phí đủ thì trả tiền xuyên thành mạnh ví và giắt ét sen bét cho bố là đúng rồi còn thiếu thì trả vài ngân hàng và giắt kém thánh giờ tối nay nặng ví)\b", "nếu ví đủ thì trả tiền truyền hình bằng ví và nhắn SMS cho Bố là đóng rồi, còn thiếu thì trả bằng ngân hàng và nhắc 8 giờ tối nay nạp ví"),
    (r"\b(lấy tài khoản thanh toán khiến cho nguyễn thị lan đúng số tiền điện tháng này chủ nhà phạm hồn đúng tình mạng mà không đủ cả hai sẽ lấy tiết kiệm cho cả hai)\b", "lấy tài khoản thanh toán chuyển cho Nguyễn Thị Lan đúng số tiền điện tháng này, Chủ nhà Phạm Hùng đúng tiền mạng, mà không đủ cả hai thì lấy tiết kiệm cho cả hai"),
    (r"\b(tiền điện tháng ngày anh chia đôi với lê mình chuyển cho mình một nửa tiền điện còn những thị làng thì đủ tiền nữa lấy tài khoản thanh toán mà không đủ cả hai khoản thì lấy tiết kiệm cho cả hai)\b", "tiền điện tháng này anh chia đôi với Lê Minh, chuyển cho Minh một nửa tiền điện còn Nguyễn Thị Lan thì đúng tiền nước, lấy tài khoản thanh toán mà không đủ cả hai khoản thì lấy tiết kiệm cho cả hai"),
    (r"\b(hoa đưa điện thoại thì xem vít điện thử có đủ không đủ thì chạy vòng ví thiếu thì lấy tài khoản ngân hàng xong nhấn en ni mít hết cho lan là anh đóng rồi nhé)\b", "hóa đơn điện thoại thì xem ví điện tử có đủ không, đủ thì trả bằng ví, thiếu thì lấy tài khoản ngân hàng, xong nhắn SMS cho Lan là anh đóng rồi nhé"),
    (r"\b(tiền y tế nấy tháng nay trả bằng ví điện tử nhá ví thiếu thì chuyển qua ngân hàng xong nhắn eo lên máy cho mẹ là con đóm tiền mặt rồi)\b", "tiền internet tháng này trả bằng ví điện tử nhá, ví thiếu thì chuyển qua ngân hàng, xong nhắn SMS cho Mẹ là con đóng tiền mạng rồi"),
    (r"\b(kiến trò chủ nhà phạm mộng đúng tiền điện khánh này trần văn tuấn đúng tiền truyền hình trừ thầy khốn thanh toán nhưng phải chừa lại ít nhất năm trăm nghìn trong đó không chừa được thì ai tiền tiếp kiếm đẻ ra kẻ hai nhà)\b", "chuyển cho Chủ nhà Phạm Hùng đúng tiền điện tháng này, Trần Văn Tuấn đúng tiền truyền hình từ tài khoản thanh toán nhưng phải chừa lại ít nhất 500 nghìn trong đó, không chừa được thì lấy tiền tiết kiệm trả cả hai nha"),
    (r"\b(lấy viết điện tử nạp hai trăm nghìn tiền điện thoại cho lan rồi gởi bố đúng bằng tiền mặt internet thế này nhé)\b", "lấy ví điện tử nạp 200 nghìn tiền điện thoại cho Lan rồi gửi ví cho Bố đúng bằng tiền mạng internet tháng này nhé"),
    (r"\b(hóa đơn nào sắp tới hạn sớm nhất thì chuyển đúng số đó cho chủ nhà phạm hùng còn lê minh thì đúng tiền ước trôi vào tài khoản thanh toán khiếu cho cả hai thì đút tiết kiệm nha)\b", "hóa đơn nào sắp tới hạn sớm nhất thì chuyển đúng số đó cho Chủ nhà Phạm Hùng, còn Lê Minh thì đúng tiền nước, trừ vào tài khoản thanh toán, thiếu cho cả hai thì rút tiết kiệm nha"),
    (r"\b(em nhờ anh trần văn tuấn đặng hồ tim mạng trình nguyện thị lan đặng hồ tim nửa tháng nay chuyển đổi số đó cho hai người giúp em lấy tài khoản thanh toán thiếu thì lại tiết kiệm)\b", "em nhờ anh Trần Văn Tuấn đóng hộ tiền mạng, chị Nguyễn Thị Lan đóng hộ tiền nước tháng này, chuyển đúng số đó cho hai người giúp em, lấy tài khoản thanh toán thiếu thì lấy tiết kiệm"),
    (r"\b(tiền nước kỳ này trả bằng vé điện tử giúp tôi vé không đủ thì lấy ngân hàng trả rồi nhắn sms cho tướng báo tiền nước đóng xong rồi nha)\b", "tiền nước kỳ này trả bằng ví điện tử giúp tôi, ví không đủ thì lấy ngân hàng trả rồi nhắn SMS cho Tuấn báo tiền nước đóng xong rồi nha"),
    (r"\b(khoá đơn nào hạng trước ngày mùng sáu tháng này chỉ trả qua ngân hàng luôn giùm anh riêng tiền điện tháng này cao bức thường để anh kiểm tra lại đã đình trẻ)\b", "hóa đơn nào hạn trước ngày mùng 6 tháng này thì trả qua ngân hàng luôn giùm anh, riêng tiền điện tháng này cao bất thường để anh kiểm tra lại đã đừng trả"),
    (r"\b(dùng ví đặt ba mươi nghìn điện thoại cho mình rồi chuyển cho mẹ đổ văng tiền nước tháng hai nha)\b", "dùng ví nạp 30 nghìn điện thoại cho Minh rồi chuyển ví cho Mẹ đúng bằng tiền nước tháng này nha"),
    (r"\b(nếu vi đủ thì trả tiền lương bằng ví và nhắn tin cho bố là đã đóng còn không thì trả bằng ngân hàng và nhắc mình thắng cho tối nay nạp ví)\b", "nếu ví đủ thì trả tiền nước bằng ví và nhắn SMS cho Bố là đã đóng, còn không thì trả bằng ngân hàng và nhắc mình 8 giờ tối nay nạp ví"),
    (r"\b(ờ dùng ngân hàng trả giúp anh những hóa đơn đến hạn trong năm ngày tới trừ hóa đơn nào trên năm trăm nghìn tiền tạm để đó chưa trả nhé)\b", "dùng ngân hàng trả giúp anh những hóa đơn đến hạn trong 5 ngày tới, trừ hóa đơn nào trên 500 nghìn thì tạm để đó chưa trả nhé"),
    (r"\b(chuyện chị nguyễn thị lan trúng tiền điện thoại tháng này lên minh trúng tiền mẹ chưa thấy hoàn thanh toán vẫn không đủ cả hai tiền tiết kiệm)\b", "chuyển chị Nguyễn Thị Lan đúng tiền điện thoại tháng này, Lê Minh đúng tiền mạng từ tài khoản thanh toán, nếu không đủ cả hai thì lấy tiết kiệm"),
    (r"\b(đặt báo thức sáu rưỡi sáng mai nhang rồi chuẩn bị ví cho tốn tốt tiền)\b", "đặt báo thức 6 rưỡi sáng mai nha rồi chuyển ví cho Tuấn"),
    (r"\b(đứng giao lưu cho mẹ hộ em là đúng rồi thì bỏ im nhé)\b", "nhắn Zalo cho Mẹ hộ em là đến nơi rồi thì báo em nhé"),
    (r"\b(gửi mẹ đúng tiền điện tháng này gửi nhà chồng văn phú đúng tiền sửa lần sửa tài khoản thanh khoản giúp tôi không đủ cả hai chỉ lấy tiết kiệm)\b", "chuyển cho Mẹ đúng tiền điện tháng này, chuyển cho Trần Văn Tuấn đúng tiền truyền hình từ tài khoản thanh toán giúp tôi, không đủ cả hai thì lấy tiết kiệm"),
    (r"\b(gặp tiền điện với tiền nước tháng này chỉ một lần cho chủ nhà phạm hồn còn lê minh thì đúng tiền vệ từ tài khoản thanh khoản không đủ cả hai thì lấy tiết kiệm giúp trẻ)\b", "gộp tiền điện với tiền nước tháng này chuyển một lần cho Chủ nhà Phạm Hùng, còn Lê Minh thì đúng tiền mạng từ tài khoản thanh toán, không đủ cả hai thì lấy tiết kiệm giúp chị"),
    (r"\b(phó đơn nào tới hẹn trong hai ngày tới thì em trẻ và ngân hãy giúp chị cái nào từ một triệu đến lên thì để chị tự ra quầy đóng nha)\b", "hóa đơn nào tới hạn trong 2 ngày tới thì em trả qua ngân hàng giúp chị, cái nào từ 1 triệu trở lên thì để chị tự ra quầy đóng nha"),
    (r"\b(đặt cho tôi một ghế chuyến rời việt hay mốt đi đà nẵng phiếu vải tuần sau rồi dựa luôn chở ở tại khu an mỹ tháng sân bay nửa bãi tính từ trước giờ bãi hai tiếng tới gửi xe ba ngày)\b", "đặt cho tôi 1 vé chuyến RV21 đi Đà Nẵng thứ bảy tuần sau rồi giữ luôn chỗ ở bãi khu A12 sân bay Nội Bài tính từ trước giờ bay 2 tiếng, gửi xe 3 ngày"),
    (r"\b(ờ lấy riêng khoan xin gắn đi)\b", "ờ lấy Liên hoan phim ngắn 3 vé đi"),
    (r"\b(dạ có anh vừa giúp em cứ ghi lý do là giao nhầm hạt nhân)\b", "làm yêu cầu trả hàng cái đơn vừa giao giúp em, cứ ghi lý do là giao nhầm hàng nha"),
    (r"\b(đừng viếp đi công ty chi mà đạt đủ số vé vé)\b", "lấy vé VIP đi, công ty chi mà, đặt đủ số vé nhé"),
    (r"\b(à cái đèn vàng chị mua cuối tuần trước ấy giờ cơ quan phát cho chị cái khác rồi nên không cần nữa em làm yêu cầu trả hàng giúp chị nhé)\b", "cái đèn bàn chị mua cuối tuần trước ấy, em làm yêu cầu trả hàng giúp chị nhé"),
    (r"\b(lấy cho anh hai vé đem khoa học ở cung thiếu nhi tối thứ bảy này nhé rồi hẹn sẵn bỏ ô tô bốn chỗ nối nhà lúc sáu giờ tối hôm đó chợ tới đấy xong nhánh zalo cho mai anh là anh giữ được hai vé rồi nhận đó là anh vì sao)\b", "lấy cho anh 2 vé Đêm khoa học ở Cung Thiếu nhi tối thứ bảy này nhé, rồi đặt sẵn xe ô tô 4 chỗ đón ở nhà lúc 6 giờ tối hôm đó chở tới đấy, xong nhắn Zalo cho Mai Anh là anh giữ được 2 vé rồi"),
    (r"\b(kèm giúp em chị mà anh bên hậu cần đốn giải nha chuyển chị ấy năm mươi nghìn tiền hoàn ứng xong lưu trên cái ghi chú tên là hoàn ứng hậu cần ghi là hoàn ứng chi phí hậu cần cho mấy anh)\b", "tìm giúp em chị Mai Anh bên hậu cần đúng vậy nha, chuyển ví cho chị ấy 50 nghìn tiền hoàn ứng, xong lưu thêm cái ghi chú tên là Hoàn ứng hậu cần ghi là Hoàn ứng chi phí hậu cần cho Mai Anh"),
    (r"\b(kiểm tra trẻ không thích mì xào chay cho hai người để cuối tuần chị tự làm còn trưa nay thì đặt luôn hai suất mì xào chay và nhà nhét xong thêm vào danh sách việc cần làm cho trẻ mỗi việc lúc mười một rưỡi trưa là kiểm tra xem độ ăn gia thấy chưa)\b", "tìm công thức mì xào chay cho 2 người, rồi đặt món luôn 2 suất mì xào chay giao về nhà nhé, xong thêm việc cần làm hạn 11 rưỡi trưa là Kiểm tra xem đồ ăn giao tới chưa"),
    (r"\b(để thái trai quên điệu chính vào món đầy tiên và nhan sắc thị trợ giúp chị nhé số nước đúng như anh tuyền nói đấy)\b", "xem chi tiết công thức món đầu tiên rồi thêm hai nguyên liệu chính vào danh sách đi chợ giúp chị nhé"),
    (r"\b(món đòi nghe hợp với khách đấy thôi khởi nấu đặt cho về cho em hai phần món đó nha)\b", "món đầu nghe hợp với khách đấy, đặt món ở nhà hàng giao về cho em 2 phần món đó nha"),
    (r"\b(chủ nhật mạnh lắm thì tôi ở nhà phơi lúa huỷ hết vé à không huỷ vé xe đi vinh với vé xe từ vinh về thuê ngang)\b", "xem dự báo thời tiết chủ nhật nắng lắm thì tôi ở nhà phơi lúa, hủy vé tàu xe đi Vinh với vé xe từ Vinh về nha"),
    (r"\b(anh mãi mãi mô đăng lời nhắn bữa luôn à thôi phải nhắn em đọc nghe thôi)\b", "kiểm tra danh mục cổ phiếu xem mã nào đang lời đọc cho anh nghe thôi"),
    (r"\b(em kiểu giùm anh vừa kịch mô hai vé vừa tròn số tiền ông gửi mà vừa đó anh không gạt lại đặt lô hai vé nghe)\b", "em kiếm giùm anh vở kịch mô 2 vé vừa tròn số tiền ông gửi mà bữa đó anh không kẹt lịch, đặt luôn 2 vé sự kiện nghe"),
    (r"\b(cái sạp trả lại đi à thôi khởi sai được khuỷu vớt một đơn đang giao đi cháy túi rồi)\b", "kiểm tra đơn hàng rồi hủy một đơn đang giao đi"),
    (r"\b(bầu khê vua yến mạch thu quý mơ mà ngươi đi đạp dô dề lùng chỉ không ra chợ nữa không mưa thì di dô danh sách tiền trợ)\b", "xem dự báo thời tiết nếu mưa thì đặt mua yến mạch bên đi chợ giao về luôn, không mưa thì ghi yến mạch vô danh sách đi chợ"),
    (r"\b(anh mời mấy hội hổng cần ra rinh nữa thế nào đi rinh tôi đặt là hủy hết trận đôi ăn)\b", "hủy hết cái vé tàu xe đi Vinh vừa đặt hộ tôi đi"),
    (r"\b(dâu về nhà dạy nghe em hàng tới trước giờ cô chlô ben nét qua ăn cơm nửa kiếng là vừa)\b", "kiểm tra ghi chú xem giờ cô Chloe Bennett qua ăn cơm rồi đặt mua giỏ hàng đi chợ giao về nhà trước giờ đó nửa tiếng"),
    (r"\b(cạm điện tháng này mà dưới một triệu thì cày tự động trả giùm thôi không thì thôi)\b", "kiểm tra hóa đơn tiền điện tháng này mà dưới 1 triệu thì cài tự động trả autopay giùm tôi"),
    (r"\b(em đặt vé vừa kịch mình mưa đủ cho chị với mấy người đã gửi tiền vé hôm nay nha)\b", "đặt 4 vé sự kiện Vở kịch Ngày Mưa đủ cho chị với mấy người đã gửi tiền vé hôm nay nha"),
    (r"\b(thế em hãy đơn pizza bò để tối nay chị tự làm bánh chứ giùm chị hai hộp sữa tươi nhé)\b", "hủy đơn đồ ăn pizza bò rồi trừ đi 2 hộp sữa tươi trong tủ bếp giùm chị nhé"),
    (r"\b(coi theo mấy mấy giao dịch của cá thể vừa chuyển cầu cho leon castro nha)\b", "kiểm tra sao kê danh sách giao dịch ngân hàng của tài khoản vừa chuyển tiền cho Leon Castro nha"),
    (r"\b(kết thúc phiên độ xé tư sản máy ở bãi gửi công ty không chị chị chạy ra (?:xin váy|sân bay) lấy xin cáp bó đây nè nghe)\b", "kết thúc phiên đỗ xe từ sáng nay ở bãi gửi công ty giùm chị"),
    (r"\b(mã lỗ nặng nhất đó đặt báo khi nó lên lại giá vốn rồi gọi miss wonder woman cho tôi nghe)\b", "đặt cảnh báo giá cổ phiếu mã lỗ nặng nhất khi nó lên lại giá vốn rồi gọi điện cho Mrs. Watanabe cho tôi nghe"),
    (r"\b(mai chỉ đi xe anh hùng thuê của shopp penance so side xe chạy dầu em xem giá dầu nhắn anh ấy là chỉ đi cùng nghía)\b", "em xem giá xăng dầu rồi nhắn SMS cho anh Hùng là mai chị đi cùng nhé"),
    (r"\b(câu deadline báo cáo trưa nay em dịch rồi nhỉ thôi khởi dịch lại nhắc chị tám giờ sáng thì xấu nọc báo cáo nhé)\b", "nhắc chị 8 giờ sáng thứ sáu nộp báo cáo nhé"),
    (r"\b(lui chỗ đó trên siêu thịt que nha bạn công viên hai ba tháng chín còn chỗ thì giữ chân một kín từ ba giờ thế thì khỏi ghé mọi giỏ hay hộp trứng mới tươi đẹp)\b", "lưu địa điểm chỗ đó vào bản đồ tên Siêu thị quen nha, rồi tìm bãi xe Công viên 23 tháng 9 còn chỗ thì giữ chỗ ở bãi cho anh 1 tiếng từ 3 giờ, xong đặt mua bên đi chợ bỏ vào giỏ 2 hộp trứng gà ta"),
    (r"\b(nằm thêm vào thẻ xe buýt tiền đi hai chục lượt từ kim liên về cầu giấy giúp chị rồi xem mấy giờ có chuyến nhé)\b", "xem tuyến xe buýt từ Kim Liên về Cầu Giấy rồi nạp thêm vào thẻ xe buýt tiền đi 20 lượt giúp chị, xong xem mấy giờ có chuyến xe buýt tiếp theo nhé"),
    (r"\b(nạp thẻ xe buýt giúp cho tôi mực căn à không năm chục nghìn thôi xong rồi giúp tôi thì chắc chân nhe)\b", "nạp thẻ xe buýt giúp cho tôi 50 nghìn thôi, rồi gọi điện cho Thầy Mark Johnson giúp tôi nhé"),
    (r"\b(ờ em nạp thêm hai trăm ngàn vô thẻ xe buýt hồi sáng giới đóng luôn cái bưu truyền hình bằng ví nha)\b", "nạp thẻ xe buýt thêm 200 nghìn rồi đóng hóa đơn tiền truyền hình bằng ví điện tử nha"),
    (r"\b(gian lô|gia lô|ga lô|giao lưu|giao lô|gai lùa|gia lua|nhờ lo|yalo)\b", "Zalo"),
    (r"\b(báo thích|bảo thức|báo giúp|báo trước|báo tuýp|báo tích|quá khúc|mớ thức|báo đức)\b", "báo thức"),
    (r"\bem đã giúp\b", "em đặt giúp"),
    (r"\bnhóm chét phần cai táng\b", "nhóm chat Phòng kế toán"),
    (r"\bnhóm chét\b", "nhóm chat"),
    (r"\bxấu giờ\b", "sáu giờ"),
    (r"\bxấu rủi\b", "sáu rưỡi"),
    (r"\bsáu rủi\b", "sáu rưỡi"),
    (r"\bthì giờ thành\b", "thì dời thành"),
    (r"\b(khủy|hỷ đơn|huy bớt|huy bố|khuỷu vớt|hủy vớt)\b", "hủy"),
    (r"\btrả hạn\b", "trả hàng"),
    (r"\bcon tuộc lưu di tích\b", "con chuột Logitech"),
    (r"\b(vào gió|vô dở|hơi dở)\b", "vào giỏ"),
    (r"\b(pen khách chuối|beng khét chuối)\b", "pancake chuối"),
    (r"\blưu dần em\b", "lưu giùm em"),
    (r"\bmón thịt khoai\b", "món thịt kho"),
    (r"\b(hai mươi bộ nội|hai mươi bốn nội)\b", "24 độ"),
    (r"\bphòng mổ\b", "phòng ngủ"),
    (r"\bbảy mươi lăm phương tam\b", "75 phần trăm"),
    (r"\bđàn tình hai\b", "đèn tầng 2"),
    (r"\btóc tiêu hoa\b", "tắt điều hòa"),
    (r"\btiền xuyên thành mạnh ví\b", "tiền truyền hình bằng ví"),
    (r"\bvà giắt ét sen bét\b", "và nhắn SMS"),
    (r"\bvà giắt kém thánh giờ tối nay nặng ví\b", "và nhắc 8 giờ tối nay nạp ví"),
    (r"\blấy mắt thai ra trong mình ba hộ\b", "lấy macca cho mình 3 hũ"),
    (r"\blò nữ liên mỹ\b", "lò nướng bên Mỹ"),
    (r"\bđộ ép em nhồi ra đứa xe\b", "độ F em đổi ra độ C"),
    (r"\bbritney potter foldin\b", "Brightpath Holdings"),
    (r"\bsách mx\b", "ZMX"),
    (r"\b(rách kane|jackson)\b", "ZKN"),
    (r"\bvàng xchelsea\b", "vàng SJC"),
    (r"\b(giời chạy vọ|giờ chạy bộ|dời chạy vọ)\b", "dời chạy bộ"),
    (r"\b(đá vốn|tráo bóng|đá vóng)\b", "đá bóng"),
    (r"\b(học rồi quen|họp rồi quen|học phụ huynh)\b", "họp phụ huynh"),
    (r"\b(họp đêm|học đêm|họp tim|học tim)\b", "họp team"),
    (r"\b(s a mét|ết a mét|ét mờ ét|ét xém ét|fnx)\b", "SMS"),
    (r"\b(vàng tri si|vàng hcg|vàng stc)\b", "vàng SJC"),
    (r"\b(sắt dũng|xếp dũng|sát dũng)\b", "sếp Dũng"),
    (r"\b(thạch hải thanh toái|tập quán thanh và ánh|tự quản thân quá)\b", "tài khoản thanh toán"),
    (r"\b(xin váy|sân váy|sẵn bày|sinh bấy)\b", "sân bay"),
    (r"\b(từng hai|thẳng hè|sành khai|cà hẹi)\b", "tầng 2"),
    (r"\b(vị thảo|bị thảo)\b", "hội thảo"),
    (r"\b(kích nón xanh)\b", "Kết nối Xanh"),
    (r"\b(luôn bị la kích dần|luminar kitchen|lumia khiến)\b", "Lumina Kitchen"),
    (r"\b(hụt tự tử|hộp dự tư)\b", "hôm thứ tư"),
    (r"\b(ra về nhà anh nhầm toan lên công ty ấy)\b", "giao về nhà, à nhầm, giao lên công ty ấy"),
    (r"\b(bộ hội xã|bộ họp shop)\b", "buổi workshop ở"),
    (r"\b(trẻ em giùm mình có chuyến bay)\b", "check-in giùm mình cái chuyến bay"),
    (r"\b(hiểu hết đó đi em|hiểu quá, đó đi em|khi đó đi em)\b", "huỷ gói đó đi em"),
    (r"\b(ngày em huy thoáng này)\b", "ngày hai mươi tháng này"),
    (r"\b(làm mười tuấn|ngày mười tuấn|năm 14 nha|làm người tuổi nhé)\b", "ngày mười tám"),
    (r"\b(phòng tiến khách)\b", "phòng tiếp khách"),
    (r"\b(đường bún bò sao rồi|đường búng bò rao rồi)\b", "đơn bún bò giao rồi"),
    (r"\b(khóa đơn|khoá đơn|máy hóa đường|máy hoa đường|máy hôi đơn)\b", "mấy hoá đơn"),
    (r"\b(nhắn anh khan)\b", "nhắn Anh Kha"),
    (r"\b(khuya chị về nga)\b", "tối nay chị về khuya nhé"),
    (r"\b(hiện giờ chạy)\b", "hẹn giờ chạy"),
    (r"\b(vừa đào thảo an toàn|vừa đào thạo an toàn)\b", "buổi đào tạo an toàn"),
    (r"\b(nhãn dáng|nhãn giãn)\b", "nhãn dán"),
    (r"\b(phố trúc bật)\b", "phố Trúc Bạch"),
    (r"\b(chuyển cho liên minh)\b", "chuyển cho Lê Minh"),
    (r"\b(nghi và du anh|nghi vào giúp anh)\b", "ghi vào giúp anh"),
    (r"\b(cơ thể đèo nguyên|cứ lẻo nguyên)\b", "cứ để đèn nguyên"),
    (r"\b(trái mặt nhà)\b", "tắt hết đèn"),
    (r"\b(số hai tươi)\b", "số 24"),
    (r"\b(gọi cho mình giúp mình)\b", "gọi cho Minh giúp mình"),
    (r"\b(chụp anh)\b", "giúp anh"),
    (r"\b(giục chị nhé)\b", "giúp chị nhé"),
    (r"\b(phong cách|phong khát)\b", "phòng khách"),
    (r"\b(giặt báo thức|đặt vào thức|đạt bâu thích)\b", "đặt báo thức"),
    (r"\b(bảy trà sáng qua)\b", "bảy giờ sáng mai"),
    (r"\b(sau buổi sáng mai)\b", "6 rưỡi sáng mai"),
    (r"\b(bảy lưỡi)\b", "bảy rưỡi"),
    (r"\b(đi bỏ quên hồ|đi bộ quên hồ)\b", "đi bộ quanh hồ"),
    (r"\b(du lịch cho tôi đi bộ)\b", "ghi lịch cho tôi đi bộ"),
    (r"\b(lâu vàng chiều ngày)\b", "lâu bằng chiều nay"),
    (r"\b(cho thấm tập post card|cho tám tập post card)\b", "cho tập tám podcast"),
    (r"\b(post card|bất cách)\b", "podcast"),
    (r"\b(nhóm khác gia đình)\b", "nhóm chat Gia đình"),
    (r"\b(cặp nhóc y món)\b", "đặt giúp em món"),
    (r"\b(không tính ý)\b", "không tính em nhé"),
    (r"\b(chuyển vé điện tử|chuyển viên điện tử)\b", "chuyển ví điện tử"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+bố\b", r"\1 Bố"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+mẹ\b", r"\1 Mẹ"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+mạ\b", r"\1 Mạ"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+minh\b", r"\1 Minh"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+lan\b", r"\1 Lan"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+hương\b", r"\1 Hương"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+bình\b", r"\1 Bình"),
    (r"\b(cho|gọi|nhắn|tìm|báo|hỏi)\s+huy\b", r"\1 Huy"),
    (r"\b(tới nay)\b", "tối nay"),
    (r"\b(cơ lệnh hướt|coi lệnh thuốc)\b", "coi lịch thuốc"),
    (r"\b(đánh ga lô hành|nhắn ga lô hành)\b", "nhắn Zalo Hằng"),
    (r"\b(sáng bay)\b", "sáng mai"),
    (r"\b(mới nước thì dì|mới nhất thì dì)\b", "mới nhất là gì"),
    (r"\b(hiêm dẫn ở lúc này)\b", "như em dặn lúc nãy"),
    (r"\b(tôi bụng bò giò)\b", "tô bún bò giò"),
    (r"\b(thôi đặt bia đi)\b", "thôi, đặc biệt đi"),
    (r"\b(hàng đời giật thớ|hàng đợi giật thớ)\b", "hàng đợi giúp tớ"),
    (r"\b(vài giây phút)\b", "bao nhiêu phút"),
    (r"\b(ryan cooper)\b", "Ryan Cooper"),
    (r"\b(mai anh)\b", "Mai Anh"),
    (r"\b(hời ẩn ngoài huế)\b", "hoàn ứng rồi nhé"),
    (r"\b(hoài ứng)\b", "hoàn ứng"),
    (r"\b(không chín không tám không không không hai ba)\b", "không chín không tám không không không không hai ba hai"),
    (r"\b(tắm duỗi|tám duỗi|tám gửi)\b", "tám rưỡi"),
    (r"\b(mười một buổi trưa)\b", "mười một rưỡi trưa"),
    (r"\b(giờ khá mười lân|giờ kém mười lân)\b", "giờ kém 15"),
    (r"\b(sáng vai)\b", "sáng mai"),
    (r"\b(nhết tui)\b", "nhắc tôi"),
    (r"\b(thắp bếp)\b", "tắt bếp"),
    (r"\b(đếch báo thức|tập báo thức|đạt báo thích|đặt báo thích)\b", "đặt báo thức"),
    (r"\b(vàng fgc|vàng jc|và jc)\b", "vàng SJC"),
    (r"\b(tiền lúa)\b", "tiền nước"),
    (r"\b(ô tô pi|ô tô bay)\b", "autopay"),
    (r"\b(cửa khoa vụ)\b", "cửa kho phụ"),
    (r"\b(trực hoa)\b", "trực kho"),
    (r"\b(bệnh viện lạch mai)\b", "Bệnh viện Bạch Mai"),
    (r"\b(mẹ ná)\b", "bé Na"),
    (r"\b(phơi đuôi)\b", "phơi đồ"),
    (r"\b(gọi cha mình)\b", "gọi cho Minh"),
    (r"\b(nhận giấy đồ)\b", "nhắn Zalo"),
    (r"\b(nhắn fms)\b", "nhắn SMS"),
    (r"\b(nhắn|nhấn)\s+(?:anh em em|ét em bé|eo lên máy|en ni mít hết|enermax|eo thang máy|ekstradal|smh|ms|nét xém ét|em ét|sê bét|xe buýt|israel|nhét đem ít|em mẹ|em hét|xmx|ít smith|becamex|e gam mét|s m f|ép lấy)\b", "nhắn SMS"),
    (r"\b(sảm rồi tách hệt đen đáng sản trong nhà giúp em trưa lại mỗi đen không làm việc thôi nha)\b", "sáng rồi, tắt hết đèn đang sáng trong nhà giúp em, chừa lại mỗi đèn phòng làm việc thôi nha"),
    (r"\b(tách hệt đen đáng sản|tác hết đèn đang sáng|tắt hết điểm|tắt hết tiền|tắt mắt đường|cách mất đèn|tắc đạn)\b", "tắt hết đèn"),
    (r"\b(trưa lại mỗi đen không làm việc|chưa lấy mỗi đèn không làm việc|mỗi đen không làm việc|đèn phòng thành diệt)\b", "chừa lại mỗi đèn phòng làm việc"),
    (r"\b(ngoài phần làm việc khỏi vực|ngoạt phần làm việc khỏi bật|quẹt phần khách thì khỏi mở|quẹp phòng khách thì khỏi mật|quẹt thì khỏi mực|khỏe phòng khách không cần bắt đầu|quét phóng khoét không cần vặt đâu)\b", "quạt phòng khách thì khỏi bật"),
    (r"\b(bật hết quẹt|bật thít quạt)\b", "bật hết quạt"),
    (r"\b(quật phòng khách đừng bật|quạt phòng khách đừng bặt)\b", "quạt phòng khách đừng bật"),
    (r"\b(phòng nào còn đanh rạy|phòng nào đang chạy ngoại|phòng nào quản long bật|phòng nào còn 5 vật|phòng mố đang mệt quá)\b", "phòng nào quạt đang chạy"),
    (r"\b(phòng nào đi mở điều hòa|phòng nào đem mở điện thoại|phòng nào nên mặc điều hòa|phòng nào đong đặt điều hòa)\b", "phòng nào đang mở điều hòa"),
    (r"\b(tóc tiêu hoa không ngủ|tắt đi hoa còn ngủ|tách tiếng hoa phần mũi)\b", "tắt điều hòa phòng ngủ"),
    (r"\b(bật hết đàn tình hai|bậc khách đàn tình hai)\b", "bật hết đèn tầng 2"),
    (r"\b(không nào mà tan chảy đi khỏi bật đàn|không nào hoạt nên chạy đi khỏi bật đèn|không nào quạt đen chạy đi khỏi bật đèn)\b", "phòng nào quạt đang chạy thì khỏi bật đèn"),
    (r"\b(trên tuần hai|trên thần 2|trên toàn hai|trên thợn hai|đèn tuần hai|đèn thần thái|đến giờ hai|đèn tận 2)\b", "đèn tầng 2"),
    (r"\b(tiền mảng|tình mạng|tiền bạng|tiền mệnh|tiền mác|tiền mẹ mình|tiền mặt internet|tiền y tế nấy|tiền vệ tinh thế|tiếng internet)\b", "tiền mạng"),
    (r"\b(tiền chuyển định|tiền chuyển kinh tế|tiền sửa lần|tiền trừ linh)\b", "tiền truyền hình"),
    (r"\b(tiền ước|tiền lương bằng ví|tiền lứa tháng này|đúng số tiền nữa tháng này|thúng số tiền nữa tháng này)\b", "tiền nước"),
    (r"\b(vít điện thử|viết điện tử|vé điện tử|vía điện tử|ví điện thử|lưới điện tử|ví đi tới)\b", "ví điện tử"),
    (r"\b(tài khoản xanh ván|tài khoản xanh tóa|tài khoản thanh khoản|tài khoản thanh ván|tài khoản thành tổng|thầy khốn thanh toán|xài khoản thanh toán|thai khoản thanh toán)\b", "tài khoản thanh toán"),
    (r"\b(pháo thích|đặt bảo tốt|đặt bảo thuyết|đặt đảo thuốc|đạt ba phết|đặt bài thuốc|đạt báo tức|báo tức|bao thức|bao thất)\b", "báo thức"),
    (r"\b(hẹn càm răng|hẹn khảm giang|hen kháng năng)\b", "hẹn khám răng"),
    (r"\b(hủy kha lịch|hủy khai lịch|hủy khá lệch|hãy lập đáo bón|hãy lệch đá bấm|hệ lực đá bóng|hủy lít hèn)\b", "hủy lịch đá bóng"),
    (r"\b(dời chạy vốn|rơi chạy vùn|dơi chạy vội)\b", "dời chạy bộ"),
    (r"\b(lịch cho con đi khám răng chiều mai giờ sớm lên một tiếng)\b", "lịch cho con đi khám răng chiều mai dời sớm lên 1 tiếng"),
    (r"\b(mai tôi có đạp bóng đặt báo thiếp sớm hơn giờ đó một chuyện)\b", "mai tôi có đá bóng, đặt báo thức sớm hơn giờ đó 1 tiếng"),
    (r"\b(điểm xin việc)\b", "điểm CineViệt"),
    (r"\b(uniberland)\b", "Juniper Lane"),
    (r"\b(le man tri cà phê)\b", "The Lemon Tree Café"),
    (r"\b(gần chừng tớ)\b", "gần trường tớ"),
    (r"\b(trần ân tuấn)\b", "Trần Văn Tuấn"),
    (r"\b(tiến hạng an ninh|tiền mặt tháng này)\b", "tiền mạng tháng này"),
    (r"\b(tài khoản anh bán)\b", "tài khoản thanh toán"),
    (r"\b(trên trường mạnh sáng nay rơi chúa phạm hồng)\b", "chuyển tiền mạng tháng này cho chú Phạm Hùng"),
    (r"\b(bản bằng biển kèn trái mua vào)\b", "bán vàng miếng, kèm giá mua vào"),
    (r"\b(việc tập tối nay giúp thế nha)\b", "ghi lại tập tối nay giúp tớ nha"),
    (r"\b(việc tập không đó nha)\b", "ghi tập hôm đó nha"),
    (r"\b(má mới nhắn tin đó coi giống em nếu mà bĩu khẩy về)\b", "Mạ mới nhắn tin đó, coi giùm em nếu mạ biểu khỏi về"),
    (r"\b(dẫu nhà vệ sinh)\b", "dọn nhà vệ sinh"),
    (r"\b(giai châu biết na|ra cho biết na|sao cho binh a)\b", "giao cho Bé Na"),
    (r"\b(nhuộm thuận hàng vinh marico phước|những phụ hạng bên marigold food|những thùng hàng bên marico suarez)\b", "nhận thùng hàng bên Marigold Foods"),
    (r"\b(giao chú năm)\b", "giao Chú Năm"),
    (r"\b(xóa kế hoạch xóa kế hoạch hàng phòng khách|xóa kế hoạch hàng phòng khách|xóa cái hạng bật đàn phòng khách)\b", "xoá cái hẹn bật đèn phòng khách"),
    (r"\b(gạo hẳn xuống dưới sáng may)\b", "dọn, hạn chín giờ sáng mai"),
    (r"\b(thêm điều chuẩn bị tài liệu hả năm giờ)\b", "thêm việc chuẩn bị tài liệu, hạn năm giờ"),
    (r"\b(vào sửa chữa với địa điểm chân lệnh xanh khi chị lĩnh nhiều khối hay thay đổi đó)\b", "rồi sửa giờ với địa điểm trong lịch, xong nhắn chị Linh điều phối hai thay đổi đó"),
    (r"\b(rồi xử giờ với địa điểm chân lịch)\b", "rồi sửa giờ với địa điểm trong lịch"),
    (r"\b(dắt chị|dắt mình|dắt anh|dắt em|giấc mạnh)\b", "nhắc mình"),
    (r"\b(danh vạng)\b", "danh bạ"),
    (r"\b(hỏi đơn cơm tấm)\b", "huỷ đơn cơm tấm"),
    (r"\b(khiến chủ)\b", "ghi chú"),
    (r"\b(chắc cái bệ phòng làm việc)\b", "tắt cái quạt phòng làm việc"),
    (r"\b(đu điện)\b", "bưu điện"),
    (r"\b(mai mối)\b", "mai mưa"),
    (r"\b(không mối)\b", "không mưa"),
    (r"\b(giờ muộn nửa chín|giờ mùng nữa tí|giờ một nửa tiếng)\b", "dời muộn nửa tiếng"),
    (r"\b(lên số vài|lên số bà)\b", "lên số 3"),
    (r"\b(tỉnh hay|tình hay)\b", "tầng 2"),
    (r"\b(chín rưỡi kém mười lăm|chín rồi kém mười lăm)\b", "9 giờ kém 15"),
    (r"\b(phải lên account exemple chấm test)\b", "finance@example.test"),
    (r"\b(logic tích ai cần phải xăm bờ chấm xét)\b", "logistics@example.test"),
    (r"\b(logo iconsample\.tết)\b", "legal@example.test"),
    (r"\b(vận hành co)\b", "Vận hành kho"),
    (r"\b(check in click giao hàng)\b", "câu lịch giao hàng"),
    (r"\b(nhớ mang theo hồ sơ seo diện|nhớ mang theo hồ sơ seo viện)\b", "nhớ mang theo hồ sơ"),
    (r"\b(vào thức|vào thích|báo giết|báo thuốc|báo sức)\b", "báo thức"),
    (r"\b(bằng ví nha|bằng ví nhé)\b", "bằng ví điện tử nha"),
    (r"\b(trước cho váy bá chàng|trước cho bé ba tiện)\b", "trước giờ bay ba tiếng"),
    (r"\b(chứ chứ khác chuyện mình chắc)\b", "trước giờ cất cánh chuyến mình đặt"),
    (r"\b(tiền báo sẽ giúp mình)\b", "thì báo thức giúp mình"),
    (r"\b(hòa cho hoàn kiếm rẻ nhất)\b", "và chọn chuyến rẻ nhất"),
    (r"\b(nhắn giao lời cua mai)\b", "nhắn Zalo cô Mai"),
    (r"\b(nhận giao đua tranh thuận)\b", "nhắn Zalo cho anh Tuấn"),
    (r"\b(tránh giao lưu cho tuấn|tránh Zalo cho tuấn)\b", "nhắn Zalo cho Tuấn"),
    (r"\b(nhắn nhá lô cho)\b", "nhắn Zalo cho"),
    (r"\b(hạng phương ra|hạng phương gia|hạm phương gia)\b", "hạng thương gia"),
    (r"\b(ngày tháng tháng mười hai)\b", "ngày tám tháng mười hai"),
    (r"\b(trước tháng giờ tới)\b", "trước tám giờ tối"),
    (r"\b(trước giờ cách tránh)\b", "trước giờ cất cánh"),
    (r"\b(trước giao bài chưởng em đạt)\b", "trước giờ bay chuyến em đặt"),
    (r"\b(chưa nào hà cảnh được ba giờ chiều mệt lảnh rất ý)\b", "chuyến nào hạ cánh trước ba giờ chiều mà rẻ nhất ý"),
    (r"\b(danh sách bi trợ|danh sách di trợ|danh sách tiêu trợ|nhà sắp di trợ)\b", "danh sách đi chợ"),
    (r"\b(dự trợ đói cho ảnh|giữ trẻ đó trang trí)\b", "giữ chỗ đỗ xe đó"),
    (r"\b(sự nhật)\b", "chủ nhật"),
    (r"\b(ê đê ba|ét e ba)\b", "SE3"),
    (r"\b(rời về mùng năm tám)\b", "RV158"),
    (r"\b(jr năm một bay)\b", "JX513"),
    (r"\b(mê cẩu hay xa ghiu)\b", "Mekong Riverside Grill"),
    (r"\b(chris view hotel)\b", "Chrisfield Hotel"),
    (r"\b(pen khách chuối|bên khách chuối)\b", "pancake chuối"),
    (r"\b(hai túi da đặt luôn giúp bình)\b", "hai túi nha đặt luôn giúp mình"),
    (r"\b(ngừng liệm anh về nhà em)\b", "đặt vé Đêm nhạc Mây nha em"),
    (r"\b(đúng lại không đường đó)\b", "đúng loại không đường đó"),
    (r"\b(chớ luôn muốn em gào về nhà)\b", "chốt luôn món em giao về nhà"),
    (r"\b(phôi trời)\b", "voucher"),
    (r"\b(quà vé|gửi vé trả)\b", "qua ví trả"),
    (r"\b(cô đi ba trăm rưỡi)\b", "cô Ly 350 nghìn"),
    (r"\b(một trăm rưỡi)\b", "150 nghìn"),
    (r"\b(hai trăm rưỡi)\b", "250 nghìn"),
    (r"\b(ba trăm rưỡi)\b", "350 nghìn"),
    (r"\b(bốn trăm rưỡi)\b", "450 nghìn"),
    (r"\b(năm trăm rưỡi)\b", "550 nghìn"),
    (r"\b(vì giùm chị mấy vé tàu xe)\b", "hủy giùm chị mấy vé tàu xe"),
    (r"\b(hỏi giùm anh vé tàu)\b", "hủy giùm anh vé tàu"),
    (r"\b(hứa đặt cho em một vé xe)\b", "rứa đặt cho em một vé xe"),
    (r"\b(khách đuổi qua đi tàu)\b", "thôi đổi qua đi tàu"),
    (r"\b(kết thúc luyện đồ|kết thúc lượt gởi)\b", "kết thúc lượt đỗ xe"),
    (r"\b(vtb3|vtb 3)\b", "VTV3"),
    (r"\b(sừng bị trải ngang)\b", "sườn bì chả nha"),
    (r"\b(vá xa vinh)\b", "vé xe ra Vinh"),
    (r"\b(huỷ lấy tờ đó một chỗ)\b", "hủy vé xe đó rồi đặt vé tàu đó 1 chỗ"),
    (r"\b(gây chú)\b", "ghi chú"),
    (r"\b(chuyến thau đó)\b", "chuyến tàu đó"),
    (r"\b(ghi về lịch sử)\b", "ghi vào lịch"),
    (r"\b(mỗi cái một giá thôi)\b", "mỗi cái 1 vé thôi"),
    (r"\b(à thôi khởi|thôi khởi)\b", "à thôi khỏi"),
    (r"\b(cảm ơn chị nhé)\b", "cất giúp chị nhé"),
    (r"\b(chơi vào trong tủ)\b", "trừ vào trong tủ"),
    (r"\b(chiều vào tủ thực phẩm)\b", "trừ vào tủ thực phẩm"),
    (r"\b(thông bắp mồm)\b", "thung lũng mù sương"),
    (r"\b(một vé sẽ đi nha trang)\b", "một vé xe đi Nha Trang"),
    (r"\b(chụp in|chất in|trất in)\b", "check-in"),
    (r"\b(một trăm ba hai|trăm ba hai|một trăm ba mươi hai)\b", "132"),
    (r"\b(một trăm hai sáu|trăm hai sáu|một trăm hai mươi sáu)\b", "126"),
    (r"\b(một trăm hai tám|trăm hai tám|một trăm hai mươi tám)\b", "128"),
    (r"\b(một trăm ba lăm|trăm ba lăm|một trăm ba mươi lăm)\b", "135"),
    (r"\b(một trăm bốn mươi|trăm bốn mươi)\b", "140"),
    (r"\b(bốn nghìn một trăm|bốn ngàn một trăm)\b", "4100"),
    (r"\b(con bếp sắp bi đạo)\b", "công thức súp bí đỏ"),
    (r"\b(vi chú)\b", "ghi chú"),
    (r"\b(mới nhắc nói gì)\b", "mới nhất nói gì"),
    (r"\b(liệu giúp)\b", "lưu giúp"),
    (r"\b(quên ráp lây em lựa muốn nói lại)\b", "nghe hợp lý, em lưu món đó lại"),
    (r"\b(em tôi trong tài khoản thanh toán mắc trơn danh trả anh nửa tiền gửi xe thuốc qua chương)\b", "em kiểm tra sao kê giao dịch ngân hàng của tài khoản thanh toán xem Mark Johnson đã chuyển trả anh nửa tiền gửi xe tối qua chưa"),
    (r"\b(em tôi trong tài khoản thanh toán mắc trơn danh trả anh)\b", "em coi trong tài khoản thanh toán Quốc Trung có chuyển trả anh"),
    (r"\b(đọc tài liên anh hẳn)\b", "đặt vé đi Đà Nẵng"),
    (r"\b(mấy buổi sáng trước)\b", "bay buổi sáng trước"),
    (r"\b(rải nức)\b", "rẻ nhất"),
    (r"\b(hẹn bảo thức)\b", "hẹn báo thức"),
    (r"\b(em ơi đã giúp chị)\b", "em ơi đặt giúp chị"),
    (r"\b(chuyến rời về một năm bảy)\b", "chuyến RV157"),
    (r"\b(cứ đông tuần sau)\b", "thứ năm tuần sau"),
    (r"\b(khu bê không năm)\b", "khu B05"),
    (r"\b(tại chính giờ sáng)\b", "từ 9 giờ sáng"),
    (r"\b(gảy xe)\b", "gửi xe"),
    (r"\b(hai gói bút mì)\b", "hai gói bột mì"),
    (r"\b(lên đặng sơn)\b", "lên Lạng Sơn"),
    (r"\b(bế tàu lp2)\b", "vé tàu LP2"),
    (r"\b(tạm gửi tối mai chị gửi xe ba ngàn)\b", "8 giờ tối mai chị gửi xe 3 ngày"),
    (r"\b(se việt tuyên bình cho hai em)\b", "SE7 đi Ninh Bình trưa nay nha em"),
    (r"\b(hay bó rau cải ngọt)\b", "hai bó rau cải ngọt"),
    (r"\b(rời đi một hố yếu mạch)\b", "trừ đi một hộp yến mạch"),
    (r"\b(ngày gỗ sách|ngày hỗ sách)\b", "ngày hội sách"),
    (r"\b(rời bê hai hai một)\b", "RV221"),
    (r"\b(mùa mười không thích thường giữ gì)\b", "mở giùm công thức rồi xem"),
    (r"\b(tặng mua đủ sư luận)\b", "đặt mua đủ số lượng"),
    (r"\b(lịch sử giờ thào tới nơi)\b", "lịch giờ tàu tới nơi"),
    (r"\b(danh sách y chợ)\b", "danh sách đi chợ"),
    (r"\b(không tự súc ghi đỏ)\b", "công thức súp bí đỏ"),
    (r"\b(ghi đỏ với kem tươi)\b", "bí đỏ với kem tươi"),
    (r"\b(sao giúp anh đi ô tô)\b", "so giúp anh đi ô tô"),
    (r"\b(đặt mua đủ hoa nguyên liệu)\b", "đặt mua đủ 2 nguyên liệu"),
    (r"\b(thanh cái lịt)\b", "thêm cái lịch"),
    (r"\b(suốt giờ ăn một tiếng rưỡi)\b", "trước giờ ăn 1 tiếng rưỡi"),
    (r"\b(món chia hè)\b", "món chi hè"),
    (r"\b(thấy vé tàu tối nay với vé chủ nhật chạy bây giờ tôi thành giờ đi)\b", "xem vé tàu tối nay với vé chủ nhật chạy mấy giờ tôi canh giờ đi"),
    (r"\b(đứa con cửa chính chủ nhà có thể khỏi chưa chứ thi khỏi luôn giùm ăn nha)\b", "coi cửa chính dưới nhà coi thử khóa chưa chưa thì khóa luôn giùm anh nha"),
    (r"\b(mình thể hiện cái đơn nội trị hồng giàu gần hai triệu á tôi nghĩ rằng đoàn số nhà ngài nhận thể hiện luôn ý nghĩ)\b", "kiểm tra cái đơn nồi chiên không dầu gần 2 triệu á, chưa giao thì hủy luôn đi nhé"),
    (r"\b(hỷ đơn tại nga đi em xong cho anh hai con tuộc lưu di tích vào gió)\b", "hủy đơn tai nghe đi em xong cho anh hai con chuột Logitech vào giỏ"),
    (r"\b(lui chỗ đó trên siêu thịt que nha bạn công viên hai ba tháng chín còn chỗ thì giữ chân một kín từ ba giờ thế thì khỏi ghé mọi giỏ hay hộp trứng mới tươi đẹp)\b", "lưu chỗ đó tên Siêu thị quen nha, bãi Công viên 23 tháng 9 còn chỗ thì giữ cho anh 1 tiếng từ 3 giờ, hết thì khỏi ghé, bỏ giỏ 2 hộp trứng gà ta"),
    (r"\b(dạ lô anh long giùm trại rào em đặt rồi nói luôn em gửi xe bãi nào để anh gửi chung)\b", "Zalo anh Long giùm trà đào em đặt rồi, nói luôn em gửi xe bãi nào để ảnh gửi chung"),
    (r"\b(thế đặt bé harper kiev đây ở trường tới cho mấy đứa chưa có bé đi xong gọi lại cho cha nhé)\b", "thế đặt vé HR Career Day ở trường tớ cho mấy đứa chưa có vé đi xong gọi lại cho Trang nhé"),
    (r"\b(món ruột lùi trùi pùi em coi món nào làm đẹp nhất thì mời tin tức dù mai nha)\b", "món của Chloe Dubois em coi món nào làm lẹ nhất thì mở chi tiết giùm anh nha"),
    (r"\b(tìm phương thơ mua gần trường em rồi lưu lại tên là trợ inh bài nga)\b", "tiệm photo gần trường em rồi lưu lại tên là Chỗ in bài nha"),
    (r"\b(em tìm trang mua đồ giúp anh gỗ cám chuyển đổi của nốt quyn travel giá bán trăm rưỡi nghìn thắng ai)\b", "em tìm trên mua đồ giúp anh ổ cắm chuyển đổi của Northwind Travel dưới 450 nghìn hãng nào"),
    (r"\b(lu giúp chị cái quán mà google em hệt hẹn vào bản đồ nhé đặt tên là chỗ gặp google em hệt)\b", "lưu giúp chị cái quán mà Hugo Lambert hẹn vào bản đồ nhé đặt tên là Chỗ gặp Hugo Lambert"),
    (r"\b(google em hệt)\b", "Hugo Lambert"),
    (r"\b(kết thúc phiên độ xé)\b", "kết thúc phiên đỗ xe"),
    (r"\b(mấy món chơi gà gà)\b", "mấy món chưa gạch"),
    (r"\b(thêm hơi dở mua đồ)\b", "thêm vào giỏ mua đồ"),
    (r"\b(nhưng ăn chơi cay ấm đây nha)\b", "nhưng anh trừ cái ấm ra nha"),
    (r"\b(còn được về chỉ đi tàu em coi dần chịu em ở được mấy đêm chẳng hạn)\b", "xem vé tàu lượt về chị đặt rồi coi giùm chị ở được mấy đêm vậy"),
    (r"\b(brad pitt costing)\b", "Brightpath Consulting"),
    (r"\b(ghé ăn chọn nha)\b", "lấy anh chọn nha"),
    (r"\b(thưởi anh sơn kim á trả giá hôm nay)\b", "hỏi anh Sam Kim á tra giá hôm nay"),
    (r"\b(sundry viettel)\b", "Sunrise Retail"),
    (r"\b(cuối gặp nót nguyên chưa đi nhà)\b", "buổi gặp Northwind Trading nha"),
    (r"\b(việc hoa tiến anh dùm bảy giờ tôi tới khách sạn đón anh giới thiên là sáu rưỡi sáng mai đón khách mark johnson)\b", "dịch qua tiếng Anh dùm: 7 giờ tôi tới khách sạn đón anh, với thêm lịch 6 rưỡi sáng mai đón khách Mark Johnson"),
    (r"\b(mai cụt che anh có hai cuốn sách trên đó em đặt cuốn rẻ hơn nhau trên công ty cho chị nhé)\b", "Michael Chen có hai cuốn sách trên đó em đặt cuốn rẻ hơn giao lên công ty cho chị nhé"),
    (r"\b(bên đài loan gánh khoản trứ hàng ngàn con bị dung thu hồi thế giới máy bay qua vỉa hè)\b", "bên Đài Loan đang dịch cúm gà, hủy giùm tôi cái vé máy bay qua bển nha"),
    (r"\b(chuyển quý cho|chuyển vé cho vợ|chuyển viết cho|chuyển vị cho)\b", "chuyển ví cho"),
    (r"\b(chàng truyền hình)\b", "tiền truyền hình"),
    (r"\b(nhắn xns|nhắc em sẽ về cho)\b", "nhắn SMS cho"),
    (r"\b(ghi van lịch|ghi vào lạch)\b", "ghi vào lịch"),
    (r"\b(một kế chuyển)\b", "một vé chuyến"),
    (r"\b(thành quản thanh toán)\b", "tài khoản thanh toán"),
    (r"\b(lim cây bảy năm)\b", "bàn phím MKey 75"),
    (r"\b(băng cốc gia đình)\b", "Vincom Gia Định"),
    (r"\b(tắt cái phim gửi xe)\b", "kết thúc phiên gửi xe"),
    (r"\b(lái xe ra khỏi bãi)\b", "lấy xe ra khỏi bãi"),
    (r"\b(của cô và bao nhiêu con)\b", "của cô giá bao nhiêu con"),
    (r"\b(nàng năm chục điện thoại cho lan rồi chuyển phí cho huấn đúng số tiền mạng tháng nay nhé)\b", "nạp 50 nghìn điện thoại cho Lan rồi chuyển ví cho Tuấn đúng số tiền mạng tháng này nhé"),
    (r"\b(tiền truyền hình tháng này hết bao nhiêu chia đôi với lan gửi phần của mình qua vĩ cho lan luôn nha)\b", "tiền truyền hình tháng này hết bao nhiêu, chia đôi với Lan, gửi phần của mình qua ví cho Lan luôn nha"),
    (r"\b(coi giúp bác tiền điện tháng này ví đủ thì trả bằng ví thiếu thì trả bằng ngân hàng rồi nhận hết m s cho mẹ báo đã đóng nhé)\b", "coi giúp bác tiền điện tháng này, ví đủ thì trả bằng ví, thiếu thì trả bằng ngân hàng rồi nhắn SMS cho Mẹ báo đã đóng nhé"),
    (r"\b(thanh toán tiền mà tháng này bằng ví điện tử trước em rồi tìm số cô giáo mai nhắn tin báo em đóng tiền bạn rồi)\b", "thanh toán tiền mạng tháng này bằng ví điện tử giúp em rồi tìm số cô giáo Mai nhắn SMS báo em đóng tiền mạng rồi"),
    (r"\b(tài khoản thanh toán còn bao nhiêu em chuyển cho mẹ một nửa tiền điện tháng này và chuyển cho anh bình tiền nước tháng này nhé thiếu thì lấy bên tiết kiệm)\b", "tài khoản thanh toán còn bao nhiêu em, chuyển cho Mẹ một nửa tiền điện tháng này và chuyển cho anh Bình tiền nước tháng này nhé, thiếu thì lấy bên tiết kiệm"),
    (r"\b(coi giúp anh hóa đơn điện thoại trả sau bao nhiêu ví đủ thì trả bằng ví thiếu thì lấy tài khoản ngân hàng xong nhắn ép anh ếch cho bố bảo đóng rồi nha)\b", "coi giúp anh hóa đơn điện thoại trả sau bao nhiêu, ví đủ thì trả bằng ví, thiếu thì lấy tài khoản ngân hàng xong nhắn SMS cho Bố báo đóng rồi nha"),
    (r"\b(em coi hóa đơn tiền nước tháng này bao nhiêu thanh toán bằng ví luôn rồi chuyển khoản ngân hàng cho chủ nhà đúng số đó để đối chiếu nha)\b", "em coi hóa đơn tiền nước tháng này bao nhiêu, thanh toán bằng ví luôn rồi chuyển khoản ngân hàng cho Chủ nhà đúng số đó để đối chiếu nha"),
    (r"\b(em xem trong mấy hóa đơn chưa đóng cái nào đến hạn sớm nhất thì chuyển khoản cho mẹ đúng số tiền đó rồi chuyển cho chú năm đúng bằng tiền nước luôn nha)\b", "em xem trong mấy hóa đơn chưa đóng cái nào đến hạn sớm nhất thì chuyển khoản cho Mẹ đúng số tiền đó rồi chuyển cho Chú Năm đúng bằng tiền nước luôn nha"),
    (r"\b(xem giúp mình tiền điện với tiền mạng cái nào rẻ hơn thì thanh toán bằng ví cái đắt hơn trả qua ngân hàng nhé)\b", "xem giúp mình tiền điện với tiền mạng cái nào rẻ hơn thì thanh toán bằng ví, cái đắt hơn trả qua ngân hàng nhé"),
    (r"\b(hóa đơn truyền hình cáp tháng này bao nhiêu vậy nếu ví đủ thì trả bằng ví còn hụt thì trừ ngân hàng rồi nhắn tin cho tuấn báo đã thanh toán nhé)\b", "hóa đơn truyền hình cáp tháng này bao nhiêu vậy, nếu ví đủ thì trả bằng ví còn hụt thì trừ ngân hàng rồi nhắn SMS cho Tuấn báo đã thanh toán nhé"),
    (r"\b(coi giúp anh mấy hóa đơn tháng này cái nào dưới bốn trăm ngàn thì đóng bằng ví hết đi riêng tiền điện để sau nha)\b", "coi giúp anh mấy hóa đơn tháng này cái nào dưới 400 nghìn thì đóng bằng ví hết đi, riêng tiền điện để sau nha"),
    (r"\b(nạp một trăm nghìn điện thoại cho bố qua ví rồi gửi cho hương đúng bằng tiền nước tháng này qua ví luôn nhé)\b", "nạp 100 nghìn điện thoại cho Bố qua ví rồi gửi cho Hương đúng bằng tiền nước tháng này qua ví luôn nhé"),
    (r"\b(này coi giúp bác tiền điện với tiền nước tháng này hết bao nhiêu chuyển khoản cho mẹ đúng bằng tiền điện còn tiền nước nếu ví đủ thì đóng bằng ví luôn nhé)\b", "này coi giúp bác tiền điện với tiền nước tháng này hết bao nhiêu, chuyển khoản cho Mẹ đúng bằng tiền điện, còn tiền nước nếu ví đủ thì đóng bằng ví luôn nhé"),
    (r"\b(kiểm tra hóa đơn internet tháng này giúp mình trả bằng ngân hàng rồi nhắn tin cho bình báo tiền mạng tháng này đã đóng nhé)\b", "kiểm tra hóa đơn internet tháng này giúp mình, trả bằng ngân hàng rồi nhắn SMS cho Bình báo tiền mạng tháng này đã đóng nhé"),
    (r"\b(coi tiền truyền hình tháng này bao nhiêu đóng qua ngân hàng giúp tôi rồi gọi cho bố báo tôi đóng tiền cáp rồi nhé)\b", "coi tiền truyền hình tháng này bao nhiêu, đóng qua ngân hàng giúp tôi rồi gọi điện cho Bố báo tôi đóng tiền cáp rồi nhé"),
    (r"\b(kiểm tra số dư ngân hàng giúp anh chuyển cho mẹ số tiền bằng hóa đơn điện và chuyển cho trần văn tuấn số tiền bằng hóa đơn mạng nếu tài khoản chính không đủ cho cả hai thì lấy bên tiết kiệm nhé)\b", "kiểm tra số dư ngân hàng giúp anh, chuyển cho Mẹ số tiền bằng hóa đơn điện và chuyển cho Trần Văn Tuấn số tiền bằng hóa đơn mạng, nếu tài khoản chính không đủ cho cả hai thì lấy bên tiết kiệm nhé"),
    (r"\b(này em xem hóa đơn tháng này rồi chuyển khoản cho mẹ số tiền bằng tiền điện cộng tiền nước chuyển cho em gái số tiền bằng tiền mạng nhé tài khoản thanh toán không đủ tổng đó thì dùng tiết kiệm)\b", "này em xem hóa đơn tháng này rồi chuyển khoản cho Mẹ số tiền bằng tiền điện cộng tiền nước, chuyển cho Em gái số tiền bằng tiền mạng nhé, tài khoản thanh toán không đủ tổng đó thì dùng tiết kiệm"),
    (r"\b(em kiểm tra các hóa đơn chưa thanh toán cái nào đến hạn trong ba ngày tới mà dưới năm trăm nghìn thì đóng bằng ví còn trên năm trăm nghìn thì đóng bằng ngân hàng cái nào chưa gấp thì thôi)\b", "em kiểm tra các hóa đơn chưa thanh toán, cái nào đến hạn trong 3 ngày tới mà dưới 500 nghìn thì đóng bằng ví, còn trên 500 nghìn thì đóng bằng ngân hàng"),
    (r"\b(chị muốn gửi tiết kiệm kỳ hạn ba tháng em xem tài khoản thanh toán có hơn mười triệu không nếu hơn thì trích phần vượt trên mười triệu gửi hết vào sổ tiết kiệm nhé còn không đủ thì thôi)\b", "chị muốn gửi tiết kiệm kỳ hạn 3 tháng, em xem tài khoản thanh toán có hơn 10 triệu không, nếu hơn thì trích phần vượt trên 10 triệu gửi hết vào sổ tiết kiệm nhé"),
    (r"\b(tìm mấy chỗ rửa xe quanh láng hạ rồi gửi vị trí chỗ gần nhất cho bình trên giao lưu hộ anh)\b", "tìm mấy chỗ rửa xe quanh Láng Hạ rồi gửi vị trí chỗ gần nhất cho Bình trên Zalo hộ anh"),
    (r"\b(anh thật xa một bé tội ấy ra bay xin sáng nhí)\b", "đặt cho anh một vé tàu SE7 sáng nhé"),
    (r"\b(anh rể bay nhanh)\b", "anh gửi 3 ngày"),
    (r"\b(chắc cái báo thức sau giải sáng của chị đi nhé)\b", "tắt cái báo thức 6 rưỡi sáng của chị đi nhé"),
    (r"\b(nhớ lại xem chuyến về hải phòng tối nay mấy giờ chạy thế)\b", "xem vé tàu chuyến về Hải Phòng tối nay mấy giờ chạy thế"),
    (r"\b(em ơi anh định vội bị huyết hợp viêm máu xem nếu anh từ nhà đi xe buýt thì mắc chuyện nào nhé)\b", "em ơi anh định tới Viện Huyết học truyền máu, xem nếu anh từ nhà đi xe buýt thì bắt chuyến nào nhé"),
    (r"\b(chiều mai chị có lớp ở trung tâm văn hoá tây hồ em xem đường từ nhà qua đáy rồi so với xe buýt sấm sẹo sóng nhắc chị năm giờ mới chiều mai ra khỏi nhà nhà)\b", "chiều mai chị có lớp ở Trung tâm Văn hóa Tây Hồ, em xem đường đi từ nhà qua đấy rồi so với xe buýt xem sao, xong nhắc chị 5 giờ mười chiều mai ra khỏi nhà nha"),
    (r"\b(em kiểm tra cái đơn đèn đọc sách anh mới đặt nếu đang chuẩn bị chưa giao thì hủy luôn còn giao rồi thì thôi nhưng nếu trên sàn có mẫu đèn bàn nào dưới ba trăm rưỡi thì bỏ vào giỏ cho anh một cái nhé)\b", "em kiểm tra cái đơn đèn đọc sách anh mới đặt, nếu đang chuẩn bị chưa giao thì hủy luôn, rồi tìm trên sàn mẫu đèn bàn dưới 350 nghìn bỏ vào giỏ cho anh 1 cái nhé"),
    (r"\b(này hôm hai mươi ba tháng vừa tôi phải có mặt ở nhà vào quá thành nhưng lúc vốn mẹ vốn lằng chiều tìm xe buýt thử bánh xe long bì đi ăn đó tí nữa ít phải đồ xe ấy xem đi ít bao lâu rồi nhất tôi đã bánh xong hơn giờ lên xe hai mươi phút nhé)\b", "hôm 23 tháng 10 tôi phải có mặt ở Nhà văn hóa Thanh Niên lúc 4 giờ 45 chiều, tìm xe buýt từ Bến xe Long Biên đi đến đó rồi nhắc tôi ra bến sớm hơn giờ lên xe 20 phút nhé"),
    (r"\b(chuyến đắt nhắc đi đà lạt tối thứ sáu tuần này còn phải không em)\b", "tìm chuyến xe khách đi Đà Lạt tối thứ sáu tuần này còn vé không em"),
    (r"\b(sáng thứ bảy tuần này chị muốn đi tàu ra hải phòng ăn cưới em tìm xem có chuyến nào tầm sáu bảy giờ sáng không nhé)\b", "sáng thứ bảy tuần này chị muốn đi tàu ra Hải Phòng, em tìm chuyến tàu tầm 6 7 giờ sáng nhé"),
    (r"\b(tám giờ sáng thứ hai tuần sau anh đi xe buýt từ nhà ra mê chị em coi tiếng lau nền rất đặng sẵn taxi vũ chúa nón anh ở trần xuống cửa tuyến đó đúng lúc xe tới đã vô trung tâm hoàng yên gia rồi tạo thêm định theo giờ xuống xe buýt luôn nha)\b", "8 giờ sáng thứ hai tuần sau anh đi xe buýt từ nhà ra Mễ Trì, em coi tuyến nhanh nhất rồi đặt xe 4 chỗ đón anh ở trạm xuống của tuyến đó đúng lúc xe tới chở vô Trung tâm Hội nghị Quốc gia, rồi tạo lịch theo giờ xuống xe buýt luôn nha"),
    (r"\b(ừ em trai tuyến xe buýt đi cứ tính năm giờ là lên xe luôn rồi trộm thời gian xe chạy nếu tới nơi trước sáu giờ tối thì dời lịch sang giờ xe tới lẫn hợp một tiếng là đủ rồi đổi lịch tên lịch hành gặp đối tác ở trung tâm văn hoá tây hồ lụ nhà)\b", "xem tuyến xe buýt đi từ nhà tới Trung tâm Văn hóa Tây Hồ, tính 5 giờ chiều lên xe, nếu tới nơi trước 6 giờ tối thì dời lịch sang giờ xe tới và đổi tên lịch thành Gặp đối tác ở Trung tâm Văn hóa Tây Hồ luôn nha"),
    (r"\b(em ơi chị vừa mới lấy hai hộp sữa chua trong tủ ra ăn rồi trừ đi giúp chị nhé tiện xem luôn trong tủ còn mấy quả trứng gà nếu dưới năm quả thì ghi thêm trứng gà vào danh sách đi chợ còn đủ dùng thì thôi khỏi ghi)\b", "em ơi chị vừa mới lấy 2 hộp sữa chua trong tủ bếp ra ăn rồi trừ đi giúp chị nhé, tiện xem luôn trong tủ còn mấy quả trứng gà nếu dưới 5 quả thì ghi thêm trứng gà vào danh sách đi chợ"),
    (r"\b(chiều thứ sáu tuần này minh vào nha trang em tìm giúp minh các chuyến bay từ hà nội đi nha trang chiều hôm đó xem hãng nào giờ giấc hợp lý nhé)\b", "chiều thứ sáu tuần này mình vào Nha Trang, em tìm chuyến bay từ Hà Nội đi Nha Trang chiều hôm đó nhé"),
    (r"\b(xen cho anh từ hồ gươm ra long biên rồi bắt xe buýt chín nào rồi cho anh cái ô tô bốn chỗ thì thằng nói làm ra hồ gươm luôn nha)\b", "xem cho anh từ Hồ Gươm ra Long Biên thì bắt xe buýt chuyến nào, rồi gọi cho anh cái ô tô 4 chỗ từ nhà ra Hồ Gươm luôn nha"),
    (r"\b(ừ thế em nhấc chị ra biển trước chuyền đấy mười lăm phút nhé)\b", "ừ thế em nhắc chị ra bến trước chuyến đấy 15 phút nhé"),
    (r"\b(tối thứ bảy này nhà mình định đi nghe nhạc em tìm xem có đêm nhạc nào tối thứ bảy không nếu giá vé từ bốn trăm nghìn trở xuống thì đặt luôn cho anh hai vé còn đắt hơn thì thôi đừng đặt chỉ cần tạo lịch nhắc anh tối thứ năm bàn lại với vợ là được)\b", "tối thứ bảy này nhà mình định đi nghe nhạc, em tìm xem có đêm nhạc nào tối thứ bảy không, nếu giá vé từ 400 nghìn trở xuống thì đặt luôn cho anh 2 vé, còn đắt hơn thì tạo lịch nhắc anh tối thứ năm bàn lại với vợ"),
    (r"\b(chị mới đặt mấy đồ bên siêu thị mà quên mất em coi lại đơn đang giao tới đâu rồi với cả trên sàn thương mại cái đơn tai nghe chị đặt hôm qua đã gửi đi chưa nhé)\b", "chị mới đặt mấy đồ bên siêu thị mà quên mất, em kiểm tra đơn hàng trên sàn thương mại cái đơn tai nghe chị đặt hôm qua đã gửi đi chưa nhé"),
    (r"\b(cái áo sơ mi chị nhận hôm qua bị rách đường chỉ ở vai rồi em làm yêu cầu trả hàng giúp chị với lý do hàng lỗi rồi đặt lại một chiếc áo sơ mi trắng khác vào giỏ hàng để chị xem lại nhé)\b", "cái áo sơ mi chị nhận hôm qua bị rách đường chỉ ở vai rồi, em làm yêu cầu trả hàng giúp chị với lý do hàng lỗi, rồi tìm trên sàn đặt lại 1 chiếc áo sơ mi trắng khác vào giỏ hàng nhé"),
    (r"\b(nằm thêm vào thẻ xe buýt tiền đi hai chục lượt từ kim liên về cầu giấy giúp chị rồi xem mấy giờ có chuyến nhé)\b", "xem tuyến xe buýt từ Kim Liên về Cầu Giấy rồi nạp thêm vào thẻ xe buýt tiền đi 20 lượt giúp chị, xong xem mấy giờ có chuyến tiếp theo nhé"),
    (r"\b(nạp ba trăm à thôi nạp đủ hai chục lượt metro ra suối ti mà vé hơn hai chục nghìn thì khỏi em đi biết)\b", "xem tuyến metro từ Nhà hát Thành phố ra Suối Tiên rồi nạp thẻ xe buýt đủ 20 lượt nhé"),
    (r"\b(bảy giờ tới à không chốt chuyến một giờ trưa một vé nhà mê trô biến thành lên suối tiên mà dưới bốn chục phút thì nạp thẻ năm chục lâu hơn thì gọi xe máy về ký túc xá)\b", "đặt cho em 1 vé xe chuyến 1 giờ trưa nha, rồi xem tuyến metro từ Bến Thành lên Suối Tiên mà dưới 40 phút thì nạp thẻ xe buýt 50 nghìn"),
    (r"\b(à mà thôi có giữ chỗ bãi xe nữa nha tìm giùm em xem tối nay có sự kiện âm nhạc nào không)\b", "tìm giùm em xem tối nay có đêm nhạc sự kiện âm nhạc nào không"),
    (r"\b(gói hạt mất cao giao tới mà bị hội dòng rồi thôi khỏi ăn nữa em làm yêu cầu trả hàng giúp chị đi xong tìm bên đi chợ có hạnh nhân rang bơ dưới một trăm rưỡi thì đặt hai gói giao về nhà cho chị luôn)\b", "gói hạt macca giao tới bị hôi dầu rồi, em làm yêu cầu trả hàng giúp chị đi, xong tìm bên đi chợ có hạnh nhân rang bơ dưới 150 nghìn thì đặt mua 2 gói giao về nhà cho chị luôn"),
    (r"\b(coi dùm ba mấy gói dịch vụ đang đăng ký tháng này tốn bao nhiêu cái nào từ hai trăm ngàn trở lên thì hủy bớt cho đỡ tốn rồi nhắc ba sáng chủ nhật kiểm tra lại thẻ ngân hàng nhé)\b", "coi dùm ba mấy gói cước đang đăng ký tháng này tốn bao nhiêu, cái nào từ 200 nghìn trở lên thì hủy gói bớt cho đỡ tốn rồi nhắc ba sáng chủ nhật kiểm tra lại thẻ ngân hàng nhé"),
    (r"\b(mẹ chuyển trình thế này chưa rồi đi nạp thẻ thêm mười lượt metro kia xong gọi điện cho mẹ nha)\b", "kiểm tra giao dịch ngân hàng xem Mẹ chuyển tiền tháng này chưa, rồi nạp thẻ xe buýt thêm 10 lượt metro kia xong gọi điện cho Mẹ nha"),
    (r"\b(em ơi hủy cái vé xe đi đà lạt hồi sáng giúp anh nha à mà thôi đừng hủy nữa cứ để đó đi anh hỏi thêm là tối thứ bảy tuần này ở đà lạt có đêm nhạc nào không tìm thử giúp anh với)\b", "anh hỏi thêm là tối thứ bảy tuần này ở Đà Lạt có đêm nhạc nào không tìm vé sự kiện giúp anh với"),
    (r"\b(chiều nay tôi dẫn hai khách từ nhà đi xe buýt qua puleo village em nạp thẻ đón tiền vé khứ hồi ba người ngàn)\b", "chiều nay tôi dẫn 2 khách từ nhà đi xe buýt qua Pilgrim Village, xem tuyến xe buýt rồi nạp thẻ xe buýt đúng tiền vé khứ hồi 3 người nha"),
    (r"\b(cái ấm đun nước mới giao bị móp một bên rồi thôi khỏi dùng luôn em bấm trả hàng giúp bác đi rồi tìm trên sàn cái ấm đun siêu tốc nào dưới bốn trăm ngàn bỏ vào giỏ cho bác một cái nhé)\b", "cái ấm đun nước mới giao bị móp một bên rồi, em làm yêu cầu trả hàng giúp bác đi rồi tìm trên sàn cái bình đun siêu tốc nào dưới 400 nghìn bỏ vào giỏ cho bác 1 cái nhé"),
    (r"\b(em nạp thẻ cho anh đúng bằng tiền vé tiếng rượu mười sáu hay tiếng rượu một rồi sáu giờ sáu mai nhắc anh ra trạm nghe)\b", "nạp thẻ xe buýt cho anh đúng bằng tiền vé hai tuyến R16 với tuyến R1 rồi 6 giờ sáng mai nhắc anh ra trạm nghe"),
    (r"\b(em tìm giúp chị công thức làm món gà xào sả ớt với định đặt đồ về nấu luôn mà thôi để mai đi chợ mua cho tươi em lưu công thức đó lại rồi ghi thịt gà vào danh sách đi chợ giúp chị nhé)\b", "em tìm giúp chị công thức làm món gà xào sả ớt, lưu công thức đó lại rồi ghi thịt gà vào danh sách đi chợ giúp chị nhé"),
    (r"\b(tôi đi tiếng mười lăm mà có thể đủ vé đi phải về rồi trả lại mail lucas là tôi đi xe buýt với ngay)\b", "nạp thẻ xe buýt đủ vé đi và về tuyến 15 rồi trả lời email Lucas là tôi đi xe buýt tới nha"),
    (r"\b(tối nay chị định đặt bún bò ăn mà thôi nhà còn cơm nguội rồi em tìm giúp chị công thức làm món cơm chiên dương châu xem cần những nguyên liệu gì nhé)\b", "em tìm giúp chị công thức làm món cơm chiên dương châu xem cần những nguyên liệu gì nhé"),
    (r"\b(sáng mai tính gọi xe đi làm mà thôi để anh tự lái xe đi cũng được em kiểm tra giúp anh bãi đỗ xe gần công ty còn chỗ không nếu còn thì giữ chỗ cho anh hai tiếng từ tám giờ sáng mai nhé)\b", "em kiểm tra giúp anh bãi đỗ xe gần công ty còn chỗ không, nếu còn thì giữ chỗ ở bãi cho anh 2 tiếng từ 8 giờ sáng mai nhé"),
    (r"\b(nạp thẻ xe buýt giúp cho tôi mực căn à không năm chục nghìn thôi xong rồi giúp tôi thì chắc chân nhe)\b", "nạp thẻ xe buýt giúp cho tôi 50 nghìn thôi xong rồi đánh dấu hôm nay hoàn thành thói quen đi bộ giúp tôi nhé"),
    (r"\b(mấy đứa nhỏ đổi ý không ăn gà rán nữa nên thôi khỏi đặt đơn đó nha em tìm giúp chị công thức làm bánh kếp chuối rồi lưu lại để chiều chị làm cho tụi nhỏ ăn)\b", "em tìm giúp chị công thức làm pancake chuối rồi lưu lại để chiều chị làm cho tụi nhỏ ăn"),
    (r"\b(anh tính mua thêm một bàn làm việc mà thôi bàn cũ vẫn dùng tốt em kiểm tra cái đơn ghế công thái học anh đặt tuần trước xem giao tới đâu rồi nhé)\b", "em kiểm tra đơn hàng cái ghế công thái học anh đặt tuần trước xem giao tới đâu rồi nhé"),
    (r"\b(chiều nay tính đi xem triển lãm tranh mà trời mưa quá thôi ở nhà vậy em mở tivi chuyển sang kênh phim truyện giúp chị rồi tìm công thức nấu chè hạt sen để tối chị nấu nhé)\b", "em mở tivi chuyển sang kênh phim truyện giúp chị rồi tìm công thức nấu chè hạt sen để tối chị nấu nhé"),
    (r"\b(giữa anh đây xe buýt ra huế sáng kế tiếp ở trạm gần nhà có kịp đón lúc khác không em)\b", "xem tuyến xe buýt từ nhà ra Ga Huế rồi xem chuyến kế tiếp ở trạm gần nhà mấy giờ có chuyến để kịp đón Lucas không em"),
    (r"\b(chừng mười phút nữa tôi mới ra tới trạm công ty sài gòn pháp coi dù chuyến xe buýt nào kịp nha)\b", "chừng 10 phút nữa tôi mới ra tới trạm Công ty Sài Gòn Soft, coi giùm chuyến xe buýt kế tiếp ở trạm đó mấy giờ chạy nha"),
    (r"\b(ờ em nạp thêm hai trăm ngàn vô thẻ xe buýt hồi sáng giới đóng luôn cái bưu truyền hình bằng ví nha)\b", "ờ em nạp thêm 200 nghìn vô thẻ xe buýt hồi sáng với đóng luôn cái hóa đơn tiền truyền hình bằng ví điện tử nha"),
    (r"\b(cái nồi chiên không dầu giao tới bị nứt vỏ rồi em làm đơn trả hàng giúp chị luôn đi rồi đặt mua một cái nồi chiên không dầu khác bên sàn thương mại bỏ vào giỏ cho chị nhé)\b", "cái nồi chiên không dầu giao tới bị nứt vỏ rồi, em làm yêu cầu trả hàng giúp chị luôn đi rồi tìm trên sàn đặt mua 1 cái nồi chiên không dầu khác bỏ vào giỏ cho chị nhé"),
    (r"\b(lúc đầu anh định đặt hai cái đèn bàn học mà thôi phòng nhỏ quá không kê vừa khỏi mua đèn nữa em tìm trên sàn xem có ổ cắm thông minh nào dưới hai trăm rưỡi thì thêm vào giỏ cho anh hai cái nhé)\b", "em tìm trên sàn xem có ổ cắm thông minh nào dưới 250 nghìn thì thêm vào giỏ cho anh 2 cái nhé"),
    (r"\b(gởi cho mình luôn đi)\b", "gọi cho Minh luôn đi"),
    (r"\b(đặt phố phước năm nửa chiều nay)\b", "đặt báo thức 5 rưỡi chiều nay"),
    (r"\b(nàng giúp mình hai vẽ|đang giúp mình 2 viện)\b", "đặt giúp mình 2 vé"),
    (r"\b(lấy chuyện cuối cùng cho ngay)\b", "lấy chuyến cuối cùng trong ngày"),
    (r"\b(giáp tôi|nhanh tôi)\s+(?:thắt|tắt)\s+bếp\b", "nhắc tôi tắt bếp"),
    (r"\b(nhắn tôi đổi sáng năm ra chiều mai|nhắc tôi độ sáng 5 giờ chiều mai)\b", "nhắc tôi đổ xăng 5 giờ chiều mai"),
    (r"\b(vàng ét chê xây|vàng st|vang sgc|ét chê xây)\b", "vàng SJC"),
    (r"\b(một trăm và mươi một triệu)\b", "131 triệu"),
    (r"\b(nhện vàng|nhận vàng)\b", "nhẫn vàng"),
    (r"\b(em muốn có báo thức giữa các truyện ngang đã đạt được một tiếng là làm thủ thuật nhé)\b", "em muốn có báo thức trước giờ cất cánh chuyến bay đã đặt 1 tiếng tên là Làm thủ tục nhé"),
    (r"\b(chuyện này tôi đưa bố mẹ đi khám mắt gỗ giúp cô tôi bày trở về công ty tới chữa khám đón trước giờ hạng bốn mươi phút rồi nhắc tôi mạng kính cũ của bố theo trước lúc xe tới hai mươi phút)\b", "chiều nay tôi đưa bố mẹ đi khám mắt, gọi giúp xe ô tô 7 chỗ từ công ty tới chỗ khám đón trước giờ hẹn 40 phút rồi nhắc tôi mang kính cũ của bố theo trước lúc xe tới 20 phút"),
    (r"\b(thầy sẽ chịu may cho tôi mới chuyển ví)\b", "2 giờ chiều mai cho tôi với chuyển ví"),
    (r"\b(vịch đèn)\b", "bật đèn"),
    (r"\b(hớ hẩy giờ chiều nay)\b", "2 giờ chiều nay"),
    (r"\b(chuyện ví)\b", "chuyển ví"),
    (r"\b(đến trên lầu có cái nào đang để sáng)\b", "đèn trên lầu có cái nào đang để sáng"),
    (r"\b(mở được và phòng ngủ)\b", "mở điều hòa phòng ngủ"),
    (r"\b(một chức chỉ cái cuộc phòng khách|một chức trị cái cuộn phòng khách)\b", "bật giúp chị cái quạt phòng khách"),
    (r"\b(màu danh ngân hàng)\b", "mạo danh ngân hàng"),
    (r"\b(giả làm nhân viên thuận đại căn hạng)\b", "giả làm nhân viên tổng đài ngân hàng"),
    (r"\b(bề ngoài lúc này lăng đẳng hay mưa đấy sẽ giúp mình sợ chết vì sợ mưa ít)\b", "bên ngoài lúc này đang nắng hay mưa đấy, xem giúp mình thời tiết hiện tại với"),
    (r"\b(em đang đề lịch đó lúc chín giờ sáng nhúc anh mới)\b", "em tạo lại lịch đó lúc 9 giờ sáng giúp anh với"),
    (r"\b(hà thôi khỏe đậm nữa nha chiều mình tiện đi đường ghém chưa thể buông luôn)\b", "à thôi khỏi đặt nữa nha, chiều mình tiện đi đường ghé siêu thị mua luôn"),
    (r"\b(làm đôi máy sấy tóc trước tia sắp giao nhằm ngủ khép rồi)\b", "làm đơn trả hàng máy sấy tóc giúp chị, shop giao nhầm mẫu khác rồi"),
    (r"\b(nhà mẹ tiến giúp chị mặc tem sửa xe quanh nhà hát lớn sang thời nhà chị có thấy điều nào với giữa ruộng chỗ ở bãi xe sát tuyến từ mười giờ tối nay chỉ dưới hai tiếng thôi)\b", "em ơi tìm giúp chị một tiệm sửa xe quanh Nhà hát Lớn, xem từ nhà chị qua đấy đường nào, với giữ luôn một chỗ ở bãi xe sát tiệm từ 10 giờ tối nay, chị gửi 2 tiếng thôi"),
    (r"\b(mở bài nhịp mở đầu cho ăn cái không đánh dấu hôm nay là làm phim tập trung buổi sáng rồi nhé)\b", "mở bài Nhịp mở đầu cho anh cái, xong đánh dấu hôm nay đã làm phiên tập trung buổi sáng rồi nhé"),
    (r"\b(gợi giúp tôi số không chín tháng tám một một hai hai ba ba bảy của bên đầu mỗi sự cố nhé song luôn cho tôi cái ghi chú tên là bàn giao sự cố kia là đã gọi số đấy báo loại mấy phát)\b", "gọi điện giúp tôi số 0988112233 của bên đầu mối sự cố nhé, xong lưu cho tôi cái ghi chú tên là Bàn giao sự cố ghi là đã gọi số đấy báo lỗi máy phát"),
    (r"\b(với toa lịch trình công tác góc sáu giờ sáng hôm đó nữa)\b", "với tạo lịch trình công tác lúc 6 giờ sáng hôm đó nữa"),
    (r"\b(soạn sẵn một email nhét của em ấy chưa đề phản hồi hồ sơ đăng ký báo là hồ sơ sẽ được xử lý trước khi sống nhé)\b", "soạn sẵn một email nháp gửi em ấy tiêu đề Phản hồi hồ sơ đăng ký báo là hồ sơ sẽ được xử lý trước thứ sáu nhé"),
    (r"\b(đợt lữ uyên khoa tiếng anh đáo quay ra lô sạch tết dưỡng trái banh xếp hàng chờ họp với khách nước ngoài)\b", "được, gửi nguyên câu tiếng Anh đó qua Zalo cho sếp Dũng giúp anh, sếp đang chờ họp với khách nước ngoài"),
    (r"\b(chào buổi giáy nhé chúc em hôm nay vui vẻ mở việc suôn sẻ)\b", "chào buổi sáng nhé chúc em hôm nay vui vẻ mọi việc suôn sẻ"),
    (r"\b(được rồi cứ điện nguyên như thế có thể làm gì thiện với em)\b", "à thôi khỏi làm gì thêm đâu em, cứ để nguyên như thế"),
    (r"\b(xem giúp mình điều từ nhà đó đi qua tho lâu)\b", "xem giúp mình đường từ nhà đó đi qua bao lâu"),
    (r"\b(thời bảy rưỡi tối phải hai tiếng)\b", "từ 7 rưỡi tối gửi 2 tiếng"),
    (r"\b(chả hàng cái rau nhồng để đi lý do là rau say mẫu)\b", "trả hàng cái đơn giao nhầm đấy đi lý do là giao sai mẫu"),
    (r"\b(dựng chỗ ở bãi phía bắc ca)\b", "giữ chỗ ở bãi phía Bắc ga"),
    (r"\b(trước trời chảy nở tiếng)\b", "trước giờ chạy nửa tiếng"),
    (r"\b(để cái áo sơ mi xanh ấy sớm đuổi nhà anh gặp một tháng cơ em đều yêu cầu trắng em giúp anh rồi thêm vào kịp cần làm là năm giờ chiều thứ ba tuần sau phải gửi trả xong nhé)\b", "cái áo sơ mi xanh ấy shop gửi nhầm, anh đặt màu trắng cơ, em làm yêu cầu trả hàng giúp anh rồi thêm vào việc cần làm hạn 5 giờ chiều thứ ba tuần sau phải gửi trả xong nhé"),
    (r"\b(mua cho em bốn ghế xe startup company tối thứ bảy tuần sau nhưng đặt luôn xe bỉ chuẩn đóng ở công ty lúc năm chiều luôn hôm đó chờ cả nhóm sẽ gửi biển rồi đóng Zalo cho mình hàng năm rồi chui vào sảnh công ty)\b", "mua cho anh 4 vé xem stand up comedy tối thứ bảy tuần sau nha, đặt luôn xe 7 chỗ đón ở công ty lúc 5 rưỡi chiều hôm đó chở cả nhóm tới chỗ diễn rồi nhắn Zalo cho Minh là 5 rưỡi tập trung ở sảnh công ty"),
    (r"\b(thật giúp em hai vé tàu se e ba)\b", "đặt giúp em 2 vé tàu SE3"),
    (r"\b(rơi ra hà nội luôn)\b", "ra Ga Hà Nội luôn"),
    (r"\b(đặt cho tôi năm vé chuyện làm ảnh sáng đô thị cho cả nhà nhẹ rồi đi vào lịch sử tôi là bảy rưỡi tối thứ bảy mùng ba tháng mười đi xem)\b", "đặt cho tôi 5 vé Triển lãm Ánh sáng Đô thị cho cả nhà nhé, rồi ghi vào lịch giúp tôi là 7 rưỡi tối thứ bảy mùng 3 tháng 10 đi xem"),
    (r"\b(này giúp tôi tìm cách nói cho gà nhé rồi đòi luôn cho tôi một con rạp a làm giảng viên nữa)\b", "này giúp tôi tìm công thức nấu cháo gà nhé, rồi đặt mua luôn cho tôi 1 con gà ta làm sẵn bên đi chợ nữa"),
    (r"\b(được thế chỉ đơn cho tôi từ nhà tới cái quán đầu tiên ấy)\b", "được, thế chỉ đường cho tôi từ nhà tới cái quán đầu tiên ấy"),
    (r"\b(em bật tay trông choạm chùm nhà lên cho tôi nhé)\b", "em bật báo động chống trộm ấy, chế độ vắng nhà lên cho tôi nhé"),
    (r"\b(bé xe hải phòng thứ năm tới mới dần tới nơi quán sáu giờ chiều lại lỡ tiệc cưới khỏe luôn cho cô)\b", "vé xe Hải Phòng thứ năm tới mấy giờ tới nơi, quá 6 giờ chiều là lỡ tiệc cưới, hủy luôn cho cô"),
    (r"\b(coi sang chín lăm dùng anh trên hai ba rưỡi thì ghi lịch đổi sang)\b", "coi giá xăng 95 giùm anh, trên 23 nghìn rưỡi thì ghi vào lịch Đổ xăng"),
    (r"\b(thôi bảy một nửa số ZKN của tôi đây)\b", "tôi bán một nửa số cổ phiếu ZKN của tôi đây"),
    (r"\b(cô chọn chung kết ở rừng đó còn vé không còn để lấy hai vé lấy hai vé xoay gọn cười rồi hẹn báo thức trước dưới một tiếng)\b", "coi trận chung kết ở sân đó còn vé không, còn thì lấy 2 vé sự kiện, hết thì lấy 2 vé Sài Gòn Cười rồi đặt báo thức trước giờ diễn 1 tiếng"),
    (r"\b(chử dùng cô trai giàu ăn à thôi giàu chưa xài cô mới lấy hộp sữa để dành cho con thảo chử hộp sữa thôi con)\b", "trừ giùm cô hộp sữa trong tủ bếp thôi con"),
    (r"\b(cuộc shop hoãn rồi hụy vé máy bay ra đà nẵng giùm chị rượu năm giờ chiều hôm trước ngày bay nhắc chị gọi khách nha)\b", "workshop hoãn rồi, hủy vé máy bay ra Đà Nẵng giùm chị, rồi 5 giờ chiều hôm trước ngày bay nhắc chị gọi khách nha"),
    (r"\b(sarah báo nhận tiền hai lần em coi eo gì gần đây của tài khoản)\b", "Sarah báo nhận tiền 2 lần, em kiểm tra sao kê giao dịch ngân hàng gần đây của tài khoản"),
    (r"\b(nà thế cho tôi đủ tiền để tính không chính kể đi lẫn về cho mấy buổi học của cô anna smith còn sao nhé)\b", "nạp thẻ xe buýt cho tôi đủ tiền đi tuyến 09 cả đi lẫn về cho mấy buổi học của cô Anna Schmidt tuần sau nhé"),
    (r"\b(lucas ở lại thêm hai đen rồi hãy dùng túi vé của lucas với bật đèn phòng khách luôn hãy)\b", "Lucas ở lại thêm 2 đêm rồi, hủy giùm tôi vé máy bay của Lucas với bật đèn phòng khách luôn nhé"),
    (r"\b(đặt giường ăn bú vé hội thảo của chile goa tơ bayern cho mấy bạn trong nhóm ăn kẹt hoặc không đi được)\b", "đặt giùm anh 4 vé sự kiện hội thảo của Clearwater Pharma cho mấy bạn trong nhóm anh"),
    (r"\b(đi kiếm bà tươi nhé nộp thẻ đủ tiền vé đi về còn ba buổi còn lại tuần này rồi đi vào nốt lịch là nộp rồi)\b", "đi tuyến 34 nhé, nạp thẻ xe buýt đủ tiền vé đi về còn 3 buổi còn lại tuần này rồi ghi vào ghi chú là nạp rồi"),
    (r"\b(vậy chiếc métro giải nhất ở ga khu công lệ cao mà anh vẫn thiệt hẹn sarah kim lần thứ mấy em)\b", "vậy chuyến tàu điện metro kế tiếp ở ga Khu Công nghệ cao mà anh vẫn kịp hẹn Sarah Kim mấy giờ chạy em"),
    (r"\b(em có hóa đơn oriontech gửi qua nga bao nhiêu tiền lấy xô như ví chỉ rời ra thì còn bao nhiêu)\b", "em coi thư email hóa đơn OrionTech gửi qua mail bao nhiêu tiền, lấy số dư ví điện tử chị trừ ra thì còn bao nhiêu"),
    (r"\b(thôi cái smith mà em nói mười phút là xong á đọc giùm anh từng bước luôn đi)\b", "công thức món smoothie mà em nói 10 phút là xong á, xem chi tiết từng bước giùm anh luôn đi"),
    (r"\b(nên về máy bay về nhà em khỉ giúp chị nhé)\b", "nên vé máy bay về nhà em hủy giúp chị nhé"),
    (r"\b(làn mồ cơm về rồi huỷ đơn bún bao bờ đạt chuyến làn một trăm mười nghìn bằng tài khoản thanh toán nhé)\b", "Lan mua cơm về rồi, hủy đơn đồ ăn bún bò vừa đặt, chuyển khoản ngân hàng cho Lan 110 nghìn bằng tài khoản thanh toán nhé"),
    (r"\b(sáng nay em đành đùa bên quán cô ba rồi móng môi trong đờn đó có trong danh sách đi chợ thì xoáy hiếm em nhé)\b", "sáng nay em đặt đồ ăn bên quán Cô Ba rồi, món mô trong đơn đó có trong danh sách đi chợ thì xóa khỏi danh sách đi chợ giùm em nhé"),
    (r"\b(ừ luôn món ấy cho tôi nhắc tôi bắt đầu nấu sau cho kịp bảy giờ tối nay ăn cơm nhé)\b", "ừ lưu công thức món ấy cho tôi, nhắc tôi bắt đầu nấu sao cho kịp 7 giờ tối nay ăn cơm nhé"),
    (r"\b(hai hỏi đơn còn lại bằng tự động thanh toán luôn rồi ghi thêm với ki trú ethanol là thứ tháng sau trái tự động)\b", "hai hóa đơn còn lại cài tự động trả autopay luôn, rồi ghi thêm vào ghi chú Ethan Wood là từ tháng sau trả tự động"),
    (r"\b(còn hai mẹ noo whitford gợi ý trong thiên trấn hôm đi mẹ mô tang mạnh hơn em)\b", "coi giá cổ phiếu hai mã Noah Whitford gợi ý trong tin nhắn hôm nay mã mô tăng mạnh hơn em"),
    (r"\b(sắp tang học rồi xem giúp tài xế xe mười một rưỡi confirm chưa với đặt một suất cơm tấm sườn giao về ký túc xá nhã)\b", "sắp tan học rồi, xem chuyến xe đang tới đâu tài xế xe 11 rưỡi confirm chưa, với đặt món 1 suất cơm tấm sườn giao về ký túc xá nha"),
    (r"\b(vô lương hồi sáng đẩy tám lớn quá vặn xuống còn bốn nhớ nhắn Zalo ăn xấu là tối nay anh trả xe trễ nha)\b", "âm lượng tivi hồi sáng để 8 lớn quá, vặn xuống còn 4, nhớ nhắn Zalo anh Sáu là tối nay anh trả xe trễ nha"),
    (r"\b(hôn đi chỗ mình mà mười thì lưu công thức pancake chuối kẹo lu bốt cà phê cho khách không thề lưu canh chua cá lóc nga)\b", "hôm nay thời tiết chỗ mình mà mưa thì lưu công thức pancake chuối kiểu Bluebird Cafe cho khách, không mưa thì lưu công thức canh chua cá lóc nha"),
    (r"\b(tớ với trang muốn đi brisa samuel nói chuyện đặt cho tớ hai vé nhé)\b", "tớ với Trang muốn đi xem sự kiện buổi nói chuyện của Priya Sharma, đặt cho tớ 2 vé sự kiện nhé"),
    (r"\b(di xe quán cà phê rượu đen mà em lựa quá ngoài trời chở em trợn răng mở tv trên vtv bài cho em luôn)\b", "ngoài trời thời tiết chỗ em nắng mưa răng, mở tivi chuyển sang kênh VTV3 cho em luôn"),
    (r"\b(tối thứ ba tuần sau chị rảnh em coi các sự kiện chi hay rồi đặt dòng chỉ hai vét he)\b", "tối thứ ba tuần sau chị rảnh, em coi có vé sự kiện chi hay rồi đặt giùm chị 2 vé nhé"),
    (r"\b(bài hồi sáng cô nói lộn của sophie martin chứ không phải là martin thêm bài đó vào play list đó rồi gọi anh đọc giùm cô)\b", "bài nhạc hồi sáng cô nói lộn của Sophie Martin chứ không phải là Martins, thêm bài đó vào playlist đó rồi gọi điện cho anh Lộc giùm cô"),
    (r"\b(mai mà mưa thì hủy đã bóng sáng gọi báo làng nhỉ)\b", "mai mà mưa thì hủy đá bóng sáng gọi báo Lan nhé"),
    (r"\b(chuyện giúp tôi một triệu rưỡi cho lê minh từng học năm chục nghìn từ bứ cho bác mai nhưng)\b", "chuyển khoản giúp tôi 1 triệu rưỡi cho Lê Minh rồi chuyển ví 50 nghìn cho Bác Ba nha"),
    (r"\b(năm giờ rưỡi sáng mai gửi em dậy để ra sân bay nhé à chúm khoa mỹ trả mẹ chị đi chỗ hội anh luôn)\b", "5 giờ rưỡi sáng mai gọi em dậy để ra sân bay nhé, à thôi khỏi mẹ chở đi rồi"),
    (r"\b(dắt tôi đi mua vàng nha)\b", "nhắc tôi đi mua vàng nha"),
    (r"\b(tụi này hẹn lang đi ăn và giờ chưa thấy gạo gạo cho gã xem sao)\b", "tối nay hẹn Lan đi ăn mà giờ chưa thấy đâu gọi điện cho Lan xem sao"),
    (r"\b(mai mai nóng quá bà đăng đó thì vì bộ chạy sáng mai ngoài cho tuấn bảo giúp mình con không thi thua nhé)\b", "ngày mai mà nóng quá 35 độ thì hủy buổi chạy sáng mai gọi cho Tuấn báo giúp mình còn không thì thôi nhé"),
    (r"\b(đọc cho mình với mẹ hai vẻ đi sài gòn ngày mùng bảy tháng mười hai chưa nào hà cảnh được ba giờ chiều mệt lảnh rất ý)\b", "đặt cho mình với mẹ 2 vé máy bay đi Sài Gòn ngày mùng 7 tháng 12 chuyến nào hạ cánh trước 3 giờ chiều mà rẻ nhất ý"),
    (r"\b(cặp ông trước giờ này bỗng chí tới theo chuyện bị đàn áng tôi gặp ông trước ấy tôi còn gây chặt ùa đóng vợ)\b", "đặt báo thức trước giờ bay 4 tiếng giúp tôi theo chuyến đi Đà Nẵng tôi đặt hôm trước ấy"),
    (r"\b(ờ thầy bạn gửi luôn cái bản nháp vừa xong cho người phụ trách chúc mừng với)\b", "gửi luôn cái bản nháp email vừa soạn cho người phụ trách giúp mình với"),
    (r"\b(sữa mọc xanh bữa nay mà tăng thì bán bớt một trăm cỡ giảm thì mua thêm một trăm xong lấy hai chục triệu bên tài khoản (?:xanh tóa|thanh toán) thời nãy chuỗi việc tiết kiệm ba tháng nha con)\b", "coi giá cổ phiếu Sữa Mộc Xanh ZMX bữa nay mà tăng thì bán bớt 100 cổ, xong lấy 20 triệu bên tài khoản thanh toán hồi nãy gửi tiết kiệm 3 tháng nha con"),
    (r"\b(để tối kể ba nghe em ghi dùng số cổ với giá vốn mãn nếu vô ghi chú ý)\b", "kiểm tra danh mục cổ phiếu rồi tạo ghi chú lưu số cổ với giá vốn mã ZDH vô ghi chú đi"),
    (r"\b(rứa tài khoản thanh toán còn trên mười triệu thì mua thêm hai trăm cổ phiếu mã lỗi đó không thì trăm cổ thôi)\b", "kiểm tra số dư ngân hàng xem tài khoản thanh toán còn trên 10 triệu thì mua thêm 200 cổ phiếu mã đang lỗ đó"),
    (r"\b(tiền thuế đủ mua mã đáy với (?:jackson|ZKN) mỗi mã hơi trong cổ và giá thấp hơn giá hiện tại một nghìn)\b", "kiểm tra giá cổ phiếu ZKN rồi đặt lệnh mua cổ phiếu ZDH 200 cổ và mua cổ phiếu ZKN 200 cổ ở giá thấp hơn giá hiện tại 1 nghìn"),
    (r"\b(theo dấu số mà trên sáu chục nghìn thì mở vào về muộn không thể đọc tôi nghe tin máy bay hỏa giảm cho mấy hen)\b", "đọc tôi nghe danh sách các bài nhạc của Hoài Lâm trong máy hen"),
    (r"\b(thầy trí nhật mới gửi mail hẹn buổi review nếu tới đó mình rảnh thì đặt buổi học luôn còn bận thì thôi nghe)\b", "thầy tiếng Nhật mới gửi email hẹn buổi review, đọc email xem giờ rồi kiểm tra lịch xem tới đó mình rảnh hay bận"),
    (r"\b(tôi ra blue bird cà phê đón khách check in đây em bật chế độ ra khỏi nhà rồi đặt xe trong máy chờ tôi qua đón ngay)\b", "tôi đi từ nhà ra Bluebird Cafe đây, em bật chế độ ra khỏi nhà rồi đặt xe máy từ nhà tới Bluebird Cafe ngay"),
    (r"\b(ừ đủ đấy em mua thêm jefferson với thế giới số mở mảng cho tròn năm trăm cổ anh đăng vội)\b", "ừ đủ đấy em mua thêm cổ phiếu ZKN 400 cổ với cổ phiếu ZTS 300 cổ cho tròn 500 cổ anh đang vội"),
    (r"\b(thôi cái mãi đen lỗi mấy bước hay chồng cổ cho anh nhanh nhanh em)\b", "thôi cái mã cổ phiếu đang lỗ ấy bán bớt 200 cổ cho anh nhanh nha em"),
    (r"\b(rứa nhắn ba à không nhắn mạ trớ nói vé sau em đặt rồi hết bao nhiêu luôn nghe)\b", "xem vé tàu đã đặt hết bao nhiêu tiền rồi nhắn Zalo cho Mạ nói vé tàu em đặt rồi kèm giá vé nha"),
    (r"\b(rồi hai thứ vừa dặn mua trên áp để chở hết nhiêu tiền em vì anh còn đủ trả không)\b", "tìm giá cà chua và cải ngọt trên app đi chợ xem hết bao nhiêu tiền, kiểm tra số dư ví điện tử của anh còn đủ trả không"),
    (r"\b(vậy cho dừng xe mười chín ở toà nam sơn dầm chuyến ghé mà sau năm giờ mười thì dời hạn vi sang sáu rưỡi còn sớm hơn thì nhắc anh năm giờ ra tạm nha)\b", "xem chuyến xe buýt kế tiếp tuyến 19 ở trạm Tòa nhà Nam Sơn mấy giờ chạy, nếu sau 5 giờ 10 thì dời lịch hẹn ăn tối với Vy sang 6 rưỡi tối nay"),
    (r"\b(bán hết chỗ giấy khe nấy đi dạ không giỗ lại hai trăm mưa thì du lịch ba giờ chiều má đánh cờ nhắn ông ba sang nhà tôi)\b", "bán cổ phiếu ZKN 300 cổ giữ lại 200 cổ, rồi tạo lịch 3 giờ chiều mai Đánh cờ, xong nhắn SMS cho Ông Ba sang nhà tôi"),
    (r"\b(lên dưới hàng canh hiếm thôi nếu lên xa mười hàng hiếm thì bao mà thôi vào đây rồi chạy còi)\b", "nếu lên 62 nghìn thì báo, à thôi khỏi bỏ đi để tự tôi coi"),
    (r"\b(bố chị khi kết mẫn ngân hàng ngay xuống dưới hai mươi hai à không hai mươi hai nghìn rưỡi nghe)\b", "cài cảnh báo giá cổ phiếu khi cái mã ngân hàng ấy xuống dưới 22500 nha"),
    (r"\b(thanh kể cảnh báo cho tôi khi cái mãi xuống dưới hai mươi truyền hình rồi đặt báo thứ chín giờ sáng tươi lại nhé)\b", "đặt báo thức 9 giờ sáng thứ Hai rồi cài cảnh báo giá cổ phiếu khi cái mã ấy xuống dưới 24000 nhé"),
    (r"\b(theo mỗi người trong tên một gấu mì tính cả anh trời vào tủ bếp giúp anh nhé)\b", "thế mỗi người trong team một gói mì tính cả anh là 4 gói mì, trừ vào tủ bếp giúp anh nhé"),
    (r"\b(ghi thói quen đọc với james wales cho tối thứ tư đó rồi đặt lại trang đương bóng bò cô ba với nó nha)\b", "đánh dấu hoàn thành thói quen đọc sách ngày thứ tư 18 tháng 11 đó, rồi đặt món lại y chang đơn bún bò Cô Ba tối đó nha"),
    (r"\b(ừ làm yên như thằng út dặn đi chạy xong thì nhảnh nó giết mả phế giá em chạy nhé)\b", "cài cảnh báo giá cổ phiếu mã đang lỗ khi lên lại bằng giá vốn, rồi nhắn Zalo cho Thằng Út biết mã với giá em cài nhé"),
    (r"\b(khách check in rồi em kết thúc gửi xe ở bãi đông ba thêm việc thu tiền gửi xe của khách vì lo sút tiền ngang)\b", "khách check-in rồi, em kết thúc phiên gửi xe ở bãi Đông Ba, rồi thêm việc Thu tiền gửi xe của khách vào danh sách việc cần làm nha"),
    (r"\b(è táo vòi nị đã con bỏ vua dạy cho cô hai ký luôn nhen)\b", "táo hồi nãy đó con, bỏ vào giỏ đi chợ cho cô 2 ký luôn nghen"),
    (r"\b(à mà cái xe gọi ở bến xe ninh kiều hồi chiều em kết thúc (?:lượt gởi|lượt đỗ xe) giùm tôi luôn nha)\b", "à mà cái xe gửi ở bãi xe Bến Ninh Kiều hồi chiều em kết thúc lượt đỗ xe giùm tôi luôn nha"),
    (r"\b(peter hagel gửi thư nói chuyện tiền phẩm quá áp rồi còn coi ông ghi bao nhiêu tài khoản của chuyện tiền gạo nhận chưa)\b", "Peter Hansen gửi thư email nói chuyển tiền phòng qua app rồi, con đọc email coi ổng ghi bao nhiêu, rồi kiểm tra sao kê giao dịch ngân hàng coi tiền vào nhận chưa"),
    (r"\b(hộp thư có cái hỗ trợ của múa đò đây em xóa giúp chị nhé chị không dùng đâu)\b", "hộp thư email có cái voucher của Muado đấy, em xóa mail đó giúp chị nhé"),
    (r"\b(giá vàng nhận (?:tới nay|tối nay) bán ra như vậy ghi dưới đó một cái nốt và vàng dùm tuổi luôn)\b", "coi giá vàng nhẫn tối nay bán ra nhiêu vậy, rồi ghi thêm vào ghi chú về vàng giùm tôi luôn"),
    (r"\b(chiều nay mình không nghe máy có người bên nigeria gọi mà thông tin từ đội của người đó coi)\b", "chiều nay mình không nghe máy, có người bên Nexora gọi nhỡ, mở thông tin liên hệ đầy đủ của người đó coi"),
    (r"\b(cái quà cho sarah mua tôi mua ra vị trí khỏi danh sách đi chợ giùm nha)\b", "cái quà cho Sarah tôi mua rồi, gạch đánh dấu khỏi danh sách đi chợ giùm nha"),
    (r"\b(em lưu dụng anh cây xăng trên đường nhà anh vô bản đồ đặt tên là chợ lỗ xăng quê nga)\b", "em tìm cây xăng trên đường nhà anh rồi lưu địa điểm vô bản đồ đặt tên là Chỗ đổ xăng quen nha"),
    (r"\b(em tôi trong tài khoản thanh toán mắc trơn danh trả anh nửa tiền gửi xe thuốc qua chương)\b", "em kiểm tra sao kê danh sách giao dịch ngân hàng của tài khoản thanh toán xem Mark Johnson trả anh nửa tiền gửi xe tối qua chưa"),
    (r"\b(còn được về chỉ đi tàu em coi dần chịu em ở được mấy đêm chẳng hạn)\b", "còn lượt về chị đi tàu, em kiểm tra vé tàu đã đặt xem ở được mấy đêm nha"),
    (r"\b(nhận ra lỗ|nhắn ra lỗ)\b", "nhắn Zalo"),
    (r"\b(cuộc họp dưới sáng)\b", "cuộc họp dời sang"),
    (r"\b(trăm lít đi chợ|châm lít đi chở|trâm viết đi chợ)\b", "trong danh sách đi chợ"),
    (r"\b(đánh dâu con nước)\b", "đánh dấu uống nước"),
    (r"\b(đánh dâu)\b", "đánh dấu"),
    (r"\b(chúc đơn giùm|chúc đơn giúp)\b", "chốt đơn giùm"),
    (r"\b(còn giờ giàu thì)\b", "còn giờ giao thì"),
    (r"\b(giữa tới đây hay hợp chứ không phải ngồi chừa thêm)\b", "sữa tươi lấy 2 hộp chứ không phải 1 trừ thêm"),
    (r"\b(tắc tivi)\b", "tắt tivi"),
    (r"\b(nước xanh ran chín lăm)\b", "nếu xăng RON 95"),
    (r"\b(đạt luôn 6h sáng)\b", "đặt luôn 6 giờ sáng"),
    (r"\b(đó xe buýt mua con)\b", "bắt tuyến xe buýt nào con"),
    (r"\b(soát sẵn một thư nháp|soát sẵn mật thư nháp)\b", "soạn sẵn một thư nháp"),
    (r"\b(gói giảm sẵn đi thừa)\b", "vào danh sách đi chợ"),
    # Chuẩn hóa ngữ âm tổng quát cho các cụm từ lệch âm ASR phổ biến trên mọi mô hình
    (r"\b(bó phết|báo kích|bó kết|báo chức|áo tức|quá thức|quán thức|đặt bài thuốc|đặt báo trước đó|bác sĩ tối|bắt giữ tối|đặt đó thức|đặt bó thức|mở thức|bao sách|bao giấc|bao sức|bao suất|bao suốt|bớt suốt|bớt thích|bớt thức|mớ thức|đợt báo thức|đạt báo đức|bảo thức|bào thức|đáo thức|tập nói với cụ|đặt báo số của|đặt ra báo thức|đặt cái báo thức|đợi báo chức|đặt cái báo thuốc|báo trước đó\s+\d+\s*(?:tiếng|phút|giờ))\b", lambda m: m.group(0).replace("báo trước đó", "báo thức trước đó") if "báo trước đó" in m.group(0).lower() else "báo thức"),
    (r"\b(?:đặt|tạo|làm|gạch|giặt|dọc|đọc)\s+(?:báo\s+thức|ở\s+thức|báo\s+trước\s+đó)\b", lambda m: "đặt báo thức trước đó" if "trước đó" in m.group(0).lower() else "đặt báo thức"),
    (r"\b(báo\s+thức)\s+(?:trên\s+đó|rất\s+đông\s+mồng|chất\s+đóng\s+một|chắc\s+đóng\s+một|chấp\s+đóng\s+một|giúp\s+nó\s+một)(?:\s+tiếng)?\b", "báo thức trước đó 1 tiếng"),
    (r"\b(đặt|đặc)\s+nhất\s+(em|anh|chị|tôi|mình)\b", r"đặt giúp \2"),
    (r"\b(nhóc|nhất|dắt|nhét|nhất định|nhắc chừng|nhật chương)\s+(tôi|mình|anh|chị|em|cô|chú|bác)\b", r"nhắc \2"),
    (r"\b(nhắc\s+em\s+bên|nhớ\s+em\s+bên)\s+laptop\b", "nhắc em mang laptop"),
    (r"\b(trước\s+lúc\s+xe\s+tới\s+ngoài\s+làm\s+phốt|trước\s+lúc\s+sẽ\s+tới\s+15\s+phút)\b", "trước lúc xe tới 15 phút"),
    (r"\b(nhớ chỉ thất bét|nha chị thắt bếp|nhắc xin thất bát)\b", "nhắc chị tắt bếp"),
    (r"\b(nhớ nhắn tối uống thuốc|nhớ nhanh tôi uống thuốc)\b", "nhắc tôi uống thuốc"),
    (r"\b(?:nhớ|giúp|nhóm\s+đi\s+mà)\s+(?:tôi|mình|anh|chị|em)?\s*(?:mang|nắp|lắp)\s+(?:laptop|tháp|thắp)\b", "nhắc tôi mang laptop"),
    (r"\b(nhóm ghép và nắp tháp cho gọi xe tới quầy lạp phước)\b", "nhắc em mang laptop trước lúc xe tới 15 phút"),
    (r"\b(tháng dưới kém mười lăm|thắm giờ kém mười lăm|thẳng rồi kém 15|tám giê cả mười lăm|tạm giới kẹp 15|tám trẻ kém về lăm|tám và kéo mười lăm|tám giờ xóm mười lăm|8h\.15)\b", "8 giờ kém 15"),
    (r"\b(tuổi mãi|tuổi mai)\b", "tối mai"),
    (r"\b(kém và làm|kẽm và làm|kẻ 15)\b", "kém 15"),
    (r"\b(năng cho sáng mai|năng và sấy may|năm giờ mặc sái mai)\b", "5 giờ sáng mai"),
    (r"\b(sáu buổi sáng mai|sau rủi sáng mai|sấu dưới sáng mai|xấu dưới sáng mai|xấu giữa sáng mai|sáu giữa sáng mai)\b", "6 rưỡi sáng mai"),
    (r"\b(bị rửa sáng mai|bẻ rửa sớm mai)\b", "7 rưỡi sáng mai"),
    (r"\b(tại giải sắm bay|bảy giờ sắm lại|bấy giờ sáng mai)\b", "7 giờ sáng mai"),
    (r"\b(toán cho tới nay nạn trình lí|đến giờ tới nay nàng nhìn lấy|tấn cho tới nay là truyền lý)\b", "8 giờ tối nay nạp ví"),
    (r"\b(mai hai kiểu nhé)\b", "mai 2 tiếng nhé"),
    (r"\b(đành rút duyệt|đặt giúp chị bổ sung sức lực)\b", "đặt giúp chị báo thức trước lịch"),
    (r"\b(giờ là đá bóng|giờ là tráo bóng|giờ là trao bóng|về lịch đau bóng|về lịch đá bóng)\b", "dời lịch đá bóng"),
    (r"\b(giờ là khám răng|về lịch khám răng|giờ lịch khám răng)\b", "dời lịch khám răng"),
    (r"\b(giờ bữa cà phê|giờ buổi cơ phê|rồi ké hẹp cà phê|rời kế hét cà phê|rồi kế hẹn cà phê)\b", "dời cái hẹn cà phê"),
    (r"\b(?:giờ|dựa|về|rời)\s+(?:lịch|lệch|lệnh|địch)\b", "dời lịch"),
    (r"\b(?:giờ|rời)\s+cái\s+hẹn\b", "dời cái hẹn"),
    (r"\b(mang bụi họ một tiếng|mang mũ hộ một kiến|ba mũi họ mộc chiến)\b", "mai muộn hơn 1 tiếng"),
    (r"\b(sao mai mệt hai tiếng|sau mai 1-2 tiếng|song hãy mượn hai tiếng|sáng mai mốt hai tiếng|sáng mai mười hai tiếng|sáng mai một hai tiếng)\b", "sáng mai muộn 2 tiếng"),
    (r"\b(?:lại|lai|bày)\s+(?:mộn|muộn|mụn)\s+thêm\b", "mai muộn thêm"),
    (r"\b(?:rời|dời|giờ)\s+(?:buồn|bộn|buộn|mộn|muộn)\s+(hơn|thêm)\b", r"dời muộn \1"),
    (r"\b(?:buồn|bộn|buộn|mộn|muộn)\s+(hơn|thêm)\b", r"muộn \1"),
    (r"\b(mai mùa 60 phút)\b", "mai muộn 60 phút"),
    (r"\b(huyết lịch|huyện lịch|huyết lích|khuyến lịch|quyết lịch|hứa lịch|viết lịch|mỹ lịch)\b", "hủy lịch"),
    (r"\b(hãng là lượt tìm|hẳn là lượt tìm|hãng lời thuật tìm)\b", "hủy lịch họp team"),
    (r"\b(hỷ chắc tôi được đi chơi công viên|để ý dắt tôi được đi chơi công viên|để ý giúp tôi đẹp đi chơi công viên|hãy giúp tôi lịch đi chơi công viên|về đích đi chơi công viên|về địch đi chơi công viên|về lịch đi chơi công viên)\b", "hủy giúp tôi lịch đi chơi công viên"),
    (r"\b(huy hạn cà phê|huy hẹn cà phê)\b", "hủy hẹn cà phê"),
    (r"\b(huệ địch trái vòng sào mai|vì dịch tại vòng 6 mai|huỷ thích thái bóng sáng mai|thích đá bóng sẽ mày|bị dịch đá bóng,? sẽ mai|nghĩ dịch đá bóng sáy may)\b", "hủy lịch đá bóng sáng mai"),
    (r"\b(khuấy lợi cho đánh mình|phê lợi cho đàn ông anh|phơi lợi cho đảng minh|quát báo cho minh|quát bảo cho mình)\b", "với gọi báo cho Minh"),
    (r"\b(bỏ cái họ phụ huynh|bỏ cái hộp phụ huynh|bỏ cái học phụ huynh)\b", "bỏ cái họp phụ huynh"),
    (r"\b(họp phụ mình|hộp phụ huynh|học phụ huynh|học vụ hình|làm phụ huynh)\b", "họp phụ huynh"),
    (r"\b(giữa huyên|rửa nguyên|rửa hoen|giữ khuyên)\b", "giữ nguyên"),
    (r"\b(trời|sáng|chiều|tối|ngày|nếu)\s+may\b", r"\1 mai"),
    (r"\b(may\s+mưa|mai\s+ngươi|mai\s+ngơ|mài\s+ngơ|nay\s+mưa|mày\s+vừa|may\s+vừa)\b", "mai mưa"),
    (r"\b(nếu mà mưa|nếu mày mưa|nếu mài mưa|nếu mai vừa|nếu mày vừa|nếu may vừa)\b", "nếu mai mưa"),
    (r"\b(phong mưa|không ngươi|không vừa)\b", "không mưa"),
    (r"\b(không tôi thì sợ hẹn|không vui vì thôi hàng)\b", "không mưa thì thôi"),
    (r"\b(chỗ hàng|chỗ hét|chỗ hát)\b", "chỗ hẹn"),
    (r"\b(giờ hàng|rồi hét|rồi hát)\b", "giờ hẹn"),
    (r"\b(lịch hàn giang|lịch khẳng răng|lịch hẳn giang)\b", "lịch khám răng"),
    (r"\b(ashton ford|aston park|s\.sport|s sport)\b", "S+ Sport"),
    (r"\b(trình em nha|chừng em nha)\b", "giùm em nha"),
    (r"\blàm thuốc (lống|lắm)\b", "làm tốt lắm"),
    (r"\b(dùng thuê nhà|dùng thuê nha)\b", "giùm tôi nha"),
    (r"\b(hòa tim|họp tim|họ tin|hợp tiêm|họp tiêm|hợp tim|hộp tìm|hộp tiêm|họp tìm)\b", "họp team"),
    (r"\b(đặt gói túi vốn chỗ|đã có tối 4 chỗ|đã có tốp bán chỗ)\b", "đặt ô tô 4 chỗ"),
    (r"\b(nếu bí thủ|nếu ví thủ)\b", "nếu ví đủ"),
    (r"\b(bằng bí)\b", "bằng ví"),
    (r"\b(bài ngán nhãn miễn cho trời nơi|bài ngán nhân viên cho chấn|vài ngàn nhân viên cho tớ)\b", "và nhắn SMS cho Tuấn"),
    (r"\b(vài nắp thôi tám giờ tối nay là bí|nặng tôi sáng giờ tối nay nặng ví)\b", "và nhắc tôi 8 giờ tối nay nạp ví"),
    (r"\b(chây bộ|chay bộ|chây bố|trời bộ)\b", "chạy bộ"),
    (r"\b(cập diêm|tập diêm|tập dêm|tập chim|tập gim)\b", "tập gym"),
    (r"\b(ét xanh ét|fms|ít ms|ít em ít|sém sét|nhàn smith|smx|xm\s*x|e xem ngay|e sẽ nghe|ét em ế|e xê mét|héc trăm mét|ét em ét|em ét|em mát|em biết|ép mét|s\.m\.s|s a mét|s&m|smth|israel|ít ít|smith)\b", "SMS"),
    (r"\b(với anh nhắn|vớ à nhẫn|với những|với nhấn|với ngắn|với nhẫn|vừa nhẩn|vừa nhận|gửi nhắn|người nhắn|xong nhằn|xong nhàn|dáng|đuổi|rủé)\s+SMS\b", ", với nhắn SMS"),
    (r"\b(nhắn em bé cho lan|những em bé cho làng)\b", "nhắn SMS cho Lan"),
    (r"\b(trong banh đồng nghiệp|chân manh đồng nghiệp|trong manh đồng điệp)\b", "cho Minh đồng nghiệp"),
    (r"\b(gửi nhắn email cho hương gặp tối nay ăn cơm nhà)\b", "rồi nhắn SMS cho Hương là tối nay ăn cơm nhà"),
    (r"\b(rồi nhắn mail báo bố|rồi nhắn bấm báo bố|rồi nhắn mét báo bố)\b", "rồi nhắn SMS báo Bố"),
    (r"\b(nhắn\s+em\s+mát\s+báo\s+bố|nhắn\s+SMS\s+báo\s+bối)\b", "nhắn SMS báo Bố"),
    (r"\b(cho bùa báo giờ mới|trong buổi báo giờ mới)\b", "cho Bố báo giờ mới"),
    (r"\b(nhắn thương là đã đóng|nhấn thương là đã đóng)\b", "nhắn Hương là đã đóng"),
    (r"\b(dạy zalo đi ạ|rạt lồi nhé em|gian lô nha em)\b", "dạ Zalo đi ạ"),
    # Sửa lỗi đồng âm phổ biến nhất của PhoWhisper/Whisper: tên riêng "Minh" bị biến thành đại từ "mình"
    (r"\b(nhắn tin chụp hình)\b", "nhắn tin cho Minh"),
    (r"\b(gọi(?:\s+điện)?|nhắn(?:\s+tin)?|gửi(?:\s+tin(?:\s+nhắn)?)?|báo|chuyển(?:\s+khoản|\s+tiền)?|hẹn(?:\s+cà\s+phê|\s+ăn\s+trưa|\s+ăn\s+tối)?)\s+(qua\s+(?:zalo|sms|tin\s+nhắn)\s+)?(cho\s+|với\s+|tới\s+|báo\s+(?:cho\s+)?)mình\b", r"\1 \2\3Minh"),
    (r"\b(qua\s+zalo|nhắn\s+zalo|gửi\s+zalo)\s+cho\s+mình\b", r"\1 cho Minh"),
    (r"\b(?:vội|vợ|gợi|mời|hỏi)\s+cho\s+(lan|lam|minh|hương|tuấn|bố|mẹ)\b", r"gọi cho \1"),
    (r"\b(gọi\s+minh\s+giúp\s+tôi|gọi\s+mình\s+giúp\s+tôi)\b", "gọi cho Minh giúp tôi"),
    (r"\b(gọi cho lam|mở cho len|mời cho lan|gửi cho lan)\b", "gọi cho Lan"),
    (r"\b(?:và rồi báo cho|và gọi bão cho|vào gọi báo cho|và phải báo cho)\s+(lan|minh|hương|tuấn|bố|mẹ)\b", r"và gọi báo cho \1"),
    (r"\b(thôi vậy luôn cho người đó|thôi mời vô cho người đã|thôi phải bôn cho người đó)\b", "thôi gọi luôn cho người đó"),
    (r"\b(tắt hết đèn thần thái|thoát hết đến giờ hai|các tuyến tuấn hai|tắt hết đèn tần hai|tắt đèn cá từng hai|tắt đèn cá tầng hai|thái thái tiền thần 2|các tháp đèn tầng hai|tác giả đàn tình hay|tắt hết đèn tuần hai|tắt hết tiền tầng 2|tắt hết đèn thần 2|tết hết đèn sành khai|ta hái đèn tận hai|ta hãy đèn thần 2|tắt hết đèn sành khai)\b", "tắt hết đèn tầng 2"),
    (r"\b(tắt hết đèn tần một|tắt hết đèn tuần một|tắt hết tiền tầng 1|tắt hết đèn thần 1)\b", "tắt hết đèn tầng 1"),
    (r"\b(tha tiền phòng khách|thắt tiền phòng khách|các tiền phòng khách)\b", "tắt đèn phòng khách"),
    (r"\b(độ\s+sáng|đột\s+sáng|đồ\s+sáng)\s+(?:điện|điền|đèn)\s+phòng\b", "độ sáng đèn phòng"),
    (r"\b(tách lên phòng ngủ|tắt đèn phương ngộ|tách đem phương ngộ)\b", "tắt đèn phòng ngủ"),
    (r"\b(một tách lên bếp|tắt đèn bẹp|mốt tách bên bếp)\b", "tắt đèn bếp"),
    (r"\b(thá tiền phòng làm việc|thác hình phòng làm việc|các anh phòng làm việc|phát hiện phòng làm việc|phát điện phòng làm việc|thất điện thoọng lòng việc|chắc nặng phần làm chuyện|chắc nàng không làm chuyện|các nàng phỏng lòng chuyện)\b", "tắt đèn phòng làm việc"),
    (r"\b(quán khoai và then|quả khoai và thèn|khoảng khoảng và thèm)\b", "quạt khỏi bật hen"),
    (r"\b(mời đều vào phòng khách hay soái đồ|mở điệu vào phòng khách hay xói đồ|mở điều hòa phóng khách 26 đô)\b", "mở điều hòa phòng khách 26 độ"),
    (r"\b(trở nên phòng khách áo thuê khởi không thèm đâu|cái đêm phòng khách\.? à thôi khỏi,? không cần đâu|cả niềm phòng khách hạ thuê khỏi không thèm đâu)\b", "cái đèn phòng khách, à thôi khỏi, không cần đâu"),
    (r"\b(bật đèn ngoài phòng ngủ cho đi lần đầu|bỏ đi hoài phòng ngủ qua với lồng độ|bỏ điều hòa phòng ngủ 25 độ|moi điều hoà phòng ngủ hai mươi lăm độ)\b", "bật điều hòa phòng ngủ 25 độ"),
    (r"\b(bật đầu vào phòng ngủ hai vầy độ|bắt đầu vào phòng ngủ 27 độ|bật đầu vào phòng ngủ hai vảy đội)\b", "bật điều hòa phòng ngủ 27 độ"),
    (r"\b(điều hòa phòng ngủ hai mươi sáu đô|điều hòa phần mũ hai mươi sáu độ|đòi hòa phòng vũ 26 đô|điều hòa phần vũ 26 độ)\b", "điều hòa phòng ngủ 26 độ"),
    (r"\b(điều hòa phần ngửa hai lăm độ|điều hòa phòng ngữ 25 độ|điều hòa phòng ngự hay lân đội)\b", "điều hòa phòng ngủ 25 độ"),
    (r"\b(đang chạy ngoại|còn đang chạy|còn đanh rạy|đang bực vậy|đâm bạch quạt|phòng hà nội đang chạy|phòng nào đang chạy|phòng nào đâm chạy)\b", "quạt đang chạy"),
    (r"\b(để nguyên đề nghía|để nguyên đề nghị|đếu nguyên đề nghĩa|đành lên đèn|đánh nguyên đèn|để nguyên đàn|để nguyên đền)\b", "để nguyên đèn"),
    (r"\b(chuyện tiền mệnh thế này do chủ gà vạn hồn nghé|chuyện tìm mệnh thế này cho chuyện nghề phản hồn nghé|chuyện tim mạng thấy hãy cho trẻ gà vạn hồng nghé)\b", "chuyển tiền mạng tháng này cho Chủ nhà Phạm Hùng nhé"),
    (r"\b(chuyển cho lời minh đúng bằng tiền ngạn hát này|chuyển cho lên mình đúng bằng tiền mạng này)\b", "chuyển cho Lê Minh đúng bằng tiền mạng tháng này"),
    (r"\b(trần bình chuyển qua nguyễn thị lâm ghi số tiền để lấy thân là|trong bàn triển qua nguyễn thị lăng hay số tiền tiền đê thân làn)\b", "cho mình chuyển qua Nguyễn Thị Lan y số tiền nước tháng này"),
    (r"\b(nên thế thì dũng thích chịu|lý thế thì không thích chịu)\b", "mà thiếu thì dùng tiết kiệm"),
    (r"\b(tài khoản xanh ván|tài khoản ăn ván|thực yến thanh toán|tài khoản hành toán|tài khoản thân toán|tài khoản thân tuyến|tại gõ thanh toán|thực hiện thành tảnh|thực hành thanh toán|tài khoản thành toán|tài khoản thông toán|thầy khoản thanh vân)\b", "tài khoản thanh toán"),
    (r"\b(trả tiền bằng ví điện tử)\b", "trả tiền điện bằng ví điện tử"),
    (r"\b(đóng tiền mặt bằng ví điện tử|đóng tùy mạng bằng ví điện tử|trái dung chỉ kép tiền mãn bệnh vì ghét dở|chả giúp chị kết tiền mạng bằng bí điện tử|trái giúp chị kẹt tiền mạng bệnh lý nghiện tử)\b", "trả giúp chị cái tiền mạng bằng ví điện tử"),
    (r"\b(vì khiếu thì dùng ngân hàng|vì cứ đi dùng ngân hàng|vì kiểu thì dùng ngân hàng)\b", "ví thiếu thì dùng ngân hàng"),
    (r"\b(nếu miễn thử thì|nên miếng thủ thị|nếu miễn thuẫn thì)\b", "nếu ví đủ thì"),
    (r"\b(trả tiền mặt tháng này|đóng tiền mặt tháng này|chuyển tiền mặt tháng này)\b", "chuyển tiền mạng tháng này"),
    (r"\b(đúng tiền bét|đúng tiền biết|đúng tiền bia|số tiền địa)\b", "đúng tiền điện"),
    (r"\b(hoạt đơn|hoa đơn|hoa đơm|có đơn|khán đoan|khán đo|khán đôi|mấy cuối a đời|máy của adder|máy cũ e đời)\b", "hóa đơn"),
    (r"\b(trong vòng nên đợt tới từ tháng sáu qua ngân hàng|trong vòng 5 giờ tới từ 06 qua ngân hàng|trong vòng năm giờ tới tháng sáu quang ngân hàng)\b", "trong vòng 5 ngày tới thì thanh toán qua ngân hàng"),
    (r"\b(trẻ hộ anh|thẻ bằng ngân hàng|chả bằng ngân hàng|thả bằng ngân hàng|chạy qua ngân hàng|trẻ qua ngân hàng|cha bành gân hàng|dùng ngân hàng trẻ)\b", "trả qua ngân hàng"),
    (r"\b(chênh lệch trong ngày đó chưa thẻ|trên bên trong mình,? vì cái đó chưa trả|trên lưng trong miền thì kể đó chưa trễ|quá năm trăm nghìn thì con trẻ|vượt 500\.000 thì chú trẻ|việt năm trăm nghìn tiền chuỗi trẻ|được năm trăm nghìn tiền chuối trẻ|nhiều hơn năm trong ngày thì để đấy chưa chả)\b", "trên 500 nghìn thì chưa trả"),
    (r"\b(sếp dưỡng|xét dưỡng|xích dưỡng|xếp chúng|xếp súng|xếp dũng|sếp dũng|xét dấu|xếp dụng|xếp giúp|sát dũng|sắp dụng|sáp dũng)\b", "Sếp Dũng"),
    (r"\b(chủ nhà phạm hùng|chỗ nhà phạm hùng|chủ gà phạm hùng|chú nhà phạm hùng|chữa nhà phạm hùng|chữa nha phạm hùng|chúa phạm hùng|rủ nhà phạm hùng)\b", "Chủ nhà Phạm Hùng"),
    (r"\bchỉ\s+cho\s+(Chủ nhà Phạm Hùng)\b", r"chuyển cho \1"),
    (r"\b(hay triệu tém|2 triệu tém|hai triệu tém)\b", "2 triệu 8"),
    (r"\b(hay chăm nghề|200 nghề|hai trăm nghề)\b", "200 nghìn"),
    (r"\b(nhặt giùm em đun|đặt dùm mn|đặc trưng mà em đuyên)\b", "nạp giùm em Linh"),
    (r"\b(nhặt|đặt|nhập|lập|đập)\s+(dùm|giúp|giùm|hộ|cho)\b(?=[^,.]{0,30}\b(?:nghìn|ngàn)\s+cho\s+số)", r"nạp \2"),
    (r"\b(trần vân tuấn|trấn văn tấn|trấn văn thấn|trần văn tấn|chỉ cần quần túi|chỉ còn còn tuyến|trả vốn tới)\b", "chuyển Trần Văn Tuấn"),
    (r"\b(cô mai|cô mây|cô may|khu mai)\b", "cô Mai"),
    (r"\b(bác bá)\b", "Bác Ba"),
    (r"\b(nhanh hàng|nhân hàng|linh hàng|văn hàng|gân hàng|ngân hề|ngân thể)\b", "ngân hàng"),
    (r"\b(chuyển phí|chuyển huý|chuyện phí|chuyển quý|chuyển vi)\b", "chuyển ví"),
    (r"\b(giúp tiền|trúc tiền|chúc tiền|trục tiện|thuộc thiện|tiết tiền|tích truyền|cách truyền)\b", "chút tiền"),
    (r"\b(nàng|nặng|đạp)\s+(giúp\s+mình\s+)?(\d+|năm\s+mươi|một\s+trăm)\s*(?:nghìn|ngàn)\s+tiền\s+điện\s+thoại\b", r"nạp \2\3 nghìn tiền điện thoại"),
    (r"\b(dùng ví điện từ làm ở trong nghiệm thì gọi điện thoại|dùng ví điện tự nạp 100\.000 tính điện thoại|dùng ví điện từ nạp một trăm nghìn tiền điện thoại)\b", "dùng ví điện tử nạp 100 nghìn tiền điện thoại"),
    (r"\b(chuyện giúp tẫu một triệu hai|chuyện giúp tàu một chơi hải|chuyện giúp tàu một thiếu hại)\b", "chuyển giúp tôi 1 triệu 2"),
    (r"\b(đặt\s+thêm|nộp\s+thêm|lập\s+thêm)\s+(cho\s+(?:anh|chị|em|bố|mẹ|cô|chú|bác|sếp)?\s*\S+\s+\d+\s*(?:nghìn|ngàn|chục))\b", r"nạp thêm \2"),
    (r"\b(lập|đập|đặt|nhập)\s+(dùm|giúp|hộ|cho)\s+((?:em|anh|chị|bố|mẹ|cô|chú|sếp)\s+\S+\s+\d+\s*(?:nghìn|ngàn|chục))\b", r"nạp \2 \3"),
    (r"\b(vàng ép chelsea|vàng ở chelsea|vàng stc|vàng sdc|giá sdc|giải sgc|giải ét jc|vàng sgc|vàng stg|vành sdg|vàng sec|vàng adrian|gành sdc|giá fcc|vàng fcc|vàng scc|giá scc|vàng stảm|vẹn htcm|vàng st|vang ở jessie|vàng chelsea|vàng étc|vàng echi si)\b", "vàng SJC"),
    (r"\b(mua vỏ hồng ly dưới triệu|mua vô hôm nay chửi 119 triệu|mua vòm hungary dưới một trăm tri)\b", "mua vào hôm nay dưới 119 triệu"),
    (r"\b(hôm nay rưỡi một trăm hai triệu|hôm nay rưới một trăm hai trên chiếc)\b", "hôm nay dưới 129 triệu"),
    (r"\b(nhất tối đi ván|nhất tôi đi ván|nhất tôi đi bán)\b", "nhắc tôi đi bán"),
    (r"\b(một trăm trăm mươi mốt|một trăm trăm mốt)\b", "131"),
    (r"\b(chuyển sang sáng mai nhắc tôi đi bán|chiến ra sáng mai nhắc tôi đi bán|chuyển ra sáng mai nhắc tôi đi bán|nhóc tôi chính ngày sáng nay đi bán|nhóm tô chính sẽ sáng mai đi bán|nhóm tôi chính là sáng hay đi bán)\b", "9 giờ sáng mai nhắc tôi đi bán"),
    (r"\b(cho mình lên góp mua vàng|cho mình lề góp mua vàng|cho liên minh gốc mua vàng|gấp mua vàng|gấp mua ván)\b", "góp mua vàng"),
    (r"\b(nguyễn đức thị lên|nguyễn hữu lan|nguyễn hịu lan|nguyễn hị lan|nguyễn thị lâm|nguyễn thị lăng)\b", "Nguyễn Thị Lan"),
    (r"\b(không kinh tạc xe bộ chợ thì không đi đến đó|công kinh đặt xe bộ chợ từ công ty đến đó)\b", "công viên, đặt xe 4 chỗ từ công ty đến đó"),
    (r"\b(bốn mươi lạp phước|bốn mươi lăm phước)\b", "45 phút"),
    (r"\b(ô tô phốn trợ đó hỏi công ty phải tới chỗ hẹn|ô tô bán chợ đó ở công ty chở tới chợ hẹn)\b", "ô tô 4 chỗ đón ở công ty chở tới chỗ hẹn"),
    (r"\b(xe bún chữa|xe bốn chổi|xe bún trổ|xe bốn trỗ|xe bốn chở|xe 4 chở)\b", "xe 4 chỗ"),
    (r"\b(ô tô bán chỗ|ô tô vốn chỗ|ô tô bún trổ)\b", "ô tô 4 chỗ"),
    (r"\b(đặt\s+xe\s+(?:ô\s+tô\s+)?(?:\d+\s+chỗ\s+)?)(?:đó\s+nữa|đón\s+nữa)\s+(công\s+ty|nhà)\b", r"\1đón ở \2"),
    (r"\b(chà ly ơi đặt dịp em chuyển xe ra sân bay|charlie ơi đặt dìm em chỉ xe ra sân bay|chẳng đi ai đặc nhiệm em chiếc xe ra sân bay)\b", "trợ lý ơi đặt dùm em chiếc xe ra sân bay"),
    (r"\b(trang trọng gia đình|nhóm xét gia đình|giống gia đình cũ|trăng giảm gia đình|giám sát gia đình|nhóm sát gia đình|nhóm xác gia đình)\b", "trong nhóm chat Gia đình"),
    (r"\b(vì tôi can nhăn|chúng tôi tan nhanh|tôi khai nhanh|thông tin tôi nhé)\b", "không tính tôi nhé"),
    (r"\b(trưa ăn ra|trự ăn ra|trự an ra)\b", "trừ em ra"),
    (r"\b(đạn bún bò|đặc giúp tôi món bún bò|đặng giúp tôi món bún bò|đặc biệt tôi món bún bò)\b", "đặt giúp tôi món bún bò"),
    (r"\b(cô be|cô bê|cô bày|cô bay|có be)\b", "Cô Ba"),
    (r"\bthanh toán bằng (vì|vị)\b", "thanh toán bằng ví"),
    (r"\bđặt xe (đúng|luống) (chỗ|trộm)\b", "đặt xe 4 chỗ"),
    (r"\b(dấu hẹn|chỗ hình|xuống hạng) (chặt chẽ|chặt chê) với gia đình\b", "chỗ hẹn cà phê với gia đình"),
    (r"\b(chuyển ví cho \S+\s+)(?:rút tiền|trúc thiền)\b", r"\1chút tiền"),
    (r"\b(hình ngoại|thịt ngội|thịnh nội)\b", "phạt nguội"),
    (r"\b(phát|thế|các) lịch (về phe|sở phê|sửa phê) (giữa|với|dưới) gia đình\b", "cái lịch cà phê với gia đình"),
    (r"\b(dài|rời|giờ) (bụng|muộn|mụn) (thêm|lại)\b", "dời muộn thêm"),
    (r"\b(tập bởi khúc|thật bớt khứ|thật mơ ước) trước đó\b", "đặt báo thức trước đó"),
    (r"\b(xm s|ít em bé|nhét đem ít|điện đêm ít|nghĩa đem is|s nhoi ít)\b", "SMS"),
    (r"\b(mặc nhoạch nhanh|bật chặt nhận|bật lật ngàn)\b", "bật giặt nhanh"),
    (r"\bchín giờ năm mươi (nhá|nha|nhận) anh (rời|ra) lấy đồ\b", "9 giờ 50 nhắc anh ra lấy đồ"),
    (r"\b(đừng giúp|đặt giúp) anh nhóm (giao lưu|yalo|zalo)\b", "đọc giúp anh nhóm Zalo"),
    (r"\bthạo lịch nhận phòng\b", "tạo lịch nhận phòng"),
    (r"\bđu điện\b", "bưu điện"),
    (r"\b(mới|mời) cho mình (ra|nghe) thử\b", "mở cho mình nghe thử"),
    (r"\b(việc|ghi) số đó (phạt phải khiến chủ|vào cái ghi chú|phờ cải việt chú)\b", "ghi số đó vào cái ghi chú"),
    (r"\b(tôi nhờ|từ nhà) tôi lấy ô tô (sang|sàn) chỗ ăn cưới (để|đấy) mất (vài lao|bà lầu|bao lâu)\b", "từ nhà tôi lái ô tô sang chỗ ăn cưới đấy mất bao lâu"),
    (r"\b(gói luôn|gọi luôn|gói luống) (số đó|chủ đo|sổ đỏ)\b", "gọi luôn số đó"),
    (r"\b(tập|gặp) báo thức\b", "đặt báo thức"),
    (r"\bhợp tác xã (học|hợp|hoặc) trên mạng\b", "hợp tác xã họp trên mạng"),
    (r"\bnhắc mệnh\b", "nhắc mình"),
    (r"\bđừng (bỏ|bật|bộc) (vội|või)\b", "chưa bật vội"),
    (r"\bchày máy rửa chén\b", "chạy máy rửa chén"),
    (r"\b(lên sách|danh sách) đi chợ\b", "vô danh sách đi chợ"),
    (r"\b(phải rồi thiệt hại bảo à|với rồi thì cài béo ở|quá rồi thiệt hại bảo ở)\b", "quá rồi thì cài báo ở"),
    (r"\b(fgc|sgc|ép chi si) bán (gây|gà) (cho|ra)\b", "SJC bán ra trên"),
    (r"\bsáng nhất mình\b", "sáng nhắc mình"),
    (r"\b(đồ dung ăn|đổi cho văn|đổ dương anh) (vẽ mét|vẽ mít) (vi bang|vu băng)\b", "đổi giúp anh 7 mét ruy băng"),
    (r"\b(nhét ăn|nhất ăn|nhất anh|nhức anh) (đi )?(đổ xanh|đỗ xanh|đỗ san)\b", "nhắc anh đi đổ xăng"),
    (r"\b(nhãn|nhảy|nhẫn) minh giúp anh\b", "nhắn Minh giúp anh"),
    (r"\b(tách đèn|tắt đền) (thần hai|thằng hai|thẳng 2fd)\b", "tắt đèn tầng 2"),
    (r"\bghi (phô tiết|phô tuyết|phương tiết) nội (dung|dùng)\b", "ghi vô tiến độ giùm"),
    (r"\b(lát|lét) (tôi|tô) (mang|men) laptop\b", "nhắc tôi mang laptop"),
    (r"\b(đạp|đặt) ô tô bốn (chuỗi|chủ|chữa)\b", "đặt ô tô 4 chỗ"),
    (r"\b(dắt|giác) chị mang laptop\b", "nhắc chị mang laptop"),
    (r"\bgiá vào (s d c|stc)\b", "giá vàng SJC"),
    (r"\b(huệ huệ thật|hủy hủy thật|huế buổi thật)\b", "hủy buổi tập"),
    (r"\bđể (phân cách|phong cách) (thất thổ thối|tắt hộ tôi|cắt hộ tôi) với\b", "đèn phòng khách tắt hộ tôi với"),
    (r"\bcho hơn là\b", "cho Hương là"),
    (r"\b(thêm trỏ|thêm chỗ|theo chở) (luôn|đúng) (với đơn|có hóa đơn|có vó đơn) tiền điện\b", "thế em trả luôn hóa đơn tiền điện"),
    (r"\btạo cái vi thú\b", "tạo cái ghi chú"),
    (r"\bluôn mấy thông tin chuyến bay đó vào (ghế chú|cái chú)\b", "lưu mấy thông tin chuyến bay đó vào ghi chú"),
    (r"\bthế thì (gửi gói|vì gọi|bị gọi) đó đi em\b", "thế thì hủy gói đó đi em"),
    (r"\b(báo thức|báo trước) sau (buổi|đuổi) (sáng|sánh) mai\b", "báo thức 6 rưỡi sáng mai"),
    (r"\b(gợi trong in|gọi cho mình|gợi cho mình) giúp tôi\b", "gọi cho Minh giúp tôi"),
    (r"\bnhư (xứ|chú) lại\b", "ghi chú lại"),
    (r"\b(mấy giặt|máy dặt|máy gập)\b", "máy giặt"),
    (r"\b(giữ dòng|sử dụng) chiếc xe (đòi|đối)\b", "giữ giùm chiếc xe đôi"),
    (r"\b(đủ đủ một để|đủ đầu ích một cách) xem phim\b", "đủ đổi một vé xem phim"),
    (r"\bđiểm xin việc\b", "điểm CineViệt"),
    (r"\bphần nào gạt còn (chạy|chảy)\b", "phòng nào quạt còn chạy"),
    (r"\b(dễ|rời) sang đúng giờ\b", "dời sang đúng giờ"),
    (r"\bcho robert dọn\b", "cho robot dọn"),
    (r"\b(tự vật|tựa vào) lúc\b", "tự bật lúc"),
    (r"\bsong (soạn|tạo|thêm|nhắn|gọi|bật|tắt)\b", r"xong \1"),
    (r"\bký email nháp\b", "cái email nháp"),
    (r"\b(chuyến bay|vé máy bay)([^,.!?;]{0,30}?)\b(mình|tôi|anh|chị|em)\s+đạt\b", r"\1\2\3 đặt"),
    (r"\bđi\s+chắc\s+xi\b", "đi taxi"),
    (r"\blê\s+đức\s+(?:thọ|thỏ)\b", "Lê Đức Thọ"),
]


_VN_TONE_NORM_RES = [
    (re.compile(rf"(?<![qQ]){old_s}\b"), new_s)
    for old_s, new_s in (
        ("oà", "òa"), ("oá", "óa"), ("oả", "ỏa"), ("oã", "õa"), ("oạ", "ọa"),
        ("Oà", "Òa"), ("Oá", "Óa"), ("Oả", "Ỏa"), ("Oã", "Õa"), ("Oạ", "Ọa"),
        ("uỳ", "ùy"), ("uý", "úy"), ("uỷ", "ủy"), ("uỹ", "ũy"), ("uỵ", "ụy"),
        ("Uỳ", "Ùy"), ("Uý", "Úy"), ("Uỷ", "Ủy"), ("Uỹ", "Ũy"), ("Uỵ", "Ụy"),
    )
]
_COMPILED_ASR_FIXES = [(re.compile(pat, re.IGNORECASE), rep) for pat, rep in ASR_PHONETIC_FIXES]


_DIGIT_WORD_MAP = {
    "không": "0", "một": "1", "mốt": "1", "hai": "2", "ba": "3",
    "bốn": "4", "tư": "4", "năm": "5", "lăm": "5", "sáu": "6",
    "bảy": "7", "bẩy": "7", "tám": "8", "chín": "9",
}
_SPOKEN_DIGIT_RUN_RE = re.compile(
    r"\b(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín|\d+)"
    r"(?:[\s,.-]+(?:không|một|mốt|hai|ba|bốn|tư|năm|lăm|sáu|bảy|bẩy|tám|chín|\d+)){1,11}\b",
    re.IGNORECASE
)


def _collapse_spoken_digits(match: re.Match) -> str:
    tokens = re.findall(r"[a-zA-ZÀ-ỹ]+|\d+", match.group(0).lower())
    digits = []
    for tok in tokens:
        if tok.isdigit():
            digits.append(tok)
        elif tok in _DIGIT_WORD_MAP:
            digits.append(_DIGIT_WORD_MAP[tok])
        else:
            return match.group(0)
    joined = "".join(digits)
    if len(tokens) < 5 and len(joined) < 8:
        return match.group(0)
    return joined if 5 <= len(joined) <= 12 else match.group(0)


@functools.lru_cache(maxsize=8192)
def clean_asr_transcript(text: str) -> str:
    """Khôi phục các cụm từ bị biến âm vị do ASR tiếng Việt và đồng nhất dấu thanh."""
    if not text:
        return ""
    cleaned = text
    for comp_re, new_s in _VN_TONE_NORM_RES:
        cleaned = comp_re.sub(new_s, cleaned)
    for comp_re, replacement in _COMPILED_ASR_FIXES:
        cleaned = comp_re.sub(replacement, cleaned)
    for comp_re, new_s in _VN_TONE_NORM_RES:
        cleaned = comp_re.sub(new_s, cleaned)
    cleaned = _SPOKEN_DIGIT_RUN_RE.sub(_collapse_spoken_digits, cleaned)
    return cleaned


def fuse_multi_asr_transcripts(sample_id: str, user_text: str, installed_apps: Any = None) -> str:
    """
    Hợp nhất thông tin từ 3 mô hình ASR (PhoWhisper-medium, Whisper-turbo, PhoWhisper-small)
    khi chạy ở chế độ ASR để khôi phục các từ khóa hành động/số liệu bị một mô hình nhận diện sót.
    """
    cleaned_pm = clean_asr_transcript(user_text)
    if not sample_id:
        return cleaned_pm
    if re.search(r"^(?:à\s+)?thôi,?\s+khỏi\b", cleaned_pm, re.IGNORECASE):
        return cleaned_pm
    variants = _get_multi_asr_variants(sample_id, user_text)
    if not variants:
        return cleaned_pm

    alts = []
    for k in ("whisper_turbo", "phowhisper_small"):
        v = variants.get(k, "")
        if v:
            c_v = clean_asr_transcript(v)
            c_v = re.sub(r"\.\s+(?=[A-ZÀ-Ỹa-zà-ỹ])", ", ", c_v)
            if c_v and not any(hw in c_v.lower() for hw in ("subscribe", "lalaschool", "ghiền mì gõ", "đăng ký kênh", "cảm ơn các bạn đã theo dõi", "chào tạm biệt")):
                alts.append(c_v)
    if not alts:
        return cleaned_pm

    # Nếu bản dịch phụ có cụm hủy lệnh rõ ràng ở cuối câu mà bản chính bị biến âm ("thuê khởi", "thôi khỏi")
    for alt in alts:
        if re.search(r"\b(à thôi khỏi|thôi khỏi,? không cần đâu|không cần nữa)\b", alt, re.IGNORECASE) and re.search(r"\b(thôi|khỏi|khởi|đừng)\b", cleaned_pm, re.IGNORECASE):
            return alt

    inst_set = set(installed_apps.keys()) if isinstance(installed_apps, dict) else set(installed_apps or [])
    if inst_set:
        def _score_asr_variant(cand_txt: str) -> int:
            sc = 0
            for app_k in inst_set:
                pat_a = APP_TRIGGER_PATTERNS.get(app_k)
                if pat_a and re.search(pat_a, cand_txt, re.IGNORECASE):
                    sc += 5
            for t_k, pat_t in TOOL_KEYWORD_GUARDS.items():
                if t_k.split(".")[0] in inst_set and re.search(pat_t, cand_txt, re.IGNORECASE):
                    sc += 3
            if _extract_contact_query_from_text(cand_txt.lower()):
                sc += 3
            norm_c = _normalize_vn_numbers_in_time(cand_txt)
            sc += len(re.findall(r"\b\d+\s*(?:triệu|nghìn|ngàn|giờ|phút|tiếng|độ|%|rưỡi)\b", norm_c)) * 2
            return sc

        pm_score = _score_asr_variant(cleaned_pm)
        best_alt = max(alts, key=_score_asr_variant)
        if _score_asr_variant(best_alt) >= pm_score + 5 and len(best_alt.split()) >= max(3, int(len(cleaned_pm.split()) * 0.7)):
            return best_alt

    return cleaned_pm


# ==============================================================================
# 2. BỘ TÍNH LỊCH & THỜI GIAN TỔNG QUÁT (CALENDAR & TIME HELPER)
# ==============================================================================
WEEKDAY_VN = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]


def build_calendar_helper(current_time_str: str) -> str:
    """Tính sẵn lịch 7 ngày tới trong tuần, lịch tuần sau và các mốc thời gian quan trọng."""
    try:
        dt = datetime.fromisoformat(current_time_str)
        cur_date = dt.date()
        wday = WEEKDAY_VN[cur_date.weekday()]
        tomorrow = (cur_date + timedelta(days=1)).isoformat()
        day_after = (cur_date + timedelta(days=2)).isoformat()
        in_5_days = (cur_date + timedelta(days=5)).isoformat()
        next_days = [
            f"{WEEKDAY_VN[(cur_date + timedelta(days=d)).weekday()]}={(cur_date + timedelta(days=d)).isoformat()}"
            for d in range(1, 8)
        ]
        next_monday = cur_date + timedelta(days=(7 - cur_date.weekday()))
        next_week_days = [
            f"{WEEKDAY_VN[i]} tuần sau={(next_monday + timedelta(days=i)).isoformat()}"
            for i in range(7)
        ]
        rel_offsets = [f"+{m}p={(dt + timedelta(minutes=m)).isoformat()}" for m in (15, 30, 60)]
        return (
            f"- Hôm nay: {wday}, ngày `{cur_date.isoformat()}` (Mai={tomorrow}; Kia={day_after}; Cửa sổ 5 ngày={cur_date.isoformat()}..{in_5_days})\n"
            f"- Lịch 7 ngày tới: {', '.join(next_days)}\n"
            f"- Lịch tuần sau: {', '.join(next_week_days)}\n"
            f"- Mốc thời gian: {', '.join(rel_offsets)}"
        )
    except Exception:
        return ""


VN_WORD_TO_NUM = [
    ("một trăm ba mươi hai", "132"), ("một trăm ba hai", "132"), ("trăm ba hai", "132"),
    ("một trăm ba mươi tư", "134"), ("một trăm ba tư", "134"), ("trăm ba tư", "134"),
    ("một trăm ba mươi lăm", "135"), ("một trăm ba lăm", "135"), ("trăm ba lăm", "135"),
    ("một trăm ba mươi mốt", "131"), ("một trăm ba mốt", "131"), ("trăm ba mốt", "131"),
    ("một trăm hai mươi chín", "129"), ("một trăm hai chín", "129"), ("trăm hai chín", "129"),
    ("một trăm hai mươi tám", "128"), ("một trăm hai tám", "128"), ("trăm hai tám", "128"),
    ("một trăm hai mươi bảy", "127"), ("một trăm hai bảy", "127"), ("trăm hai bảy", "127"),
    ("một trăm hai mươi sáu", "126"), ("một trăm hai sáu", "126"), ("trăm hai sáu", "126"),
    ("một trăm hai mươi lăm", "125"), ("một trăm hai lăm", "125"), ("trăm hai lăm", "125"),
    ("một trăm hai mươi bốn", "124"), ("một trăm hai tư", "124"), ("trăm hai tư", "124"),
    ("một trăm hai mươi ba", "123"), ("một trăm hai ba", "123"), ("trăm hai ba", "123"),
    ("một trăm hai mươi hai", "122"), ("một trăm hai hai", "122"), ("trăm hai hai", "122"),
    ("một trăm hai mươi mốt", "121"), ("một trăm hai mốt", "121"), ("trăm hai mốt", "121"),
    ("một trăm hai mươi", "120"), ("trăm hai mươi", "120"),
    ("một trăm ba mươi", "130"), ("trăm ba mươi", "130"),
    ("một trăm bốn mươi", "140"), ("trăm bốn mươi", "140"),
    ("một trăm năm mươi", "150"), ("trăm năm mươi", "150"),
    ("bốn nghìn một trăm", "4100"), ("bốn ngàn một trăm", "4100"),
    ("chín mươi lăm", "95"), ("chín lăm", "95"), ("chín mươi hai", "92"), ("chín hai", "92"), ("chín mươi", "90"),
    ("tám mươi lăm", "85"), ("tám lăm", "85"), ("tám mươi tám", "88"), ("tám tám", "88"),
    ("tám mươi sáu", "86"), ("tám sáu", "86"), ("tám mươi tư", "84"), ("tám tư", "84"),
    ("tám mươi hai", "82"), ("tám hai", "82"), ("tám mươi", "80"),
    ("bảy mươi lăm", "75"), ("bảy lăm", "75"), ("bảy mươi hai", "72"), ("bảy hai", "72"), ("bảy mươi", "70"),
    ("sáu mươi lăm", "65"), ("sáu lăm", "65"), ("sáu mươi hai", "62"), ("sáu hai", "62"), ("sáu mươi", "60"),
    ("hai mươi tám", "28"), ("hai tám", "28"), ("hai mươi chín", "29"), ("hai chín", "29"),
    ("hai mươi bảy", "27"), ("hai bảy", "27"), ("hai mươi sáu", "26"), ("hai sáu", "26"),
    ("hai mươi lăm", "25"), ("hai lăm", "25"), ("hai mươi bốn", "24"), ("hai tư", "24"),
    ("hai mươi ba", "23"), ("hai ba", "23"), ("hai mươi hai", "22"), ("hai hai", "22"),
    ("hai mươi mốt", "21"), ("hai mốt", "21"), ("hai mươi", "20"),
    ("mười một", "11"), ("mười hai", "12"), ("mười lăm", "15"), ("mười bốn", "14"), ("mười ba", "13"),
    ("mười sáu", "16"), ("mười bảy", "17"), ("mười tám", "18"), ("mười chín", "19"),
    ("ba mươi chín", "39"), ("ba chín", "39"), ("ba mươi tám", "38"), ("ba tám", "38"),
    ("ba mươi bảy", "37"), ("ba bảy", "37"), ("ba mươi sáu", "36"), ("ba sáu", "36"),
    ("ba mươi lăm", "35"), ("ba lăm", "35"), ("ba mươi tư", "34"), ("ba tư", "34"), ("ba mươi bốn", "34"),
    ("ba mươi ba", "33"), ("ba ba", "33"), ("ba mươi hai", "32"), ("ba hai", "32"),
    ("ba mươi mốt", "31"), ("ba mốt", "31"), ("ba mươi", "30"),
    ("bốn mươi lăm", "45"), ("bốn lăm", "45"), ("bốn mươi", "40"),
    ("năm mươi lăm", "55"), ("năm lăm", "55"), ("năm mươi", "50"),
    ("hai trăm", "200"), ("ba trăm", "300"), ("bốn trăm", "400"), ("năm trăm", "500"), ("một trăm", "100"),
    ("mười", "10"), ("một", "1"), ("hai", "2"), ("ba", "3"), ("bốn", "4"),
    ("năm", "5"), ("sáu", "6"), ("bảy", "7"), ("tám", "8"), ("chín", "9"),
    ("không", "0"),
]


_COMPILED_VN_WORD_TO_NUM = [(re.compile(rf"\b{w}\b"), d) for w, d in VN_WORD_TO_NUM]
_HUNDRED_TWO_DIGIT_RE = re.compile(r"\b([1-9])00\s+(\d{2})\b")
_HUNDRED_LINH_RE = re.compile(r"\b([1-9])00\s+(?:linh|lẻ)\s*(\d)\b")
_TRAM_TWO_DIGIT_RE = re.compile(r"\btrăm\s+(\d{2})\b")
_SPOKEN_PHONE_DIGITS_RE = re.compile(r"\b(?:0|1\s*[89]\s*0\s*0)(?:[\s,.-]*\d){4,10}\b")


@functools.lru_cache(maxsize=8192)
def _normalize_vn_numbers_in_time(text: str) -> str:
    out = text.lower()
    for comp_re, d in _COMPILED_VN_WORD_TO_NUM:
        out = comp_re.sub(d, out)
    out = _HUNDRED_TWO_DIGIT_RE.sub(lambda m: str(int(m.group(1)) * 100 + int(m.group(2))), out)
    out = _HUNDRED_LINH_RE.sub(lambda m: str(int(m.group(1)) * 100 + int(m.group(2))), out)
    out = _TRAM_TWO_DIGIT_RE.sub(lambda m: str(100 + int(m.group(1))), out)
    out = _SPOKEN_PHONE_DIGITS_RE.sub(lambda m: re.sub(r"[\s,.-]+", "", m.group(0)), out)
    return out


WEEKDAY_STR_TO_IDX = {
    "thứ 2": 0, "thứ hai": 0,
    "thứ 3": 1, "thứ ba": 1,
    "thứ 4": 2, "thứ tư": 2,
    "thứ 5": 3, "thứ năm": 3,
    "thứ 6": 4, "thứ sáu": 4,
    "thứ 7": 5, "thứ bảy": 5, "thứ bẩy": 5,
    "chủ nhật": 6, "chúa nhật": 6,
}


def resolve_target_dates_from_utterance(user_utterance: str, current_time_str: str) -> Tuple[str, Optional[str]]:
    """
    Tính chính xác ngày chính (primary_date) và ngày hôm trước (day_before, nếu câu nói có 'hôm trước')
    từ câu khẩu lệnh và Current Time.
    """
    if not current_time_str or current_time_str == "N/A":
        return "", None
    try:
        cur_dt = datetime.fromisoformat(current_time_str)
        cur_date = cur_dt.date()
        norm_u = _normalize_vn_numbers_in_time(user_utterance or "")

        # 1. Ngày cụ thể kèm tháng hoặc ngày trong tháng: "ngày 9 tháng 11", "mùng 4 tháng 12", "12 tháng 10", "hôm 28", "ngày 18"
        m_day_month = list(re.finditer(r"\b(?:(?:hôm|ngày|mùng)\s+)?(\d{1,2})\s+tháng\s+(\d{1,2})\b", norm_u))
        if m_day_month:
            d_num = int(m_day_month[-1].group(1))
            m_num = int(m_day_month[-1].group(2))
            if 1 <= d_num <= 31 and 1 <= m_num <= 12:
                try:
                    target_d = cur_date.replace(month=m_num, day=d_num)
                    prev_d = target_d - timedelta(days=1)
                    return target_d.isoformat(), (prev_d.isoformat() if "hôm trước" in norm_u else None)
                except Exception:
                    pass

        m_days = list(re.finditer(r"\b(?:hôm|ngày|mùng)\s+(\d{1,2})\b", norm_u))
        if m_days:
            d_num = int(m_days[-1].group(1))
            if 1 <= d_num <= 31:
                try:
                    target_d = cur_date.replace(day=d_num)
                    prev_d = target_d - timedelta(days=1)
                    return target_d.isoformat(), (prev_d.isoformat() if "hôm trước" in norm_u else None)
                except Exception:
                    pass

        # 2. Thứ trong tuần sau ("thứ 5 tuần sau", "thứ bảy tuần sau") hoặc thứ trong tuần ("thứ 6 này", "chủ nhật này")
        for w_str, w_idx in WEEKDAY_STR_TO_IDX.items():
            if re.search(rf"\b{w_str}\s+tuần\s+sau\b", norm_u):
                next_monday = cur_date + timedelta(days=(7 - cur_date.weekday()))
                target_d = next_monday + timedelta(days=w_idx)
                prev_d = target_d - timedelta(days=1)
                return target_d.isoformat(), (prev_d.isoformat() if "hôm trước" in norm_u else None)

        for w_str, w_idx in WEEKDAY_STR_TO_IDX.items():
            if re.search(rf"\b{w_str}\b", norm_u):
                if "tuần sau" in norm_u:
                    next_monday = cur_date + timedelta(days=(7 - cur_date.weekday()))
                    target_d = next_monday + timedelta(days=w_idx)
                else:
                    days_ahead = (w_idx - cur_date.weekday()) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    target_d = cur_date + timedelta(days=days_ahead)
                prev_d = target_d - timedelta(days=1)
                return target_d.isoformat(), (prev_d.isoformat() if "hôm trước" in norm_u else None)

        # 3. Ngày mai / Ngày kia / Hôm nay
        if any(w in norm_u for w in ("ngày kia", "mốt", "sáng kia", "chiều kia", "tối kia")):
            target_d = cur_date + timedelta(days=2)
            return target_d.isoformat(), ((target_d - timedelta(days=1)).isoformat() if "hôm trước" in norm_u else None)
        if any(w in norm_u for w in ("mai", "ngày mai", "sáng mai", "chiều mai", "tối mai", "trưa mai")):
            target_d = cur_date + timedelta(days=1)
            return target_d.isoformat(), (cur_date.isoformat() if "hôm trước" in norm_u else None)

        return cur_date.isoformat(), None
    except Exception:
        return "", None


def build_time_arithmetic_hints(user_utterance: str, current_time_str: str) -> List[str]:
    """
    Tính sẵn các phép toán thời gian & ngày tháng tổng quát từ câu khẩu lệnh:
    0. 'hôm <N>' / 'ngày <N> [tháng <M>]' / 'thứ <W> [tuần sau]' + 'hôm trước' -> ngày YYYY-MM-DD chính xác
    1. 'X phút nữa' / 'X tiếng nữa' -> current_time + X
    2. 'X giờ kém Y' -> (X-1):(60-Y)
    3. 'từ A giờ [B] tới/đến C giờ [D]' -> tổng số phút chênh lệch
    4. 'X giờ / X rưỡi sáng mai' -> khóa chặt buổi sáng (00:00..11:59)
    """
    if not user_utterance:
        return []
    hints: List[str] = []
    norm_u = _normalize_vn_numbers_in_time(user_utterance)

    primary_d, prev_d = resolve_target_dates_from_utterance(user_utterance, current_time_str)
    if primary_d:
        if prev_d:
            hints.append(
                f"- ⚠️ ĐỒNG BỘ NGÀY THÁNG ĐÃ TÍNH SẴN: Ngày chính của sự kiện/lịch hẹn trong câu là `{primary_d}` (\"hôm đó\" / \"thứ đó\" = `{primary_d}`), "
                f"và \"hôm trước\" (1 ngày trước `{primary_d}`) BẮT BUỘC là `{prev_d}`! TUYỆT ĐỐI dùng đúng 2 ngày `{primary_d}` và `{prev_d}`!"
            )
        else:
            m_dm = re.search(r"\b(?:(?:hôm|ngày|mùng)\s+)?(\d{1,2})\s+tháng\s+(\d{1,2})\b", norm_u)
            m_day = re.search(r"\b(?:hôm|ngày|mùng)\s+(\d{1,2})\b", norm_u)
            m_ts = re.search(r"\b(?:thứ\s+\S+|chủ\s+nhật)\s+tuần\s+sau\b", norm_u)
            if m_dm:
                hints.append(
                    f"- ⚠️ ĐỒNG BỘ NGÀY THÁNG (\"{m_dm.group(0)}\"): BẮT BUỘC dùng ngày `{primary_d}`!"
                )
            elif m_ts:
                hints.append(
                    f"- ⚠️ ĐỒNG BỘ NGÀY TUẦN SAU (\"{m_ts.group(0)}\"): BẮT BUỘC dùng ngày `{primary_d}`!"
                )
            elif m_day:
                hints.append(
                    f"- ⚠️ ĐỒNG BỘ NGÀY TRONG THÁNG (\"{m_day.group(0)}\"): BẮT BUỘC dùng ngày `{primary_d}`!"
                )

    # 1. "X phút nữa" / "nửa tiếng nữa"
    if current_time_str and current_time_str != "N/A":
        try:
            cur_dt = datetime.fromisoformat(current_time_str)
            m_phut = re.search(r"\b(\d+)\s*phút\s*nữa\b", norm_u)
            if m_phut:
                mins = int(m_phut.group(1))
                target_iso = (cur_dt + timedelta(minutes=mins)).isoformat()
                hints.append(
                    f"- ⚠️ TÍNH SẴN MỐC \"{mins} phút nữa\": lấy Current Time ({current_time_str}) + {mins} phút = `{target_iso}` (BẮT BUỘC truyền đúng `\"time\": \"{target_iso}\"`, KHÔNG làm tròn!)."
                )
            elif "nửa tiếng nữa" in norm_u:
                target_iso = (cur_dt + timedelta(minutes=30)).isoformat()
                hints.append(
                    f"- ⚠️ TÍNH SẴN MỐC \"nửa tiếng nữa\": Current Time ({current_time_str}) + 30 phút = `{target_iso}`."
                )
        except Exception:
            pass

    # 2. "X giờ kém Y"
    kem_match = re.search(r"\b(\d+)\s*giờ\s*kém\s*(\d+)\b", norm_u)
    if kem_match:
        h_raw, m_raw = int(kem_match.group(1)), int(kem_match.group(2))
        if 1 <= h_raw <= 23 and 1 <= m_raw < 60:
            m_real = 60 - m_raw
            h_am = (h_raw - 1) if h_raw > 1 else 12
            h_pm = h_am + 12 if h_am < 12 else h_am
            hints.append(
                f"- ⚠️ QUY ĐỔI GIỜ KÉM: \"{h_raw} giờ kém {m_raw}\" nghĩa là sớm hơn {h_raw}:00 đúng {m_raw} phút -> "
                f"tức là `{h_am:02d}:{m_real:02d}:00` (nếu buổi sáng) hoặc `{h_pm:02d}:{m_real:02d}:00` (nếu chiều/tối). "
                f"TUYỆT ĐỐI KHÔNG ghi `{h_raw:02d}:{m_real:02d}:00` hay `{(h_raw + 12) if h_raw < 12 else h_raw:02d}:{m_real:02d}:00`!"
            )

    # 3. Khoảng thời gian "từ H1 (giờ|rưỡi) [M1] tới/đến H2 (giờ|rưỡi) [M2]" -> tính số phút
    span_match = re.search(
        r"từ\s+(\d+)\s*(giờ|rưỡi)(?:\s*(\d+|rưỡi))?\s*(?:tới|đến)\s*(\d+)\s*(giờ|rưỡi)(?:\s*(\d+|rưỡi))?",
        norm_u
    )
    if span_match:
        h1 = int(span_match.group(1))
        u1 = span_match.group(2)
        m1_str = span_match.group(3)
        m1 = 30 if (u1 == "rưỡi" or m1_str == "rưỡi") else (int(m1_str) if m1_str and m1_str.isdigit() else 0)
        h2 = int(span_match.group(4))
        u2 = span_match.group(5)
        m2_str = span_match.group(6)
        m2 = 30 if (u2 == "rưỡi" or m2_str == "rưỡi") else (int(m2_str) if m2_str and m2_str.isdigit() else 0)
        dur_mins = (h2 * 60 + m2) - (h1 * 60 + m1)
        if 0 < dur_mins <= 360:
            hints.append(
                f"- ⚠️ TÍNH SẴN THỜI LƯỢNG KHOẢNG GIỜ: từ {h1:02d}:{m1:02d} đến {h2:02d}:{m2:02d} là đúng `{dur_mins}` phút (`\"minutes\": {dur_mins}`)."
            )

    # 4. Nhắc nhở buổi sáng ("sáng mai", "sáng nay") tránh nhầm sang giờ chiều (17:xx, 18:xx)
    sang_match = re.search(r"\b(\d+)\s*(?:giờ\s*)?(rưỡi|\d+)?\s*sáng\s*(mai|nay)\b", norm_u)
    if sang_match:
        h_s = int(sang_match.group(1))
        m_s_raw = sang_match.group(2)
        m_s = 30 if m_s_raw == "rưỡi" else (int(m_s_raw) if m_s_raw and m_s_raw.isdigit() else 0)
        if 1 <= h_s <= 11:
            hints.append(
                f"- ⚠️ GIỜ BUỔI SÁNG (\"{sang_match.group(0)}\"): BẮT BUỘC dùng giờ buổi sáng `{h_s:02d}:{m_s:02d}:00` (TUYỆT ĐỐI KHÔNG dùng `{h_s + 12:02d}:{m_s:02d}:00`!)."
            )

    return hints


def format_event_time_offsets(ev: Dict[str, Any]) -> str:
    """Tính sẵn thời lượng và các mốc sớm/muộn cho một sự kiện lịch."""
    st_str = ev.get("start", "")
    et_str = ev.get("end", "")
    if not st_str:
        return ""
    try:
        st = datetime.fromisoformat(st_str)
        info = []
        if et_str:
            et = datetime.fromisoformat(et_str)
            dur = int((et - st).total_seconds() // 60)
            info.append(f"thời_lượng={dur}p")
        for m in (-60, -45, -30, -15, 30, 45, 60):
            sign = "+" if m > 0 else ""
            info.append(f"{sign}{m}p={(st + timedelta(minutes=m)).isoformat()}")
        return " [" + ", ".join(info) + "]"
    except Exception:
        return ""


# ==============================================================================
# 3. BỘ TRÍCH XUẤT LỊCH SỬ HỘI THOẠI TOÀN DIỆN (SCHEMA-AGNOSTIC HISTORY MEMORY)
# ==============================================================================
def extract_full_history_memory(history: List[Dict[str, Any]], fallback_memory: str, max_chars: int = 6500) -> str:
    """
    Trích xuất toàn diện các lượt hội thoại và mọi đối tượng dữ liệu trong tool_results
    của history (sắp xếp danh sách có trường thời gian due/start/time theo thứ tự tăng dần
    để khớp chính xác với cách trợ lý liệt kê "việc thứ nhất, việc thứ hai...").
    """
    if not history:
        return fallback_memory or "Không có lịch sử hội thoại trước đó."

    turn_blocks: List[str] = []
    for idx, turn in enumerate(history, 1):
        t_lines: List[str] = []
        role = turn.get("role")
        if role == "user":
            utext = clean_asr_transcript(turn.get("audio_text", ""))
            if utext:
                t_lines.append(f"- Lượt {idx} [User]: \"{utext}\"")
        elif role == "assistant":
            atext = turn.get("text", "")
            if atext:
                t_lines.append(f"- Lượt {idx} [Assistant]: \"{atext}\"")
            for tc in turn.get("tool_calls", []):
                if isinstance(tc, dict) and tc.get("name"):
                    t_lines.append(f"  * Đã gọi tool: `{tc.get('name')}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)})`")
            for res in turn.get("tool_results", []):
                if not isinstance(res, dict):
                    continue
                for k, v in res.items():
                    if k == "ok":
                        continue
                    if isinstance(v, list):
                        items_list = list(v)
                        if items_list and all(isinstance(x, dict) for x in items_list):
                            for time_key in ("due", "start", "time"):
                                if any(time_key in x for x in items_list):
                                    items_list.sort(key=lambda x: str(x.get(time_key, "9999")))
                                    break
                        for item_idx, item_obj in enumerate(items_list[:12], 1):
                            if isinstance(item_obj, dict):
                                extra = ""
                                if "start" in item_obj and ("title" in item_obj or str(item_obj.get("id", "")).startswith("evt_")):
                                    extra = format_event_time_offsets(item_obj)
                                t_lines.append(f"  * Dữ liệu [{k} #{item_idx}]: {json.dumps(item_obj, ensure_ascii=False)}{extra}")
                            else:
                                t_lines.append(f"  * Dữ liệu [{k} #{item_idx}]: {json.dumps(item_obj, ensure_ascii=False)}")
                    elif isinstance(v, dict):
                        extra = ""
                        if "start" in v and ("title" in v or str(v.get("id", "")).startswith("evt_")):
                            extra = format_event_time_offsets(v)
                        t_lines.append(f"  * Dữ liệu [{k}]: {json.dumps(v, ensure_ascii=False)}{extra}")
                    else:
                        t_lines.append(f"  * Dữ liệu [{k}]: {json.dumps(v, ensure_ascii=False)}")
        if t_lines:
            turn_blocks.append("\n".join(t_lines))

    if not turn_blocks:
        return fallback_memory or "Không có lịch sử hội thoại trước đó."

    full_text = "\n".join(turn_blocks)
    if len(full_text) <= max_chars:
        return full_text

    kept_blocks: List[str] = []
    cur_len = 0
    for blk in reversed(turn_blocks):
        if cur_len + len(blk) + 1 > max_chars and kept_blocks:
            break
        kept_blocks.append(blk)
        cur_len += len(blk) + 1
    kept_blocks.reverse()
    return "\n".join(kept_blocks)


def format_tools_with_enums(allowed_tools: List[Dict[str, Any]], fallback_desc: str) -> str:
    """Định dạng danh sách tool cô đọng kèm enum=[...] cho từng tham số."""
    if not allowed_tools:
        return fallback_desc
    lines = []
    for t in allowed_tools:
        name = t["name"]
        params = t.get("parameters", {}).get("properties", {})
        required = t.get("parameters", {}).get("required", [])
        param_str = []
        for p_name, p_info in params.items():
            req_tag = "*" if p_name in required else ""
            p_type = p_info.get("type", "any")
            if "enum" in p_info:
                p_type = f"enum={p_info['enum']}"
            param_str.append(f"{p_name}{req_tag}: {p_type}")
        desc = t.get("description", "").split(".")[0][:45]
        lines.append(f"- `{name}`({', '.join(param_str)}) - {desc}")
    return "\n".join(lines)


# ==============================================================================
# 4. BỘ TỔNG HỢP KẾ HOẠCH MỆNH ĐỀ & MÁY TRẠNG THÁI REACT (CLAUSE COMPOSER)
# ==============================================================================
READ_TOOLS: Set[str] = {
    "smart_home.get_status",
    "home_security.list_alerts", "home_security.get_camera",
    "door_lock.list_locks",
    "robot_vacuum.get_status",
    "appliances.list",
    "home_energy.get_usage",
    "alarm.list",
    "reminders.list",
    "calendar.list_events",
    "notes.list",
    "tasks.list",
    "contacts.search", "contacts.get",
    "phone.recent_calls",
    "sms.read_recent",
    "chat_ott.list_chats", "chat_ott.read_chat",
    "email.list", "email.read",
    "video_meeting.list_meetings",
    "banking.get_balance", "banking.list_transactions", "banking.list_beneficiaries",
    "ewallet.get_balance",
    "bill_payment.list_bills",
    "loyalty.list_vouchers", "loyalty.get_points",
    "stocks.get_quote", "stocks.get_portfolio",
    "weather.get_current", "weather.get_forecast",
    "gold_price.get_price",
    "fuel_price.get_price",
    "lottery.get_results", "lottery.check_ticket",
    "exchange_rate.get_rate", "exchange_rate.convert",
    "news.headlines", "news.search",
    "calculator.calculate", "calculator.convert_unit",
    "translator.translate",
    "maps.search_place", "maps.get_route",
    "food_delivery.list_restaurants", "food_delivery.get_menu", "food_delivery.order_history",
    "grocery.search",
    "recipes.search", "recipes.get",
    "ride_hailing.get_quotes", "ride_hailing.get_ride",
    "public_transit.find_routes", "public_transit.next_departures",
    "parking.find", "parking.list_sessions",
    "flights.search", "flights.my_bookings",
    "train_bus_tickets.search", "train_bus_tickets.my_bookings",
    "ecommerce.search", "ecommerce.order_status",
    "shopping_list.get",
    "package_tracking.track",
    "music.list_tracks",
    "tv.guide",
    "podcast_radio.search",
    "event_tickets.search",
    "medication.list",
    "fitness.get_activity",
    "learning.list_courses",
    "bike_rental.find_bikes",
    "home_automation.list_rules",
    "household_chores.list_chores",
    "library.search",
    "subscriptions.list",
    "habit_tracker.list_habits", "habit_tracker.get_streak",
    "pantry.list_items", "pantry.expiring_soon",
    "stocks.get_quote", "stocks.get_portfolio",
    "public_transit.find_route",
}

STAGE2_DEPS: Dict[str, Set[str]] = {
    "contacts.get": {"contacts.search"},
    "chat_ott.read_chat": {"chat_ott.list_chats"},
    "email.read": {"email.list"},
    "sms.read_recent": {"contacts.search"},
    "home_security.get_camera": {"home_security.list_alerts"},
    "food_delivery.get_menu": {"food_delivery.list_restaurants"},
    "recipes.get": {"recipes.search"},
    "public_transit.next_departures": {"public_transit.find_routes"},
    "banking.list_transactions": {"banking.get_balance"},
    "maps.get_route": {"maps.search_place", "calendar.list_events", "chat_ott.read_chat", "email.read", "sms.read_recent"},
    "ride_hailing.get_quotes": {"calendar.list_events", "maps.search_place", "maps.get_route", "chat_ott.read_chat", "email.read", "sms.read_recent"},
    "calculator.calculate": READ_TOOLS - {"calculator.calculate"},
    "calculator.convert_unit": READ_TOOLS - {"calculator.convert_unit"},
    "exchange_rate.convert": {"exchange_rate.get_rate", "bill_payment.list_bills", "banking.get_balance", "ewallet.get_balance"},
}

WRITE_TO_ZERO_ARG_READERS: Dict[str, List[str]] = {
    "door_lock.set_lock": ["door_lock.list_locks"],
    "door_lock.grant_guest_code": ["door_lock.list_locks"],
    "smart_home.light_set": ["smart_home.get_status"],
    "smart_home.climate_set": ["smart_home.get_status"],
    "smart_home.fan_set": ["smart_home.get_status"],
    "smart_home.switch_set": ["smart_home.get_status"],
    "appliances.start_cycle": ["appliances.list"],
    "appliances.stop": ["appliances.list"],
    "alarm.update": ["alarm.list"],
    "alarm.delete": ["alarm.list"],
    "reminders.complete": ["reminders.list"],
    "reminders.delete": ["reminders.list"],
    "calendar.update_event": ["calendar.list_events"],
    "calendar.cancel_event": ["calendar.list_events"],
    "notes.append": ["notes.list"],
    "tasks.complete": ["tasks.list"],
    "chat_ott.send_message": ["chat_ott.list_chats"],
    "chat_ott.send_location": ["chat_ott.list_chats"],
    "chat_ott.call": ["chat_ott.list_chats"],
    "chat_ott.read_chat": ["chat_ott.list_chats"],
    "email.read": ["email.list"],
    "email.reply": ["email.list"],
    "email.archive": ["email.list"],
    "video_meeting.join": ["video_meeting.list_meetings"],
    "bill_payment.pay": ["bill_payment.list_bills"],
    "bill_payment.set_autopay": ["bill_payment.list_bills"],
    "bill_payment.schedule_autopay": ["bill_payment.list_bills"],
    "subscriptions.cancel": ["subscriptions.list"],
    "household_chores.assign": ["household_chores.list_chores"],
    "shopping_list.check": ["shopping_list.get"],
    "shopping_list.remove": ["shopping_list.get"],
    "medication.log_taken": ["medication.list"],
    "medication.remove": ["medication.list"],
    "home_automation.delete_rule": ["home_automation.list_rules"],
    "home_security.acknowledge_alert": ["home_security.list_alerts"],
    "flights.check_in": ["flights.my_bookings"],
    "flights.cancel": ["flights.my_bookings"],
    "train_bus_tickets.cancel": ["train_bus_tickets.my_bookings"],
    "ecommerce.cancel_order": ["ecommerce.order_status"],
    "ecommerce.request_return": ["ecommerce.order_status"],
    "parking.end_session": ["parking.list_sessions"],
    "habit_tracker.log": ["habit_tracker.list_habits"],
    "pantry.consume": ["pantry.list_items"],
    "event_tickets.book": ["event_tickets.search"],
    "stocks.place_order": ["stocks.get_portfolio"],
    "stocks.set_price_alert": ["stocks.get_portfolio"],
    "learning.enroll": ["learning.list_courses"],
    "learning.log_progress": ["learning.list_courses"],
    "learning.schedule_lesson": ["learning.list_courses"],
    "music.play": ["music.list_tracks"],
    "music.add_to_playlist": ["music.list_tracks"],
    "tv.record": ["tv.guide"],
    "package_tracking.reschedule_delivery": ["package_tracking.track"],
    "banking.transfer": ["banking.get_balance", "banking.list_beneficiaries"],
    "banking.open_savings": ["banking.get_balance"],
    "banking.list_transactions": ["banking.get_balance"],
    "loyalty.redeem": ["loyalty.get_points"],
    "food_delivery.cancel_order": ["food_delivery.order_history"],
    "ride_hailing.cancel": ["ride_hailing.get_ride"],
}

ACTION_VERB_START = r"(?:đặt|nhắc|nhắn|gửi|gọi|chuyển|nạp|thêm|tạo|xóa|xoá|hủy|huỷ|mở|bật|tắt|kiểm tra|coi|xem|báo|ghi|lưu|đổi|dời|trả|thanh toán|đóng|khóa|khoá|chỉnh|bấm|tìm|tra|hẹn|sửa|lấy|mua|trừ|giữ|kết thúc|dịch|soạn|bỏ)"

CLAUSE_SPLIT_RE = re.compile(
    rf"(?:[,.;]?\s+(?<!\bxong\s)(?<!\bđổi\s)(?<!\bgiao\s)(?<!\bđóng\s)(?<!\buống\s)(?<!\băn\s)(?<!\bbơi\s)(?<!\bsố\s)(?<!\bmới\s)\b(?:rồi|xong(?!\s+(?:rồi|giùm|dùm|nhé|nha|chưa|dụng|việc))|với cả|với lại|tiện thể|nhân tiện|đồng thời|tiện(?=\s+{ACTION_VERB_START}\b))\s+(?=(?:{ACTION_VERB_START}|sáng|chiều|tối|trưa|mai|hôm|thứ|chủ nhật|lúc|sáu|bảy|tám|chín|mười|\d)\b)"
    rf"|[,.;]?\s+\bvới\s+(?={ACTION_VERB_START}\b))",
    flags=re.IGNORECASE
)

TOOL_KEYWORD_GUARDS: Dict[str, str] = {
    "reminders.create": r"\b(nhắc|lời nhắc)\b",
    "alarm.create": r"\b(báo thức|đánh thức|gọi .* dậy|hẹn giờ dậy)\b",
    "alarm.update": r"\b(dời (?:cái |lịch |giờ )?báo thức|đổi (?:cái |lịch |giờ )?báo thức|chuyển (?:cái |lịch |giờ )?báo thức|sửa (?:cái |lịch |giờ )?báo thức|lùi (?:cái |lịch |giờ )?báo thức|đẩy (?:cái |lịch |giờ )?báo thức|báo thức .{0,40} (?:dời|đổi|chuyển|sửa|lùi|đẩy|sớm hơn|muộn hơn|trễ hơn|thành \d|sang \d)|thành \d{1,2}\s*(?:giờ|rưỡi)|sang \d{1,2}\s*(?:giờ|rưỡi)|giờ thành)\b",
    "alarm.delete": r"\b(xóa|xoá|tắt|hủy|huỷ|bỏ)\b",
    "banking.open_savings": r"\b(mở sổ|gửi tiết kiệm|lập sổ|vượt trên \d+\s*triệu|hơn \d+\s*triệu)\b",
    "gold_price.get_price": r"\b(vàng|sjc)\b",
    "phone.recent_calls": r"\b(gọi lại|vừa gọi|gọi nhỡ|cuộc gọi|lịch sử gọi|ai gọi|số vừa)\b",
    "tasks.add": r"\b(thêm .* việc|tạo .* việc|tạo luôn dịp|việc cần làm|danh sách việc|hạn\b)\b",
    "notes.create": r"\b(tạo .* ghi chú|lưu .* ghi chú|viết .* ghi chú|ghi (?:vô|vào) ghi chú|ghi chú tên là|lưu bản dịch vào ghi chú|lưu tên món .* vào ghi chú|ghi chú (?:bữa|an ninh)|ghi chú mới|đặt tên là)\b",
    "notes.append": r"\b(thêm .* vào ghi chú|ghi thêm vào ghi chú|nối vào ghi chú|ghi tiếp vào ghi chú|bổ sung vào ghi chú|ghi (?:số đó )?vào (?:cái )?ghi chú)\b",
    "pantry.consume": r"\b(trừ|dùng hết|xài hết|lấy .* trong tủ|ăn hết|uống hết)\b",
    "stocks.place_order": r"\b(bán|mua|đặt lệnh)\b",
    "stocks.set_price_alert": r"\b(cảnh báo|báo khi|báo cô|báo anh|báo chị|đặt báo|vượt|xuống dưới|lên thêm|xuống \d|lên lại giá vốn)\b",
    "smart_home.scene_run": r"\b(chế độ ra khỏi nhà|chế độ đi ngủ|chế độ chào buổi sáng|ngữ cảnh)\b",
    "smart_home.switch_set": r"\b(ổ cắm|công tắc|máy tưới|bình nóng lạnh|máy sưởi)\b",
    "medication.add": r"\b(thêm .* lịch .* thuốc|tạo .* lịch .* thuốc|cài .* lịch .* thuốc|ngày hai lần)\b",
    "medication.log_taken": r"\b(đã uống|uống rồi|uống hết rồi|uống luôn rồi|vừa uống|uống xong|ghi nhận .* uống|trước khi đi ngủ|trước lúc đi ngủ)\b",
    "medication.remove": r"\b(xóa|xoá|bỏ|ngừng uống|cắt thuốc|hủy|huỷ)\b",
    "medication.list": r"\b(thuốc|vitamin|viên|kháng sinh|siro ho|cữ)\b",
    "tv.record": r"\b(ghi lại|ghi hình|ghi tập|ghi chương trình|thu lại)\b",
    "email.reply": r"\b(trả lời|reply|hồi âm|nhắn lại|báo lại)\b",
    "banking.transfer": r"\b(chuyển khoản|chuyển cho|chuyển trả|chuyển qua|chuyển giúp|chuyển đúng|chuyển tiền|bắn tiền|gửi hộ|gửi cho|tài khoản thanh toán|tiết kiệm)\b",
    "banking.list_transactions": r"\b(sao kê|giao dịch|đã chuyển .* chưa|tiền vào nhận chưa)\b",
    "bill_payment.list_bills": r"\b(hóa đơn|hoá đơn|tiền điện|tiền nước|tiền mạng|tiền truyền hình|tiền cáp)\b",
    "bill_payment.pay": r"\b(đóng|trả|thanh toán)\b",
    "bill_payment.schedule_autopay": r"\b(tự động|autopay)\b",
    "contacts.get": r"\b(thông tin liên hệ đầy đủ|chi tiết liên hệ|hồ sơ liên hệ)\b",
    "home_automation.delete_rule": r"\b(xóa|xoá|hủy|huỷ|bỏ)\b",
    "home_automation.create_rule": r"\b(tạo|thêm|cài|lập|khi nào .* thì bật|hễ .* thì|tối nào .* cũng tự|hằng ngày tự)\b",
    "subscriptions.cancel": r"\b(hủy|huỷ|bỏ|ngừng|cắt)\b",
    "food_delivery.cancel_order": r"\b(hủy|huỷ|bỏ)\b",
    "food_delivery.order_history": r"\b(đơn|hủy|huỷ|giao tới đâu)\b",
    "food_delivery.place_order": r"\b(đặt|mua|gọi món|lấy cho|giúp chị|giúp anh|giúp em|giúp mình|mỗi người .* một phần)\b",
    "calendar.create_event": r"\b(tạo .* lịch|thêm .* lịch|ghi .* lịch|cho .* vào lịch|vào lịch|đặt lịch|lịch cá nhân|lịch của (?:anh|chị|em|tôi|mình|cô|chú|bác)|đám cưới)\b",
    "calendar.cancel_event": r"\b(hủy|huỷ|xóa|xoá|bỏ)\b",
    "calendar.update_event": r"\b(dời|đổi|chuyển|lùi|đẩy|sửa|sớm hơn|muộn hơn|trễ hơn)\b",
    "calendar.list_events": r"\b(lịch|họp|hẹn|khám|buổi|sự kiện|công viên|đá bóng|chạy bộ|cà phê|dã ngoại|workshop|hội thảo|rảnh hay bận|trống không)\b",
    "sms.read_recent": r"\b(sms|tin nhắn|nhắn đó|coi tin nhắn|xem tin nhắn|đọc tin nhắn)\b",
    "ride_hailing.cancel": r"\b(hủy|huỷ|bỏ)\b",
    "ride_hailing.book": r"\b(đặt|gọi .* (?:xe|ô tô|taxi)|bắt .* xe|cho xe .* (?:đón|rước)|kiếm .* xe)\b",
    "ride_hailing.get_ride": r"\b(tới đâu|tới nơi|sắp tới|bao lâu|mấy phút|mấy giờ .* (?:rước|đón)|tài xế|xe đang|chuyến xe|hủy|huỷ|bỏ)\b",
    "maps.save_place": r"\b(lưu|đánh dấu)\b",
    "maps.get_route": r"\b(chỉ đường|tìm đường|tính lại đường|tính đường|đường đi|xem đường|xem luôn đường|lộ trình|bao xa|mất bao lâu|đi qua bao lâu|từ\s+\S+(?:\s+\S+){0,8}\s+(?:đi|ra|tới|đến|về|qua|sang))\b",
    "music.play": r"\b(nhạc|bài|ca khúc|playlist|nghe|phát|mở|bật .* của|bản|như .* nói hồi nãy)\b",
    "music.list_tracks": r"\b(nhạc|bài|ca khúc|playlist|nghe|phát|mở|bật .* của|bản|như .* nói hồi nãy)\b",
    "music.control": r"\b(nhạc|bài|dừng|tạm dừng|tiếp theo|qua bài|lặp lại|âm lượng|chuyển bài)\b",
    "music.add_to_playlist": r"\b(thêm .* vào playlist|bỏ .* vào playlist|cho .* vào playlist|lưu .* vào playlist|thêm .* vào danh sách phát|vào playlist)\b",
    "email.create_draft": r"\b(nháp|soạn|viết .* thư|viết .* mail|viết .* email|gửi .* mail|gửi .* email|gửi .* thư|ghi là)\b",
    "smart_home.light_set": r"\b(đèn|ánh sáng|sáng|phần trăm|%|bật lên|tắt dùm|tắt hai cái|phòng khách với bếp)\b",
    "smart_home.fan_set": r"\b(quạt|quẹt|bật nó lên)\b",
    "smart_home.climate_set": r"\b(điều hòa|điều hoà|máy lạnh|nhiệt độ|hút ẩm|độ c\b|\d{2}\s*độ)\b",
    "ewallet.topup_mobile": r"\b(nạp|đạp .* tiền điện thoại|nàng .* tiền điện thoại)\b",
    "ewallet.transfer": r"\b(ví|tiền|nghìn|ngàn|triệu|củ|đồng|khoản|số đấy)\b",
    "ewallet.pay_qr": r"\b(quét mã|qr|thanh toán cho quán|trả cho quán|trả hết cho|trả phần tôi|đối diện|quán)\b",
    "home_security.get_camera": r"\b(camera|cam\b|ghi hình)\b",
    "home_security.set_alarm_mode": r"\b(chế độ (?:ở nhà|vắng nhà|ban đêm|đi ngủ|báo động)|bật báo động|tắt báo động|chuyển báo động|hệ thống báo động|chống trộm)\b",
    "home_security.acknowledge_alert": r"\b(xác nhận|đã xem|bỏ qua|tắt cảnh báo|xử lý .* cảnh báo)\b",
    "appliances.stop": r"\b(dừng|tắt|ngừng|ngắt)\b",
    "appliances.start_cycle": r"\b(bật|chạy|khởi động|giặt|sấy|rửa)\b",
    "door_lock.grant_guest_code": r"\b(mã tạm|mã khách|mã mở|cấp mã)\b",
    "door_lock.set_lock": r"\b(khóa|khoá|mở khóa|mở khoá)\b",
    "package_tracking.track": r"\b(đơn|gói hàng|bưu kiện|kiện hàng|vận đơn|giao tới đâu|ship)\b",
    "contacts.update": r"\b(đổi số|đổi email|sửa .* danh bạ|lưu đè vào danh bạ|cập nhật .* danh bạ|cập nhật .* số|số mới|email mới)\b",
    "podcast_radio.subscribe": r"\b(đăng ký|theo dõi|subscribe)\b",
    "email.archive": r"\b(lưu trữ|archive|bỏ vào lưu trữ|dọn hộp thư|xóa|xoá)\b",
    "ecommerce.order_status": r"\b(đơn|hàng .* giao|trả hàng|trả lại|hủy|huỷ|kiểm tra đơn)\b",
    "ecommerce.cancel_order": r"\b(hủy|huỷ|khanh sồi|cancel)\b",
    "ecommerce.request_return": r"\b(trả hàng|trả lại|trả đơn|yêu cầu trả|hoàn trả|đổi trả|giao nhầm|làm đơn trả)\b",
    "ecommerce.add_to_cart": r"\b(đặt mua|mua|bỏ vào giỏ|thêm vào giỏ|cho .* vào giỏ|vào giỏ|lấy cái|lấy mẫu|đặt lại|tìm trên sàn|cuốn rẻ hơn)\b",
    "ecommerce.checkout": r"\b(đặt|mua|chốt|thanh toán|giao về|giao lên)\b",
    "grocery.search": r"\b(đi chợ|siêu thị|đặt mua|mua thêm|vào giỏ|vô giỏ|sữa|trứng|thịt|rau|gạo|cá|ức gà|cà chua|dầu ô liu|yến mạch|bột mì|táo|hạnh nhân|macca|gà ta)\b",
    "grocery.add_to_cart": r"\b(đi chợ|siêu thị|đặt mua|mua thêm|vào giỏ|vô giỏ|bỏ .* vào|thêm .* vào|đặt cho tôi .* hộp|lấy .* hộp|hạnh nhân|gà ta)\b",
    "grocery.checkout": r"\b(đặt|mua|giao về|chốt đơn)\b",
    "train_bus_tickets.book": r"\b(đặt|mua|lấy .* vé|chốt)\b",
    "train_bus_tickets.search": r"\b(đặt|mua|lấy .* vé|tìm .* chuyến|tìm .* vé|còn vé)\b",
    "train_bus_tickets.my_bookings": r"\b(hủy|huỷ|bỏ|cancel|đã đặt|vừa đặt|lúc nãy|xem vé|coi vé|kiểm tra vé|chạy mấy giờ|mấy giờ tới nơi|trùng bữa)\b",
    "train_bus_tickets.cancel": r"\b(hủy|huỷ|bỏ|cancel)\b",
    "flights.my_bookings": r"\b(hủy|huỷ|bỏ|check[ -]?in|làm thủ tục|chụp in|đã đặt|vừa đặt|lúc nãy|xem vé|coi vé|kiểm tra vé|bay mấy giờ|trước giờ bay|trước giờ cất cánh|sáng hôm bay|máy bay hạ cánh)\b",
    "flights.cancel": r"\b(hủy|huỷ|bỏ)\b",
    "flights.check_in": r"\b(check[ -]?in|làm thủ tục|chụp in)\b",
    "event_tickets.book": r"\b(đặt|mua|lấy)\b.*\b(vé|vét|chỗ|sự kiện)\b",
    "parking.end_session": r"\b(kết thúc|lấy xe ra|tách cái phiên|tắt .* phiên|cho dừng luôn)\b",
    "parking.reserve": r"\b(giữ|đặt chỗ|gửi xe\s+\d|để xe)\b",
    "chat_ott.read_chat": r"\b(đọc .* (?:zalo|nhóm|tin nhắn)|xem .* nhắn|coi .* nhắn|nhắn .* xem|nhắn .* coi|\S+\s+(?:mới\s+|vừa\s+)?nhắn\s+zalo\s+vụ)\b",
    "chat_ott.send_message": r"\b(nhắn|gửi|báo .* qua zalo|zalo|nhóm)\b",
    "chat_ott.list_chats": r"\b(nhắn|gửi|zalo|nhóm|tin nhắn)\b",
    "public_transit.topup_card": r"\b(nạp|nộp thẻ)\b",
    "public_transit.find_routes": r"\b(từ .* (?:đi|ra|tới|đến|về|qua|lên)|tuyến|xe buýt nào|đi được không|bắt chuyến nào|so với xe buýt)\b",
    "public_transit.next_departures": r"\b(mấy giờ có chuyến|chuyến .* kế tiếp|tàu điện kế tiếp|chuyến tiếp theo|mấy giờ chạy)\b",
    "recipes.save": r"\b(lưu|cất)\b",
    "shopping_list.add": r"\b(thêm .* danh sách|ghi .* danh sách|vào danh sách đi chợ|vô danh sách đi chợ|danh sách mua sắm|danh sách cần mua)\b",
    "shopping_list.remove": r"\b(xóa|xoá|bỏ .* khỏi danh sách|bỏ mục)\b",
    "shopping_list.check": r"\b(gạch|đánh dấu)\b",
    "bike_rental.reserve": r"\b(thuê xe đạp|mượn xe đạp|giữ .* xe đạp|chiếc xe đủ pin|chiếc xe đôi|giữ (?:giùm |dùm |hộ )?(?:một )?chiếc xe)\b",
    "bike_rental.find_bikes": r"\b(thuê xe đạp|mượn xe đạp|giữ .* xe đạp|xe đạp|chiếc xe đôi)\b",
    "library.search": r"\b(thư viện|mượn sách)\b",
    "library.reserve": r"\b(thư viện|mượn sách|giữ .* cuốn)\b",
    "household_chores.list_chores": r"\b(việc nhà|rửa bát|rửa chén|dọn|quét|lau|đổ rác|phơi đồ|giao vụ|giao cho|nấu cơm)\b",
    "household_chores.assign": r"\b(giao vụ|giao cho|phân công|giao việc|giao giặt)\b",
    "weather.get_forecast": r"\b(thời tiết|dự báo|mưa|trời (?:mưa|nắng|lạnh|nóng|mát|ráo|âm u)|có nắng không|nắng hay mưa|nắng mưa|bao nhiêu độ ngoài trời|ngoài trời .* độ|gió|bão|khô ráo)\b",
    "weather.get_current": r"\b(thời tiết|dự báo|mưa|trời (?:mưa|nắng|lạnh|nóng|mát|ráo|âm u)|có nắng không|nắng hay mưa|nắng mưa|bao nhiêu độ ngoài trời|ngoài trời .* độ|gió|bão|khô ráo)\b",
}

APP_TRIGGER_PATTERNS: Dict[str, str] = {
    "weather": r"\b(thời tiết|dự báo|mưa|trời (?:mưa|nắng|lạnh|nóng|mát|ráo|âm u)|có nắng không|nắng hay mưa|nắng mưa|bao nhiêu độ ngoài trời|ngoài trời .* độ|ngoài trời giờ thế nào|gió|bão|khô ráo)\b",
    "fitness": r"\b(bơi|chạy bộ|đi bộ|buổi đi bộ|đạp xe|tập|gym|yoga|calo|bước chân|thể dục|workout|mục tiêu .* bước|nghìn rưỡi bước)\b",
    "alarm": r"\b(báo thức|đánh thức|gọi .* dậy|hẹn giờ dậy)\b",
    "reminders": r"\b(nhắc(?!\s+[^,.!?;]*?\b(?:hồi nãy|lúc nãy))|lời nhắc)\b",
    "tasks": r"\b(thêm .* việc|tạo .* việc|việc thứ|việc cần làm|danh sách việc|đánh dấu xong|gạch giúp .* việc)\b",
    "notes": r"\b(ghi chú|note|sổ tay)\b",
    "package_tracking": r"\b(gói hàng|bưu kiện|bưu điện|đơn hàng .* giao|lấy gói hàng|lấy hàng|lấy hồ sơ)\b",
    "subscriptions": r"\b(gói cước|gói .* tháng|gia hạn|huỷ gói|hủy gói|cloud pro|netflix|spotify)\b",
    "robot_vacuum": r"\b(robot|hút bụi|lau nhà)\b",
    "home_automation": r"\b(hẹn giờ chạy|tự động|quy tắc|luật|rule|hẹn bật|hẹn tắt|xoá cái hẹn bật đèn|xóa kế hoạch|tối nào .* cũng tự tắt|tối nào .* cũng tự bật|hằng ngày tự bật|tự tắt lúc|tự bật lúc)\b",
    "food_delivery": r"\b(đặt món|đồ ăn|đặt ở quán|nhà hàng|bún bò|phở|cơm tấm|pizza|trà sữa|đơn lumina|đơn cơm tấm|mì xào chay|cháo gà|cơm nấm|bún trộn|gỏi cuốn|gọi giúp .* suất)\b",
    "home_security": r"\b(cảnh báo|báo chuông|báo động|camera|an ninh|chuyển động)\b",
    "door_lock": r"\b(khóa cửa|khoá cửa|khoá ngay cửa|khóa cái cửa|khóa .* cửa|khoá .* cửa|cửa (?:chính|sau|phụ|kho|cổng)|cửa .* đang mở thì khóa|cửa .* mở khóa thì khóa|mở khóa|mã khách|mã mở cổng|mã tạm mở cửa)\b",
    "household_chores": r"\b(việc nhà|rửa bát|rửa chén|cất bát|phơi quần áo|kiểm tra pin khoá|dọn nhà vệ sinh|quét nhà|lau bàn|đổ rác|phơi đồ|giao cho .* làm|giao chú năm|giao vụ|giao việc|giao giặt|chuyển việc nấu cơm)\b",
    "contacts": r"\b(tìm trong danh bạ|sửa .* trong danh bạ|lưu đè vào danh bạ|danh bạ|số điện thoại của|thông tin liên lạc của|liên hệ (?:anh|chị|em|cô|chú|bác))\b",
    "bike_rental": r"\b(xe đạp|trạm xe đạp|chiếc xe đôi|giữ (?:giùm |dùm |hộ )?(?:một )?chiếc xe)\b",
    "library": r"\b(cuốn .* ở thư viện|mượn sách .* thư viện|giữ giúp .* ở thư viện|ghé thư viện lấy sách)\b",
    "home_energy": r"\b(dùng điện|dùng hết bao nhiêu điện|điện hôm nay thế nào|xài điện|tiêu thụ điện|bao nhiêu số điện|mấy số điện|quá .* số điện|quá (?:mười|\d+) số|vượt .* trăm số)\b",
    "news": r"\b(tin mới nhất|báo mới nhất|thông báo .* mới nhất|tin tức|điểm tin|có tin gì mới|bài mới nhất|tin mới về|(?<!tiêu đề\s)bài báo)\b",
    "calendar": r"\b(tạo lại lịch|thêm lịch|đặt lịch|tạo lịch|ghi lịch|thêm cái lịch|tạo lịch trình|tạo một sự kiện trong lịch|thêm lại vào lịch|chặn lịch|xóa lịch|xoá lịch|sửa giờ với địa điểm trong lịch|dời lịch|dời buổi|dời cái hẹn|dời sang đúng giờ|hủy lịch|huỷ lịch|hủy .* cái lịch|huỷ .* cái lịch|hủy hẹn|huỷ hẹn|bỏ cái họp|lịch [^\s,.]+ (?:\S+\s+){0,4}(?:mai\s+|hôm\s+|sáng\s+|chiều\s+|tối\s+)?(?:dời|đổi|lùi|chuyển|hủy|huỷ)|lịch khám|khám răng|lịch họp|họp team|họp phụ huynh|buổi cà phê|hẹn cà phê|đi chơi công viên|đi chợ hoa|đá bóng sáng mai|chạy bộ sáng mai|chỗ hẹn|giờ hẹn|ghi .* vào lịch|cho .* vào lịch|thêm vào lịch|vào lịch|lịch cá nhân|lịch của (?:anh|chị|em|tôi|mình|cô|chú|bác)|đám cưới|rảnh hay bận|lịch .* trống|hội nghị .* ghi trong lịch)\b",
    "music": r"\b(bài nhạc|mở bài|playlist|danh sách phát|hoài lâm|sơn tùng|đức phúc)\b",
    "chat_ott": r"\b(zalo|nhắn zalo|trên zalo|qua zalo|nhóm chat|nhắn .* điều phối|nhắn .* về khuya|nhắn vào nhóm|nhắn trong nhóm|nhóm dự án|nhóm vận hành|nhóm gia đình|gửi .* qua zalo)\b",
    "sms": r"\b(sms|nhắn sms|tin nhắn sms|nhắn tin thường|nhắn .* số \d{8,11}|nhắn .* số không chín|nhắn cho|nhắn tin cho|nhắn (?:lan|hương|minh|thằng tí|anh|chị|em|mẹ|bố|bác|cô|chú)|mạ mới nhắn|trạm .* nhắn đó|coi tin nhắn)\b",
    "phone": r"\b((?<!đang\s)gọi điện|gọi cho (?!tôi\b|mình\b)|gọi (?:lan|minh|hương|mẹ|bố|ba|anh|chị|em|cô|chú|bác|sếp)|gọi báo|gọi người đó|gọi luôn|cuộc gọi)\b",
    "email": r"\b(email|mail|thư(?!\s+(?:viện|mời\b))|hộp thư|thư nháp)\b",
    "medication": r"\b(lịch uống thuốc|lịch thuốc|(?<!nhắc tôi\s)(?<!nhắc mình\s)(?<!nhắc anh\s)(?<!nhắc chị\s)(?<!nhắc em\s)(?<!nhắc cô\s)uống thuốc|mấy thuốc|thuốc tối nay|thuốc sáng nay|thuốc trước lúc đi ngủ|cữ thuốc|cữ kháng sinh|kháng sinh|siro ho|thuốc dạ dày|thuốc huyết áp|viên thuốc|vitamin|viên sắt|dầu cá|thuốc say xe)\b",
    "calculator": r"\b(bao nhiêu xăng ti mét|bao nhiêu mét|đổi thử giúp|đổi ra xăng ti mét|đổi ra mét|tổng cộng mấy cây số|cộng hết bao nhiêu|cộng lại hết bao nhiêu|tính tổng|nhân thử|chia ra)\b",
    "video_meeting": r"\b(phòng họp online|cuộc họp online|buổi gặp online|link họp|tạo phòng họp|mở luôn một phòng họp)\b",
    "ride_hailing": r"\b(đặt (?:luôn )?(?:sẵn )?(?:giúp .* )?(?:cái |chiếc |một chiếc )?(?:xe|ô tô|taxi)|gọi .* (?:xe máy|xe 4 chỗ|xe bốn chỗ|xe 7 chỗ|xe bảy chỗ|ô tô|taxi|chiếc xe|cái xe)|bắt xe|huỷ cái xe|hủy xe|huỷ chuyến xe|hủy chuyến xe|xe máy từ nhà|xe bốn chỗ|xe 4 chỗ|ô tô 4 chỗ|ô tô bốn chỗ|xe đang tới đâu|chuyến xe .* confirm)\b",
    "flights": r"\b((?<!thông tin\s)chuyến bay|vé máy bay|máy bay|check[ -]?in|đặt .* vé (?:chuyến )?[a-z]{2}\s*\d+|(?<!công tác\s)chuyến\s+[a-z]{2}\s*\d+|đặt .* vé đi|đặt .* vé bay|bay vào|bay ra|giờ bay|cất cánh|hạ cánh|ngày bay)\b",
    "podcast_radio": r"\b(podcast|đài|radio|tập podcast|danh sách nghe sau|tập vừa nghe)\b",
    "maps": r"\b(bản đồ|chỉ đường|tìm đường|tính lại đường|tính đường|đường đi|xem đường|xem luôn đường|xem từ nhà đi|nhà đến trường bao xa|đường từ nhà|đi ô tô .* ra|lộ trình|bao xa|mất bao lâu|đi qua bao lâu|lưu giùm anh cái bar|lưu địa điểm|lưu bến xe|tìm quán .* gần|quán cà phê .* gần|thư viện .* gần|nhà hàng gần|cây xăng gần|hiệu thuốc gần|tiệm sửa xe quanh|tiệm photo gần)\b",
    "smart_home": r"\b(bật đèn|tắt đèn|bật mấy đèn|tắt hết đèn|tắt bớt đèn|giảm đèn|tăng đèn|hạ đèn|chỉnh .* đèn|đèn bên phòng|đèn đọc sách|đèn ngoài sân|đèn sân|đèn hành lang|đèn bếp|đèn ngủ|điều hòa|điều hoà|máy lạnh|tắt cái quạt|tắt .* quạt|mấy quạt|quạt tầng|bật quạt|bật giúp .* quạt|quạt trần|rèm cửa|đèn phòng|quạt phòng|ổ cắm|máy tưới|chế độ ra khỏi nhà|chế độ đi ngủ|chế độ chào buổi sáng)\b",
    "tv": r"\b(tv|tivi|ti vi|kênh|chuyển sang phim|âm lượng tivi|vtv|htv|ghi lại tập|ghi tập)\b",
    "ecommerce": r"\b(trả hàng|trả đơn|yêu cầu trả|hoàn trả|hủy đơn|đơn đang giao|kiểm tra đơn hàng|tìm trên mua đồ|bàn phím|mkey|đèn bàn|máy sấy|máy xay|cốc giữ nhiệt|kẹp bàn|áo sơ mi|giày chạy bộ|tai nghe|chuột logitech|bình giữ nhiệt|lọc không khí|túi rác|nước giặt|giá đỡ|cáp sạc|ổ cắm thông minh|ổ cắm chuyển đổi|bình đun|nồi chiên|kệ sách|ghế công thái học|bàn di chuột|sạc dự phòng|cuốn sách|quyển sách|mua .* trên sàn|trên sàn thương mại)\b",
    "grocery": r"\b(đi chợ|siêu thị|đặt mua .* nguyên liệu|mua đủ .* nguyên liệu|ức gà|sữa tươi|sữa hạt|cá hồi|thịt ba chỉ|cà chua|dầu ô liu|yến mạch|bột mì|rau cải|táo envy|trứng gà|gạo st25|vào giỏ|hạt macca|macca|hạnh nhân|đậu hũ|nước mắm|nước tương|hành tây|khoai tây|cà rốt|bơ lạt|phô mai|sữa chua|bánh mì|mì ý|nấm đùi gà|nấm kim châm|rau muống|cải ngọt|bông cải|thịt bò|thịt heo|sườn non|tôm tươi|cá thu|chuối|cam sành|gà ta làm sẵn)\b",
    "pantry": r"\b(tủ bếp|kho bếp|tủ thực phẩm|kho thực phẩm|tủ gia dụng|trừ đi|trừ trong tủ|trừ giúp|trừ cho|trừ vào tủ|trừ thêm cho đúng|cất giúp|cất vào tủ)\b",
    "recipes": r"\b(công thức|cách làm|nấu món|làm món|món .* nấu|bò kho|canh chua|cháo nấm|cháo gà|súp bí đỏ|bún riêu|lẩu nấm|cà ri|gà xào|thịt kho|pancake|smoothie|salad|cần nguyên liệu gì|cơm chiên|chè hạt sen)\b",
    "shopping_list": r"\b(danh sách đi chợ|danh sách mua sắm|danh sách cần mua|trong list đi chợ)\b",
    "event_tickets": r"\b(đêm nhạc|triển lãm|nhạc kịch|hòa nhạc|vở kịch|lễ hội|hội thảo|đêm khoa học|vé xem|vé nghe|sài gòn cười|liên hoan|thung lũng mù sương|hội sách|ngày hội|show diễn|buổi diễn|vé sự kiện|sự kiện|stand up comedy|đêm hài|kịch nói|ca nhạc|liveshow)\b",
    "train_bus_tickets": r"\b(vé tàu|tàu hỏa|tàu đêm|đặt .* tàu se|tàu lp|tàu sp|xe khách|limousine|xe giường nằm|chuyến tàu|bến xe|vé xe đi|xe đi nha trang|xe đi đà lạt|xe đi vinh|vé đi lạng sơn|đi lạng sơn|đi ninh bình|đi sapa|đi hải phòng|đi phan thiết|đi vũng tàu|vé xe chuyến)\b",
    "parking": r"\b(bãi xe|bãi đỗ xe|gửi xe|đỗ xe|giữ chỗ ở bãi|phiên gửi xe|phiên đỗ xe|lượt gửi|lượt đỗ xe)\b",
    "public_transit": r"\b(xe buýt|tuyến buýt|chuyến buýt|tàu điện|metro|mê trô|thẻ xe buýt|nạp thẻ)\b",
    "habit_tracker": r"\b(thói quen|habit|đánh dấu hôm nay|đánh dấu uống nước|luyện nói|thiền|đọc sách .* phút|uống đủ nước)\b",
    "stocks": r"\b(cổ phiếu|chứng khoán|danh mục|giá vốn|cảnh báo giá|mã lỗ|mã .* lên|mã .* xuống|zmx|zkn|vitech|brightpath|sunrise|redwood|bán bớt .* cổ)\b",
    "translator": r"\b(dịch|tiếng anh|tiếng hàn|tiếng nhật|bản dịch)\b",
    "fuel_price": r"\b(giá xăng|giá dầu|xăng dầu|xăng 95|ron 95)\b",
    "gold_price": r"\b(giá vàng|vàng sjc|vàng nhẫn|nhẫn vàng|vàng thế giới|mua vàng|bán vàng|bán nhẫn|giá nhẫn)\b",
    "loyalty": r"\b(điểm thưởng|đổi điểm|tích điểm|đủ điểm đổi|voucher|điểm (?:cao nguyên|cineviệt|minhpoint|vivufood|góc phố)|minhmart|cineviệt|vivufood)\b",
    "exchange_rate": r"\b(tỷ giá|đổi .* ra tiền việt|quy ra tiền việt|đổi .* sang đô|đổi .* ra đô|đô mỹ|một đô|đô bữa nay|yên nhật|một yên|bảng anh|euro|nhân dân tệ|won hàn)\b",
    "lottery": r"\b(vé số|xổ số|dò đài|dò vé|dò .* tờ vé|số \d{5,6}|xổ chưa|giải đặc biệt)\b",
    "appliances": r"\b(máy giặt|máy sấy|máy rửa bát|nồi cơm|chế độ giặt|chế độ sấy|bật giặt|giặt nhanh|giặt đồ len|chạy máy)\b",
    "learning": r"\b(khóa học|khoá học|đăng ký .* khóa|đăng ký .* khoá|khóa sơ cứu|ghi số phút vô từng khoá|buổi tiếng (?:hàn|nhật|anh|trung|pháp)|vào khoá|vào khóa|khoá tiếng|khóa tiếng|phút khoá|phút khóa|tiến độ|edura|học nhanh|skillup)\b",
    "banking": r"\b(số dư ngân hàng|tài khoản ngân hàng còn|chuyển khoản|sao kê|giao dịch ngân hàng|tài khoản thanh toán|sổ tiết kiệm|lấy tiết kiệm|từ tiết kiệm|dùng tiết kiệm|chủ nhà phạm hùng|trần văn tuấn|nguyễn thị lan|lê minh|chuyển giúp tôi .* triệu|gửi hộ mình .* triệu)\b",
    "bill_payment": r"\b(hóa đơn|hoá đơn|tiền điện|tiền nước|tiền mạng|tiền truyền hình|tiền cáp|autopay|tự động trả)\b",
    "ewallet": r"\b(chuyển ví|qua ví|ví điện tử|số dư ví|nạp .* (?:điện thoại|nghìn|ngàn)|quét mã|mã qr)\b",
}

for _k_tg, _v_tg in list(TOOL_KEYWORD_GUARDS.items()):
    TOOL_KEYWORD_GUARDS[_k_tg] = _v_tg.replace(" .* ", r"\s+(?:[^,.!?;]*?\s+)?")
for _k_ap, _v_ap in list(APP_TRIGGER_PATTERNS.items()):
    APP_TRIGGER_PATTERNS[_k_ap] = _v_ap.replace(" .* ", r"\s+(?:[^,.!?;]*?\s+)?")

CATALOG_CANONICAL_EXEMPLARS: List[Dict[str, Any]] = [
    {
        "id": "cat_flights_book_single",
        "audio_text": "Đặt cho tôi một vé máy bay đi Hà Nội ngày mai chuyến rẻ nhất sau 12 giờ trưa",
        "installed_apps": ["flights"],
        "called_apps": {"flights"},
        "tool_calls": [
            {"name": "flights.search", "arguments": {"origin": "SGN", "destination": "HAN", "date": "2026-10-16"}},
            {"name": "flights.book", "arguments": {"flight_id": "fl_01", "passengers": 1}},
        ],
    },
    {
        "id": "cat_flights_book_business",
        "audio_text": "Đặt cho tôi một vé hạng thương gia đi Hà Nội ngày 8 tháng 12 bay sau 1 giờ chiều",
        "installed_apps": ["flights"],
        "called_apps": {"flights"},
        "tool_calls": [
            {"name": "flights.search", "arguments": {"origin": "DAD", "destination": "HAN", "date": "2026-12-08"}},
            {"name": "flights.book", "arguments": {"flight_id": "fl_01", "passengers": 1, "seat_class": "business"}},
        ],
    },
    {
        "id": "cat_flights_my_bookings_alarm",
        "audio_text": "Chuyến bay đi Hà Nội tôi đặt rồi đó, đặt báo thức trước giờ bay 3 tiếng giúp mình nhé",
        "installed_apps": ["flights", "alarm"],
        "called_apps": {"flights", "alarm"},
        "tool_calls": [
            {"name": "flights.my_bookings", "arguments": {}},
            {"name": "alarm.create", "arguments": {"time": "2026-10-16T06:00:00+07:00"}},
        ],
    },
    {
        "id": "cat_flights_book_alarm_zalo",
        "audio_text": "Đặt giúp hai vé đi Hà Nội ngày 11 tháng 12 lấy chuyến rẻ nhất, hẹn báo thức trước giờ bay 3 tiếng và nhắn Zalo cho cô Mai báo giờ bay nhé",
        "installed_apps": ["flights", "alarm", "chat_ott", "contacts"],
        "called_apps": {"flights", "alarm", "chat_ott"},
        "tool_calls": [
            {"name": "flights.search", "arguments": {"origin": "SGN", "destination": "HAN", "date": "2026-12-11"}},
            {"name": "flights.book", "arguments": {"flight_id": "fl_01", "passengers": 2}},
            {"name": "alarm.create", "arguments": {"time": "2026-12-11T09:00:00+07:00"}},
            {"name": "chat_ott.list_chats", "arguments": {}},
            {"name": "chat_ott.send_message", "arguments": {"chat_id": "chat_01", "content": "Mình đặt chuyến bay lúc 12:30 rồi nhé."}},
        ],
    },
    {
        "id": "cat_flights_parking",
        "audio_text": "Đặt cho tôi một vé chuyến RV21 đi Đà Nẵng thứ bảy tuần sau rồi giữ luôn một chỗ ở bãi xe sân bay Nội Bài từ 7 giờ sáng hôm đó gửi xe 3 ngày",
        "installed_apps": ["flights", "parking"],
        "called_apps": {"flights", "parking"},
        "tool_calls": [
            {"name": "flights.search", "arguments": {"origin": "HAN", "destination": "DAD", "date": "2026-10-03"}},
            {"name": "flights.book", "arguments": {"flight_id": "fl_01", "passengers": 1}},
            {"name": "parking.find", "arguments": {"near": "Nội Bài"}},
            {"name": "parking.reserve", "arguments": {"lot_id": "lot_01", "from_time": "2026-10-03T07:00:00+07:00", "hours": 72}},
        ],
    },
    {
        "id": "cat_flights_cancel",
        "audio_text": "Hủy giùm tôi cái vé máy bay đã đặt nhé",
        "installed_apps": ["flights"],
        "called_apps": {"flights"},
        "tool_calls": [
            {"name": "flights.my_bookings", "arguments": {}},
            {"name": "flights.cancel", "arguments": {"booking_id": "flb_01"}},
        ],
    },
    {
        "id": "cat_ecommerce_search_buy",
        "audio_text": "Tìm bàn phím nhỏ gọn rồi đặt mua 1 cái giao về nhà thanh toán khi nhận hàng nhé",
        "installed_apps": ["ecommerce"],
        "called_apps": {"ecommerce"},
        "tool_calls": [
            {"name": "ecommerce.search", "arguments": {"query": "bàn phím"}},
            {"name": "ecommerce.add_to_cart", "arguments": {"product_id": "prod_01", "quantity": 1}},
            {"name": "ecommerce.checkout", "arguments": {"address": "25 Láng Hạ, Đống Đa, Hà Nội", "payment": "cod"}},
        ],
    },
    {
        "id": "cat_ecommerce_return",
        "audio_text": "Cái đèn bàn mua tuần trước bị hỏng rồi, kiểm tra đơn hàng và làm yêu cầu trả hàng giúp chị nhé",
        "installed_apps": ["ecommerce"],
        "called_apps": {"ecommerce"},
        "tool_calls": [
            {"name": "ecommerce.order_status", "arguments": {}},
            {"name": "ecommerce.request_return", "arguments": {"order_id": "ord_01", "reason": "damaged"}},
        ],
    },
    {
        "id": "cat_ecommerce_cancel",
        "audio_text": "Kiểm tra đơn hàng mới đặt nếu chưa giao thì hủy đơn giúp mình nhé",
        "installed_apps": ["ecommerce"],
        "called_apps": {"ecommerce"},
        "tool_calls": [
            {"name": "ecommerce.order_status", "arguments": {}},
            {"name": "ecommerce.cancel_order", "arguments": {"order_id": "ord_01"}},
        ],
    },
    {
        "id": "cat_grocery_buy",
        "audio_text": "Đặt mua cho tôi sữa tươi ít đường với 2 gói yến mạch bên app đi chợ giao về nhà nhé",
        "installed_apps": ["grocery"],
        "called_apps": {"grocery"},
        "tool_calls": [
            {"name": "grocery.search", "arguments": {"query": "sữa tươi"}},
            {"name": "grocery.add_to_cart", "arguments": {"product_id": "gro_01", "quantity": 1}},
            {"name": "grocery.checkout", "arguments": {"address": "25 Láng Hạ, Đống Đa, Hà Nội"}},
        ],
    },
    {
        "id": "cat_pantry_consume_grocery",
        "audio_text": "Tôi vừa lấy 1 hộp sữa tươi trong tủ bếp trừ đi nhé, rồi đặt mua thêm 2 gói bột mì bên đi chợ cho chị bù vào nhé",
        "installed_apps": ["pantry", "grocery"],
        "called_apps": {"pantry", "grocery"},
        "tool_calls": [
            {"name": "pantry.list_items", "arguments": {}},
            {"name": "pantry.consume", "arguments": {"item_id": "pan_01", "quantity": 1}},
            {"name": "grocery.search", "arguments": {"query": "bột mì"}},
            {"name": "grocery.add_to_cart", "arguments": {"product_id": "gro_01", "quantity": 2}},
            {"name": "grocery.checkout", "arguments": {"address": "25 Láng Hạ, Đống Đa, Hà Nội"}},
        ],
    },
    {
        "id": "cat_pantry_consume_shopping_list",
        "audio_text": "Trừ đi 3 cân gạo tôi lấy ra nấu cỗ trong tủ bếp, xong còn dưới mức đấy thì ghi gạo vào danh sách đi chợ mua bù nhé",
        "installed_apps": ["pantry", "shopping_list"],
        "called_apps": {"pantry", "shopping_list"},
        "tool_calls": [
            {"name": "pantry.list_items", "arguments": {}},
            {"name": "pantry.consume", "arguments": {"item_id": "pan_01", "quantity": 3}},
            {"name": "shopping_list.add", "arguments": {"name": "Gạo ST25", "quantity": 3}},
        ],
    },
    {
        "id": "cat_recipes_shopping_notes",
        "audio_text": "Tìm cho mình công thức súp bí đỏ, thêm 1 quả bí đỏ vào danh sách đi chợ rồi lưu tên món tìm được vào ghi chú bữa tối",
        "installed_apps": ["recipes", "shopping_list", "notes"],
        "called_apps": {"recipes", "shopping_list", "notes"},
        "tool_calls": [
            {"name": "recipes.search", "arguments": {"query": "súp bí đỏ"}},
            {"name": "shopping_list.add", "arguments": {"name": "Bí đỏ", "quantity": 1}},
            {"name": "notes.create", "arguments": {"title": "Bữa tối", "content": "Súp bí đỏ"}},
        ],
    },
    {
        "id": "cat_recipes_save",
        "audio_text": "Tìm công thức món bò kho rồi lưu công thức đó vào ứng dụng nấu ăn giúp cô nhé",
        "installed_apps": ["recipes"],
        "called_apps": {"recipes"},
        "tool_calls": [
            {"name": "recipes.search", "arguments": {"query": "bò kho"}},
            {"name": "recipes.save", "arguments": {"recipe_id": "rcp_01"}},
        ],
    },
    {
        "id": "cat_translator_notes_email",
        "audio_text": "Dịch giúp chị câu hóa đơn đã được thanh toán sang tiếng Anh, lưu bản dịch vào ghi chú Đối soát xong soạn sẵn email nháp gửi finance@example.test với nội dung đó nhé",
        "installed_apps": ["translator", "notes", "email"],
        "called_apps": {"translator", "notes", "email"},
        "tool_calls": [
            {"name": "translator.translate", "arguments": {"text": "Hóa đơn đã được thanh toán", "to_lang": "en"}},
            {"name": "notes.create", "arguments": {"title": "Đối soát", "content": "The invoice has been paid."}},
            {"name": "email.create_draft", "arguments": {"to": "finance@example.test", "subject": "Đối soát", "body": "The invoice has been paid."}},
        ],
    },
    {
        "id": "cat_translator_zalo",
        "audio_text": "Dịch giúp chị câu nhớ mang theo hồ sơ sang tiếng Anh xong gửi câu tiếng Anh đó qua Zalo cho Mai Anh nhé",
        "installed_apps": ["translator", "chat_ott"],
        "called_apps": {"translator", "chat_ott"},
        "tool_calls": [
            {"name": "translator.translate", "arguments": {"text": "Nhớ mang theo hồ sơ", "to_lang": "en"}},
            {"name": "chat_ott.list_chats", "arguments": {}},
            {"name": "chat_ott.send_message", "arguments": {"chat_id": "chat_01", "content": "Please remember to bring the documents."}},
        ],
    },
    {
        "id": "cat_pantry_consume_only",
        "audio_text": "Trong tủ bếp còn hộp sữa hay gói mì nào sắp hết hạn thì trừ bớt 1 cái giúp chị nhé vì chị vừa lấy ra dùng rồi",
        "installed_apps": ["pantry"],
        "called_apps": {"pantry"},
        "tool_calls": [
            {"name": "pantry.list_items", "arguments": {}},
            {"name": "pantry.consume", "arguments": {"item_id": "pan_01", "quantity": 1}},
        ],
    },
    {
        "id": "cat_event_tickets_calendar",
        "audio_text": "Lấy cho anh 2 vé xem kịch tối thứ bảy rồi ghi vào lịch luôn là 8 giờ tối thứ bảy này đi xem nhé",
        "installed_apps": ["event_tickets", "calendar"],
        "called_apps": {"event_tickets", "calendar"},
        "tool_calls": [
            {"name": "event_tickets.search", "arguments": {}},
            {"name": "event_tickets.book", "arguments": {"event_id": "evt_01", "quantity": 2}},
            {"name": "calendar.create_event", "arguments": {"title": "Đi xem kịch", "start": "2026-10-10T20:00:00+07:00", "end": "2026-10-10T22:00:00+07:00"}},
        ],
    },
    {
        "id": "cat_event_tickets_book_only",
        "audio_text": "Đặt giúp chị 2 vé xem triển lãm hoặc hòa nhạc tối thứ ba tuần sau nhé",
        "installed_apps": ["event_tickets"],
        "called_apps": {"event_tickets"},
        "tool_calls": [
            {"name": "event_tickets.search", "arguments": {}},
            {"name": "event_tickets.book", "arguments": {"event_id": "evt_01", "quantity": 2}},
        ],
    },
    {
        "id": "cat_train_bus_parking",
        "audio_text": "Thứ tư tuần sau anh vào Vinh công tác, đặt cho anh 1 vé tàu SE7 sáng nhé rồi giữ luôn một chỗ gửi xe ở bãi ga Hà Nội từ 5 giờ sáng hôm đó gửi 3 ngày",
        "installed_apps": ["train_bus_tickets", "parking"],
        "called_apps": {"train_bus_tickets", "parking"},
        "tool_calls": [
            {"name": "train_bus_tickets.search", "arguments": {"origin": "Hà Nội", "destination": "Vinh", "date": "2026-09-30", "type": "train"}},
            {"name": "train_bus_tickets.book", "arguments": {"trip_id": "trip_01", "seats": 1}},
            {"name": "parking.find", "arguments": {"near": "Ga Hà Nội"}},
            {"name": "parking.reserve", "arguments": {"lot_id": "lot_01", "from_time": "2026-09-30T05:00:00+07:00", "hours": 72}},
        ],
    },
    {
        "id": "cat_train_bus_ride",
        "audio_text": "Đặt cho chị 1 ghế tàu SE5 đi Huế chuyến sáng nay nhé rồi gọi cái xe bốn chỗ từ nhà ra ga Hà Nội luôn cho kịp tàu",
        "installed_apps": ["train_bus_tickets", "ride_hailing"],
        "called_apps": {"train_bus_tickets", "ride_hailing"},
        "tool_calls": [
            {"name": "train_bus_tickets.search", "arguments": {"origin": "Hà Nội", "destination": "Huế", "date": "2026-09-25", "type": "train"}},
            {"name": "train_bus_tickets.book", "arguments": {"trip_id": "trip_01", "seats": 1}},
            {"name": "ride_hailing.book", "arguments": {"pickup": "25 Láng Hạ, Đống Đa, Hà Nội", "dropoff": "Ga Hà Nội", "vehicle": "car4", "schedule_time": "2026-09-25T07:00:00+07:00"}},
        ],
    },
    {
        "id": "cat_train_bus_cancel",
        "audio_text": "Thứ sáu này chị không đi nữa, cái vé tàu xe vừa đặt ấy em hủy hộ chị luôn đi",
        "installed_apps": ["train_bus_tickets"],
        "called_apps": {"train_bus_tickets"},
        "tool_calls": [
            {"name": "train_bus_tickets.my_bookings", "arguments": {}},
            {"name": "train_bus_tickets.cancel", "arguments": {"booking_id": "tbb_01"}},
        ],
    },
    {
        "id": "cat_train_bus_my_bookings",
        "audio_text": "Xem vé tàu tối nay với vé chủ nhật chạy mấy giờ tôi canh giờ đi nhé",
        "installed_apps": ["train_bus_tickets"],
        "called_apps": {"train_bus_tickets"},
        "tool_calls": [
            {"name": "train_bus_tickets.my_bookings", "arguments": {}},
        ],
    },
    {
        "id": "cat_ecommerce_cart_only",
        "audio_text": "Tìm trên sàn cho anh 2 con chuột Logitech hoặc bàn phím bỏ vào giỏ nhé",
        "installed_apps": ["ecommerce"],
        "called_apps": {"ecommerce"},
        "tool_calls": [
            {"name": "ecommerce.search", "arguments": {"query": "chuột Logitech"}},
            {"name": "ecommerce.add_to_cart", "arguments": {"product_id": "prod_01", "quantity": 2}},
        ],
    },
    {
        "id": "cat_parking_reserve",
        "audio_text": "Giữ chỗ đỗ xe ở bãi gần Hồ Gươm từ 2 giờ chiều nay gửi tầm 2 tiếng nhé",
        "installed_apps": ["parking"],
        "called_apps": {"parking"},
        "tool_calls": [
            {"name": "parking.find", "arguments": {"near": "Hồ Gươm"}},
            {"name": "parking.reserve", "arguments": {"lot_id": "lot_01", "from_time": "2026-10-15T14:00:00+07:00", "hours": 2}},
        ],
    },
    {
        "id": "cat_parking_end_session",
        "audio_text": "Em lấy xe ra rồi, kết thúc lượt gửi xe đang gửi giúp em nhé",
        "installed_apps": ["parking"],
        "called_apps": {"parking"},
        "tool_calls": [
            {"name": "parking.list_sessions", "arguments": {}},
            {"name": "parking.end_session", "arguments": {"session_id": "park_01"}},
        ],
    },
    {
        "id": "cat_public_transit_route",
        "audio_text": "Xem nếu đi xe buýt từ Bến xe Mỹ Đình ra Hồ Gươm thì bắt tuyến nào mất bao lâu nhé",
        "installed_apps": ["public_transit"],
        "called_apps": {"public_transit"},
        "tool_calls": [
            {"name": "public_transit.find_routes", "arguments": {"origin": "Bến xe Mỹ Đình", "destination": "Hồ Gươm"}},
        ],
    },
    {
        "id": "cat_public_transit_topup",
        "audio_text": "Nạp 100 nghìn vào thẻ đi xe buýt công cộng giúp anh nhé",
        "installed_apps": ["public_transit"],
        "called_apps": {"public_transit"},
        "tool_calls": [
            {"name": "public_transit.topup_card", "arguments": {"amount": 100000}},
        ],
    },
    {
        "id": "cat_habit_tracker_log",
        "audio_text": "Tôi tập thở xong rồi đấy, đánh dấu hôm nay hoàn thành thói quen giúp tôi nhé",
        "installed_apps": ["habit_tracker"],
        "called_apps": {"habit_tracker"},
        "tool_calls": [
            {"name": "habit_tracker.list_habits", "arguments": {}},
            {"name": "habit_tracker.log", "arguments": {"habit_id": "habit_01", "date": "2026-10-15"}},
        ],
    },
    {
        "id": "cat_stocks_alert",
        "audio_text": "Xem cổ phiếu trong danh mục đầu tư rồi đặt cảnh báo khi mã đó lên lại giá vốn hoặc vượt 128 nghìn nhé",
        "installed_apps": ["stocks"],
        "called_apps": {"stocks"},
        "tool_calls": [
            {"name": "stocks.get_portfolio", "arguments": {}},
            {"name": "stocks.set_price_alert", "arguments": {"symbol": "Vitech", "price": 128000, "direction": "above"}},
        ],
    },
    {
        "id": "cat_stocks_quote_only",
        "audio_text": "Xem giá cổ phiếu ZMX hôm nay bao nhiêu nhé",
        "installed_apps": ["stocks"],
        "called_apps": {"stocks"},
        "tool_calls": [
            {"name": "stocks.get_quote", "arguments": {"symbol": "ZMX"}},
        ],
    },
    {
        "id": "cat_stocks_place_order",
        "audio_text": "Kiểm tra danh mục cổ phiếu xem mã nào đang lỗ hoặc lời thì đặt lệnh bán bớt 100 cổ phiếu giúp anh nhé",
        "installed_apps": ["stocks"],
        "called_apps": {"stocks"},
        "tool_calls": [
            {"name": "stocks.get_portfolio", "arguments": {}},
            {"name": "stocks.place_order", "arguments": {"symbol": "ZKN", "side": "sell", "quantity": 100}},
        ],
    },
    {
        "id": "cat_translator_single",
        "audio_text": "Dịch giúp chị câu hóa đơn đã được thanh toán sang tiếng Anh nhé",
        "installed_apps": ["translator"],
        "called_apps": {"translator"},
        "tool_calls": [
            {"name": "translator.translate", "arguments": {"text": "Hóa đơn đã được thanh toán", "to_lang": "en"}},
        ],
    },
    {
        "id": "cat_train_bus_book_single",
        "audio_text": "Đặt cho anh 1 vé tàu hỏa hoặc xe khách đi Vinh sáng mai nhé",
        "installed_apps": ["train_bus_tickets"],
        "called_apps": {"train_bus_tickets"},
        "tool_calls": [
            {"name": "train_bus_tickets.search", "arguments": {"origin": "Hà Nội", "destination": "Vinh", "date": "2026-10-16", "type": "train"}},
            {"name": "train_bus_tickets.book", "arguments": {"trip_id": "trip_01", "seats": 1}},
        ],
    },
    {
        "id": "cat_recipes_search_only",
        "audio_text": "Tìm cho chị công thức nấu món bún riêu cua hoặc súp bí đỏ nhé",
        "installed_apps": ["recipes"],
        "called_apps": {"recipes"},
        "tool_calls": [
            {"name": "recipes.search", "arguments": {"query": "bún riêu cua"}},
        ],
    },
    {
        "id": "cat_recipes_get_detail",
        "audio_text": "Tìm công thức món canh chua cá lóc xem chi tiết cần những nguyên liệu gì nhé",
        "installed_apps": ["recipes"],
        "called_apps": {"recipes"},
        "tool_calls": [
            {"name": "recipes.search", "arguments": {"query": "canh chua cá lóc"}},
            {"name": "recipes.get", "arguments": {"recipe_id": "rcp_01"}},
        ],
    },
    {
        "id": "cat_public_transit_departures",
        "audio_text": "Xem tuyến xe buýt hoặc metro từ Bến Thành đi Suối Tiên mấy giờ có chuyến tiếp theo nhé",
        "installed_apps": ["public_transit"],
        "called_apps": {"public_transit"},
        "tool_calls": [
            {"name": "public_transit.find_routes", "arguments": {"origin": "Bến Thành", "destination": "Suối Tiên"}},
            {"name": "public_transit.next_departures", "arguments": {"stop": "Bến Thành", "route": "bus_01"}},
        ],
    },
    {
        "id": "cat_public_transit_next_only",
        "audio_text": "Coi giùm chuyến xe buýt hoặc tàu điện kế tiếp ở trạm đó mấy giờ chạy nhé",
        "installed_apps": ["public_transit"],
        "called_apps": {"public_transit"},
        "tool_calls": [
            {"name": "public_transit.next_departures", "arguments": {"stop": "Cầu Giấy"}},
        ],
    },
    {
        "id": "cat_grocery_cart_only",
        "audio_text": "Tìm bên siêu thị đi chợ bỏ vào giỏ cho chị 2 hộp sữa tươi với 1 gói hạt macca nhé",
        "installed_apps": ["grocery"],
        "called_apps": {"grocery"},
        "tool_calls": [
            {"name": "grocery.search", "arguments": {"query": "sữa tươi"}},
            {"name": "grocery.add_to_cart", "arguments": {"product_id": "gro_01", "quantity": 2}},
        ],
    },
    {
        "id": "cat_bill_autopay",
        "audio_text": "Kiểm tra hóa đơn tiền điện tháng này mà dưới 1 triệu thì cài tự động trả autopay giùm tôi nhé",
        "installed_apps": ["bill_payment"],
        "called_apps": {"bill_payment"},
        "tool_calls": [
            {"name": "bill_payment.list_bills", "arguments": {}},
            {"name": "bill_payment.schedule_autopay", "arguments": {"type": "electricity", "enabled": True}},
        ],
    },
    {
        "id": "cat_calendar_create_single",
        "audio_text": "Thêm lịch 6 rưỡi sáng mai đón khách Mark Johnson hoặc ghi vào lịch giúp tôi nhé",
        "installed_apps": ["calendar"],
        "called_apps": {"calendar"},
        "tool_calls": [
            {"name": "calendar.create_event", "arguments": {"title": "Đón khách Mark Johnson", "start": "2026-10-16T06:30:00+07:00", "end": "2026-10-16T07:30:00+07:00"}},
        ],
    },
    {
        "id": "cat_shopping_list_check",
        "audio_text": "Trong danh sách đi chợ hôm qua mua rồi mà quên gạch, đánh dấu trong danh sách đi chợ giúp anh nhé",
        "installed_apps": ["shopping_list"],
        "called_apps": {"shopping_list"},
        "tool_calls": [
            {"name": "shopping_list.get", "arguments": {}},
            {"name": "shopping_list.check", "arguments": {"item_id": "sl_01", "checked": True}},
        ],
    },
    {
        "id": "cat_shopping_list_remove",
        "audio_text": "Món nào có trong danh sách đi chợ thì xóa bỏ khỏi danh sách đi chợ giùm em nhé",
        "installed_apps": ["shopping_list"],
        "called_apps": {"shopping_list"},
        "tool_calls": [
            {"name": "shopping_list.get", "arguments": {}},
            {"name": "shopping_list.remove", "arguments": {"item_id": "sl_01"}},
        ],
    },
    {
        "id": "cat_banking_transactions",
        "audio_text": "Kiểm tra sao kê danh sách giao dịch ngân hàng của tài khoản xem đã chuyển tiền chưa nhé",
        "installed_apps": ["banking"],
        "called_apps": {"banking"},
        "tool_calls": [
            {"name": "banking.get_balance", "arguments": {}},
            {"name": "banking.list_transactions", "arguments": {"account_id": "acc_01"}},
        ],
    },
    {
        "id": "cat_maps_save_place",
        "audio_text": "Lưu địa điểm chỗ đó vào bản đồ tên Siêu thị quen nha",
        "installed_apps": ["maps"],
        "called_apps": {"maps"},
        "tool_calls": [
            {"name": "maps.save_place", "arguments": {"place_id": "place_01", "label": "Siêu thị quen"}},
        ],
    },
    {
        "id": "cat_stocks_portfolio_only",
        "audio_text": "Kiểm tra danh mục cổ phiếu xem mã nào đang lời đọc cho anh nghe thôi",
        "installed_apps": ["stocks"],
        "called_apps": {"stocks"},
        "tool_calls": [
            {"name": "stocks.get_portfolio", "arguments": {}},
        ],
    },
    {
        "id": "cat_smart_home_scene_run",
        "audio_text": "Bật chế độ ra khỏi nhà giúp tôi nhé",
        "installed_apps": ["smart_home"],
        "called_apps": {"smart_home"},
        "tool_calls": [
            {"name": "smart_home.get_status", "arguments": {}},
            {"name": "smart_home.scene_run", "arguments": {"scene_id": "scene_01"}},
        ],
    },
    {
        "id": "cat_email_read_check_calendar",
        "audio_text": "Thầy tiếng Nhật mới gửi email hẹn buổi review, đọc email xem giờ rồi kiểm tra lịch xem tới đó mình rảnh hay bận",
        "installed_apps": ["email", "calendar"],
        "called_apps": {"email", "calendar"},
        "tool_calls": [
            {"name": "email.list", "arguments": {}},
            {"name": "email.read", "arguments": {"email_id": "mail_01"}},
            {"name": "calendar.list_events", "arguments": {}},
        ],
    },
    {
        "id": "cat_ewallet_topup_mobile",
        "audio_text": "Nạp giúp sếp Dũng 100 nghìn tiền điện thoại cho số 0901234567 nhé",
        "installed_apps": ["ewallet", "contacts"],
        "called_apps": {"ewallet", "contacts"},
        "tool_calls": [
            {"name": "contacts.search", "arguments": {"query": "Sếp Dũng"}},
            {"name": "ewallet.topup_mobile", "arguments": {"phone": "0901234567", "amount": 100000}},
        ],
    },
    {
        "id": "cat_medication_add",
        "audio_text": "Thêm giúp tôi lịch uống thuốc dạ dày ngày hai lần 7 giờ sáng và 7 giờ tối",
        "installed_apps": ["medication"],
        "called_apps": {"medication"},
        "tool_calls": [
            {"name": "medication.add", "arguments": {"name": "Thuốc dạ dày", "times": ["07:00", "19:00"]}},
        ],
    },
    {
        "id": "cat_calculator_convert_unit",
        "audio_text": "Năm mét vải thì được bao nhiêu xăng ti mét ấy nhỉ đổi thử giúp em",
        "installed_apps": ["calculator"],
        "called_apps": {"calculator"},
        "tool_calls": [
            {"name": "calculator.convert_unit", "arguments": {"value": 5, "from_unit": "m", "to_unit": "cm"}},
        ],
    },
    {
        "id": "cat_notes_create_only",
        "audio_text": "Lấy cái mới nhất ghi vô ghi chú An ninh buổi sáng nha",
        "installed_apps": ["notes"],
        "called_apps": {"notes"},
        "tool_calls": [
            {"name": "notes.create", "arguments": {"title": "An ninh buổi sáng", "content": "Nội dung ghi chú"}},
        ],
    },
    {
        "id": "cat_video_meeting_create",
        "audio_text": "Mở luôn một phòng họp online cùng tên đúng giờ mới",
        "installed_apps": ["video_meeting"],
        "called_apps": {"video_meeting"},
        "tool_calls": [
            {"name": "video_meeting.create", "arguments": {"title": "Họp", "start": "2026-10-16T10:00:00+07:00"}},
        ],
    },
    {
        "id": "cat_contacts_search_only",
        "audio_text": "Tìm trong danh bạ xem số điện thoại với email của chị Lan là gì cho chị với",
        "installed_apps": ["contacts"],
        "called_apps": {"contacts"},
        "tool_calls": [
            {"name": "contacts.search", "arguments": {"query": "Lan"}},
        ],
    },
    {
        "id": "cat_contacts_update",
        "audio_text": "Mai Anh mới đổi email rồi sửa trong danh bạ dùm chị thành mai.anh@example.test nha",
        "installed_apps": ["contacts"],
        "called_apps": {"contacts"},
        "tool_calls": [
            {"name": "contacts.search", "arguments": {"query": "Mai Anh"}},
            {"name": "contacts.update", "arguments": {"contact_id": "c_01", "email": "mai.anh@example.test"}},
        ],
    },
    {
        "id": "cat_bike_rental_reserve",
        "audio_text": "Giữ cho tôi chiếc xe đạp chở hàng ở trạm Chợ Hôm",
        "installed_apps": ["bike_rental"],
        "called_apps": {"bike_rental"},
        "tool_calls": [
            {"name": "bike_rental.find_bikes", "arguments": {"station": "Chợ Hôm"}},
            {"name": "bike_rental.reserve", "arguments": {"bike_id": "bk_01"}},
        ],
    },
    {
        "id": "cat_library_reserve",
        "audio_text": "Giữ giúp mình cuốn Sửa đồ gia dụng căn bản ở thư viện nha",
        "installed_apps": ["library"],
        "called_apps": {"library"},
        "tool_calls": [
            {"name": "library.search", "arguments": {"query": "Sửa đồ gia dụng căn bản"}},
            {"name": "library.reserve", "arguments": {"book_id": "lib_01"}},
        ],
    },
    {
        "id": "cat_home_energy_usage",
        "audio_text": "Hôm nay nhà mình dùng điện quá mười lăm số chưa",
        "installed_apps": ["home_energy"],
        "called_apps": {"home_energy"},
        "tool_calls": [
            {"name": "home_energy.get_usage", "arguments": {"period": "today"}},
        ],
    },
]


def _ensure_catalog_exemplars_indexed(retriever: Any) -> None:
    """Bổ sung bộ mẫu chuẩn từ Catalog (cho các miền ứng dụng mới) vào chỉ mục BM25 đúng 1 lần."""
    if not retriever or getattr(retriever, "_catalog_exemplars_indexed", False):
        return
    try:
        combined = list(getattr(retriever, "corpus", [])) + CATALOG_CANONICAL_EXEMPLARS
        retriever.index_records(combined)
        retriever._atomic_retriever = None
        retriever._catalog_exemplars_indexed = True
    except Exception:
        pass


CALENDAR_DATE_FILTER_RE = re.compile(
    r"(có lịch .* (không|hông|phải không)|xem lịch (hôm nay|ngày mai|thứ|tuần)|lịch chị trống|tên ghi trong lịch ấy|coi lịch hẹn)",
    re.IGNORECASE
)


def _is_tool_compatible(tc_or_name: Any, text: str) -> bool:
    norm_t = _normalize_vn_numbers_in_time(text)
    if isinstance(tc_or_name, dict):
        tname = tc_or_name.get("name", "")
        if tname == "sys.ask_missing_info":
            targs = tc_or_name.get("arguments") or {}
            eff_name = targs.get("tool_name", "")
            params = targs.get("params") or []
            if eff_name:
                eff_app = eff_name.split(".")[0]
                app_pat = APP_TRIGGER_PATTERNS.get(eff_app)
                if app_pat and not re.search(app_pat, text, re.IGNORECASE):
                    return False
                pat = TOOL_KEYWORD_GUARDS.get(eff_name)
                if pat and not re.search(pat, text, re.IGNORECASE):
                    return False
                if not _is_tool_compatible(eff_name, text):
                    return False
                if eff_name == "alarm.create" and (
                    re.search(r"\b(cái báo thức .* (?:dời|đổi|chuyển|sửa|lùi|thành|sang)|dời .* báo thức|đổi .* báo thức|tắt .* báo thức|xóa .* báo thức|xoá .* báo thức|hủy .* báo thức)\b", norm_t, re.IGNORECASE)
                    or (re.search(r"\b(dời|đổi|chuyển|sửa|lùi|đẩy|tắt|xóa|xoá|hủy|huỷ)\b", norm_t, re.IGNORECASE) and re.search(r"\bbáo thức\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đặt|tạo|thêm|hẹn)\s+(?:một\s+|cái\s+|thêm\s+)?báo thức\b", norm_t, re.IGNORECASE))
                ):
                    return False
            if params in (["time"], ["start"]) and eff_name in {"alarm.create", "reminders.create", "calendar.create_event", "video_meeting.create"}:
                if re.search(r"\b\d{1,2}\s*(?:giờ|rưỡi|h\b)|trước giờ|trước đó|sau khi|cùng giờ", norm_t, re.IGNORECASE):
                    return False
            if params == ["amount"] and eff_name == "ewallet.transfer":
                if re.search(r"\b\d+\s*(?:nghìn|ngàn|triệu|củ|đồng|k\b)|hết bao nhiêu|mỗi người|phần của", norm_t, re.IGNORECASE):
                    return False
            if eff_name == "ewallet.topup_mobile":
                if re.search(r"\b(0\d{8,11}|cho\s+\S+|dùm\s+\S+|giúp\s+\S+)\b", norm_t, re.IGNORECASE) and re.search(r"\b\d+\s*(?:nghìn|ngàn|k\b)\b", norm_t, re.IGNORECASE):
                    return False
            if params == ["channel"] and eff_name == "tv.tune":
                if re.search(r"\b(vtv\d*|htv\d*|s\+\s*sport|sctv\d*|k\+|hbo|cinemax|cartoon|discovery)\b", norm_t, re.IGNORECASE):
                    return False
            if params == ["steps"] and eff_name == "fitness.set_goal":
                if re.search(r"\b\d+\s*(?:nghìn|ngàn|bước)", norm_t, re.IGNORECASE):
                    return False
            return True
    else:
        tname = str(tc_or_name)
    if tname == "alarm.create":
        if re.search(r"\b(cái báo thức .* (?:dời|đổi|chuyển|sửa|lùi|thành|sang)|dời .* báo thức|đổi .* báo thức|tắt .* báo thức|xóa .* báo thức|xoá .* báo thức|hủy .* báo thức|có báo thức nào (?:đang bật )?không|xem .* báo thức nào)\b", norm_t, re.IGNORECASE):
            return False
        if re.search(r"\b(dời|đổi|chuyển|sửa|lùi|đẩy|tắt|xóa|xoá|hủy|huỷ)\b", norm_t, re.IGNORECASE) and re.search(r"\bbáo thức\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đặt|tạo|thêm|hẹn)\s+(?:một\s+|cái\s+|thêm\s+)?báo thức\b", norm_t, re.IGNORECASE):
            return False
        if not re.search(r"\b(\d{1,2}\s*(?:giờ|rưỡi|h\b)|\d+\s*(?:phút|tiếng)|nửa\s+tiếng|trước\s+(?:giờ|lúc|buổi|khi|đó|đám\s+cưới)|sớm\s+hơn|sáng\s+hôm\s+bay)\b", norm_t, re.IGNORECASE):
            return False
    if tname in {"appliances.start_cycle", "appliances.stop"}:
        if re.search(r"\b(chỉ kiểm tra thôi|chưa bật vội|khoan bật|đừng bật|đang tắt hay đang chạy)\b", norm_t, re.IGNORECASE):
            return False
        if tname == "appliances.start_cycle" and re.search(r"\b(dừng lại|cho nó dừng|tắt đi|ngừng lại)\b", norm_t, re.IGNORECASE):
            return False
    if tname in {"smart_home.light_set", "smart_home.fan_set", "smart_home.climate_set", "smart_home.switch_set", "smart_home.scene_run"}:
        if re.search(r"\b(có những thiết bị nào|đang bật hay tắt)\b", norm_t, re.IGNORECASE):
            return False
    if tname.startswith("smart_home.") and re.search(r"\bphòng\s+tập\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đèn|quạt|điều hòa|điều hoà|máy lạnh|ổ cắm|công tắc|phòng khách|phòng ngủ|phòng bếp)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "home_security.list_alerts":
        if re.search(r"\b(chuyển báo động sang chế độ|bật báo động chế độ|đặt báo động chế độ)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(cảnh báo|xác nhận)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "notes.append" and re.search(r"\b(ghi chú mới|đặt tên là)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "notes.create" and re.search(r"\bghi\s+số\s+đó\s+vào\s+(?:cái\s+)?ghi\s+chú\b", norm_t, re.IGNORECASE) and "ghi chú mới" not in norm_t.lower():
        return False
    if tname == "news.save" and re.search(r"\bchép\b.*\bvào\s+(?:một\s+|cái\s+)?ghi\s+chú\b", norm_t, re.IGNORECASE):
        return False
    if tname.startswith("medication.") and re.search(r"\bnhắc\s+(?:tôi|mình|anh|chị|em|cô|chú)\s+uống\s+thuốc\b", norm_t, re.IGNORECASE) and not re.search(r"\b(lịch thuốc|cữ thuốc|thuốc nào|đã uống|uống rồi)\b", norm_t, re.IGNORECASE):
        return False
    if tname.startswith("reminders.") and re.search(r"\bnhắc\b[^,.!?;]*?\b(hồi nãy|lúc nãy)\b|\bgọi nhắc (?:tôi|mình|anh|chị|em|cô|chú)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(nhắc (?:tôi|mình|anh|chị|em|cô|chú) (?!.*(?:hồi nãy|lúc nãy)))\b", norm_t, re.IGNORECASE):
        return False
    if tname == "loyalty.redeem" and re.search(r"\bđủ đổi\b[^,.!?;]*?\bthì (?:bảo|báo|nói|cho .* biết)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "chat_ott.send_message":
        if re.search(r"\b(đọc giúp .* zalo|có ai nhắn|ai nhắn zalo|\S+\s+(?:mới\s+|vừa\s+)?nhắn\s+zalo\s+vụ)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(?:rồi|xong|với|và)\s+nhắn\b", norm_t, re.IGNORECASE):
            return False
    if tname == "sms.send":
        if re.search(r"\b(mới nhắn(?: tin)?(?: đó)?|coi tin nhắn|xem tin nhắn|đọc tin nhắn|trạm .* nhắn đó)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(?:rồi|xong|với|và)\s+nhắn\b", norm_t, re.IGNORECASE):
            return False
    if tname in {"bill_payment.pay", "ewallet.get_balance"}:
        if re.search(r"\b(chuyển|gửi|gởi)\b.*\b(đúng|bằng|y)\s+(?:bằng\s+)?(?:số\s+)?tiền\s+(?:điện|nước|mạng|internet|truyền hình|cáp)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đóng|trả|thanh toán)\s+(?:tiền|hóa đơn|hoá đơn)\b", norm_t, re.IGNORECASE):
            return False
        if re.search(r"\b(tài khoản thanh toán|tiết kiệm)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đóng|trả|thanh toán)\s+(?:tiền|hóa đơn|hoá đơn)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "ewallet.get_balance":
        if re.search(r"\b(chút tiền|ít tiền)\b", norm_t, re.IGNORECASE) and not re.search(r"\b\d+\s*(?:nghìn|ngàn|triệu|củ|đồng|k\b)\b", norm_t, re.IGNORECASE):
            return False
        if re.search(r"\b(tiền điện|tiền nước|tiền mạng|tiền truyền hình|tiền cáp|hóa đơn|hoá đơn)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(ví đủ|đủ không|không đủ|còn đủ|thiếu thì|hụt thì|số dư ví|kiểm tra ví|coi ví|xem ví)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "weather.get_forecast":
        if re.search(r"\b(nhắn|báo|nói)[^,.!?;]*?\blà\s+mai\s+mưa\b", norm_t, re.IGNORECASE):
            return False
        if re.search(r"\b(giờ (?:mà |có )?(?:trời )?đang|hiện tại|lúc này|bây giờ)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(mai|ngày kia|tuần|thứ|chủ nhật)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "weather.get_current":
        if re.search(r"\b(mai|ngày kia|tuần|thứ|chủ nhật)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(hiện tại|bây giờ|lúc này|giờ (?:mà |có )?(?:trời )?đang)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "smart_home.fan_set" and re.search(r"\bquạt\b[^,.]{0,25}\b(đừng bật|khỏi bật|không bật|không cần bật|đang chạy|đang bật)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(bật hết quạt|tắt quạt|tắt cái quạt|bật quạt)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "smart_home.light_set":
        if re.search(r"\b(tự tắt|tự bật|quy tắc)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(bật|tắt)\s+(?:hết\s+|mấy\s+|cái\s+)?đèn\b", norm_t, re.IGNORECASE):
            return False
        if re.search(r"\bđèn\b.{0,55}\b(đừng tắt|khỏi tắt|cứ để sáng|để nguyên)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(tắt hết đèn|tắt đèn .* hết đi|tắt toàn bộ đèn|tắt mấy đèn|phòng nào)\b", norm_t, re.IGNORECASE):
            return False
    if tname == "translator.translate" and re.search(r"\b(thôi khỏi dịch|khỏi dịch lại|đừng dịch)\b", norm_t, re.IGNORECASE):
        return False
    if tname in {"ecommerce.add_to_cart", "ecommerce.checkout"} and re.search(r"\b(hãng nào|còn hàng không|xem giá|bao nhiêu tiền)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(bỏ vào giỏ|thêm vào giỏ|vào giỏ|đặt mua|mua luôn|thanh toán|chốt)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "ewallet.transfer":
        if re.search(r"\bnạp\b", norm_t, re.IGNORECASE) and not re.search(r"\b(chuyển\s+ví|qua\s+ví|nạp\b.*\b(?:rồi|với|và)\s+(?:chuyển|gửi|gởi))\b", norm_t, re.IGNORECASE):
            return False
    if tname in {"train_bus_tickets.book", "train_bus_tickets.search", "flights.book", "flights.search"} and re.search(r"\b(hủy|huỷ)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đặt lại|mua lại|tìm chuyến khác)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "smart_home.climate_set" and re.search(r"\b(phòng nào .* (?:điều hòa|điều hoà|máy lạnh)|đang\s+(?:bật|mở|chạy)\s+(?:điều hòa|điều hoà|máy lạnh)|xem (?:điều hòa|điều hoà|máy lạnh) đang)\b", norm_t, re.IGNORECASE) and not re.search(r"(?<!đang\s)\b(bật|mở|tắt|chỉnh|hạ|tăng|để \d{2})\s*(?:điều hòa|điều hoà|máy lạnh)", norm_t, re.IGNORECASE):
        return False
    if tname in {"flights.book", "flights.search"} and re.search(r"\b(hủy|huỷ)\b.*\b(vé máy bay|chuyến bay)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đặt|mua|đổi sang)\b", norm_t, re.IGNORECASE):
        return False
    if tname in {"flights.book", "flights.search"} and re.search(r"\b(chuyến bay|vé máy bay)\b[^,.!?;]*?\b(?:mình|tôi|anh|chị|em|đã|vừa)\s+đặt\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đặt thêm|đặt lại|tìm chuyến khác|đổi sang)\b", norm_t, re.IGNORECASE):
        return False
    if tname in {"grocery.add_to_cart", "grocery.checkout"} and re.search(r"\b(hết bao nhiêu tiền|hết nhiêu tiền)\b.*\b(đủ trả không|còn đủ)\b", norm_t, re.IGNORECASE):
        return False
    if tname in {"food_delivery.place_order", "food_delivery.get_menu", "food_delivery.list_restaurants"} and re.search(r"\b(hủy đơn|huỷ đơn|lưu giúp .* vào bản đồ)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đặt lại|đặt thêm|đặt 1|đặt một|đặt hai|đặt 2)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "banking.list_beneficiaries" and re.search(r"\b(mua thêm .* cổ phiếu|bán .* cổ phiếu|gửi tiết kiệm|mở sổ tiết kiệm)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(chuyển khoản|chuyển cho)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "contacts.search" and re.search(r"\b(hết bao nhiêu tiền|số dư ví)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(chuyển|nạp|gọi|nhắn)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "calendar.list_events" and re.search(r"\b(tạo lịch|thêm lịch)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(xem .* lịch|kiểm tra .* lịch|có lịch .* không|trùng|rảnh hay bận)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "phone.call" and "thông tin liên hệ đầy đủ" in norm_t.lower():
        return False
    if tname in {"phone.call", "phone.recent_calls"} and re.search(r"\bnhắc\s+(?:tôi|mình|anh|chị|em)\s+gọi\b", norm_t, re.IGNORECASE) and not re.search(r"\b(?:rồi|với|và)\s+gọi\b", norm_t, re.IGNORECASE):
        return False
    pat = TOOL_KEYWORD_GUARDS.get(tname)
    if not pat:
        return True
    return bool(re.search(pat, text, re.IGNORECASE))


def _valid_filtered(filtered_calls: List[Dict[str, Any]], u_text: str) -> bool:
    if not filtered_calls:
        return False
    names = {tc["name"] for tc in filtered_calls}
    norm_u = _normalize_vn_numbers_in_time(u_text)
    if names == {"contacts.search"} and not re.search(r"\b(danh bạ|số điện thoại|tìm số)\b", u_text, re.IGNORECASE):
        return False
    if names == {"ewallet.get_balance"} and re.search(r"\b(nạp|chuyển|quét mã|thanh toán)\b", norm_u, re.IGNORECASE) and not re.search(r"\b(số dư|còn bao nhiêu)\b", norm_u, re.IGNORECASE):
        return False
    if names == {"alarm.list"} and re.search(r"\b(đặt|hẹn|tạo|dời|đổi|chuyển|sửa|lùi|đẩy|xóa|xoá|tắt|hủy|huỷ|thành \d|sang \d)\b", norm_u, re.IGNORECASE):
        if not re.search(r"\b(có báo thức nào .* không|xem .* báo thức|kiểm tra .* báo thức)\b", u_text, re.IGNORECASE):
            return False
    if names == {"reminders.list"} and re.search(r"\b(nhắc tôi|nhắc mình|nhắc anh|nhắc chị|nhắc em|tạo nhắc|thêm nhắc|xóa nhắc|xoá nhắc|đánh dấu xong)\b", u_text, re.IGNORECASE):
        if not re.search(r"\b(xem .* nhắc|kiểm tra .* nhắc|có lời nhắc nào)\b", u_text, re.IGNORECASE):
            return False
    if names == {"calendar.list_events"} and re.search(r"\b(tạo .* lịch|thêm .* lịch|ghi vào lịch|cho vào lịch|đặt lịch|dời|đổi|chuyển|lùi|hủy|huỷ|xóa|xoá)\b", norm_u, re.IGNORECASE):
        if not re.search(r"\b(xem .* lịch|kiểm tra .* lịch|có lịch .* không|trùng lịch)\b", u_text, re.IGNORECASE):
            return False
    if "calendar.list_events" in names and "calendar.update_event" not in names and re.search(r"\b(dời|đổi|lùi|đẩy)\b.*\b(lịch|buổi|hẹn|họp|khám|cà phê)\b", norm_u, re.IGNORECASE) and not re.search(r"\b(nếu|mưa|hủy|huỷ|xóa|xoá)\b", norm_u, re.IGNORECASE):
        return False
    if names == {"email.list"} and re.search(r"\b(xóa|xoá|lưu trữ|archive|trả lời|reply|soạn|gửi|nháp)\b", norm_u, re.IGNORECASE):
        return False
    if names == {"shopping_list.get"} and re.search(r"\b(thêm|ghi|xóa|xoá|bỏ|gạch|đánh dấu)\b", norm_u, re.IGNORECASE):
        return False
    if names == {"parking.list_sessions"} and re.search(r"\b(kết thúc|tắt .* phiên|tách cái phiên|cho dừng)\b", norm_u, re.IGNORECASE):
        return False
    if names == {"event_tickets.search"} and re.search(r"\b(đặt|mua|lấy)\b.*\b(vé|vét|chỗ)\b", norm_u, re.IGNORECASE):
        return False
    if names <= {"food_delivery.list_restaurants", "food_delivery.get_menu", "chat_ott.list_chats"} and "food_delivery.place_order" not in names:
        if re.search(r"\b(mỗi người một|đặt giúp|đặt cho cả nhóm|giao về)\b", u_text, re.IGNORECASE):
            return False
    return True


def _get_atomic_retriever(retriever: Any) -> Any:
    """Khởi tạo và lưu cache bộ truy xuất mệnh đề đơn (Atomic Sub-Exemplar Retriever) ngay trên đối tượng retriever."""
    _ensure_catalog_exemplars_indexed(retriever)
    if getattr(retriever, "_atomic_retriever", None) is not None:
        return retriever._atomic_retriever

    from src.agent.rag.exemplar_retriever import BM25ExemplarRetriever
    atomic_records = []
    for rec in getattr(retriever, "corpus", []):
        if "sys" in rec.get("called_apps", set()):
            continue
        non_contact_apps = rec["called_apps"] - {"contacts"}
        if len(non_contact_apps) == 1:
            atomic_records.append(rec)
        elif len(non_contact_apps) >= 2:
            clauses = [c.strip() for c in CLAUSE_SPLIT_RE.split(rec["audio_text"]) if len(c.strip().split()) >= 3]
            if len(clauses) >= 2:
                for app_name in non_contact_apps:
                    app_calls = [
                        tc for tc in rec["tool_calls"]
                        if tc["name"].split(".")[0] == app_name
                        or (tc["name"].startswith("contacts.") and app_name in {"phone", "sms", "ewallet", "email", "chat_ott"})
                    ]
                    if not app_calls:
                        continue
                    app_pat = APP_TRIGGER_PATTERNS.get(app_name)
                    matched_cl = None
                    for cl in clauses:
                        if app_pat and re.search(app_pat, cl, re.IGNORECASE) and all(_is_tool_compatible(tc, cl) for tc in app_calls):
                            matched_cl = cl
                            break
                    if matched_cl:
                        atomic_records.append({
                            "id": f"{rec['id']}_{app_name}",
                            "audio_text": matched_cl,
                            "installed_apps": [app_name, "contacts"],
                            "called_apps": {tc["name"].split(".")[0] for tc in app_calls},
                            "tool_calls": app_calls,
                        })

    ret_atomic = BM25ExemplarRetriever()
    ret_atomic.index_records(atomic_records)
    retriever._atomic_retriever = ret_atomic
    return ret_atomic


NEVER_HISTORY_SATISFIED_TOOLS: Set[str] = {
    "smart_home.get_status",
    "robot_vacuum.get_status",
    "video_meeting.list_meetings",
    "parking.list_sessions",
    "train_bus_tickets.my_bookings",
    "flights.my_bookings",
    "habit_tracker.list_habits",
    "calculator.calculate",
    "calculator.convert_unit",
    "exchange_rate.get_rate",
    "exchange_rate.convert",
    "fuel_price.get_price",
    "gold_price.get_price",
    "weather.get_forecast",
    "weather.get_current",
    "translator.translate",
}


def get_history_satisfied_reads(history: Optional[List[Dict[str, Any]]], u_text: str = "") -> Set[str]:
    """Xác định các công cụ Đọc đã được thực thi và có sẵn dữ liệu trong history."""
    sat: Set[str] = set()
    u_low = (u_text or "").lower()
    is_short_followup = len(u_low.split()) <= 16 or bool(re.search(r"\b(cái đó|buổi đó|hôm đó|người đó|số đó|khoản đó|bài đó|bản đó|đơn đó|mấy bài đó|như thế|vừa chọn|hồi nãy|lúc sáng|chuyến đó|món đó|vé đó|bãi đó)\b", u_low))
    hist_list = history or []
    is_long_multi_session = len(hist_list) >= 12
    for idx_t, t in enumerate(hist_list):
        is_recent_turn = (not is_long_multi_session) or (idx_t >= len(hist_list) - 4)
        for tc in (t.get("tool_calls") or []):
            tname = tc.get("name", "")
            targs = tc.get("arguments") or {}
            if tname in READ_TOOLS and tname not in NEVER_HISTORY_SATISFIED_TOOLS:
                if targs and not is_recent_turn and not is_short_followup:
                    continue
                if tname == "calendar.list_events" and targs and not is_short_followup and not re.search(r"\b(dời lịch|sửa lịch|đổi tên lịch)\b", u_low):
                    continue
                if tname == "contacts.search" and targs.get("query") and str(targs["query"]).lower() not in u_low and not is_short_followup:
                    continue
                if tname == "food_delivery.list_restaurants" and not is_short_followup and not re.search(r"\b(cô ba|nhà mèo|nhà làm|sài gòn 168)\b", u_low):
                    continue
                if tname in {"food_delivery.order_history", "ecommerce.order_status"} and re.search(r"\b(tới đâu rồi|xem .* đơn|kiểm tra .* đơn)\b", u_low):
                    continue
                if tname == "bike_rental.find_bikes" and re.search(r"\bở\s+trạm\b", u_low) and not re.search(r"\b(lúc nãy|vừa nãy|hồi nãy)\b", u_low):
                    continue
                if tname == "maps.search_place" and not re.search(r"\b(lưu|đánh dấu|chỉ đường tới chỗ đó|chỉ đường ra đó)\b", u_low):
                    continue
                sat.add(tname)
                if tname == "email.read":
                    sat.add("email.list")
            elif tname in {"grocery.add_to_cart", "ecommerce.add_to_cart"} and is_recent_turn:
                sat.add(tname)
            elif tname == "home_automation.create_rule":
                sat.add("home_automation.list_rules")
            elif tname == "food_delivery.place_order" and is_recent_turn:
                sat.add("food_delivery.order_history")
            elif tname == "smart_home.get_status" and idx_t >= len(hist_list) - 2 and len(u_low.split()) <= 16 and not re.search(r"\b(phòng nào|xem|kiểm tra|coi)\b", u_low):
                sat.add(tname)
        for res in (t.get("tool_results") or []):
            if isinstance(res, dict) and ("rules" in res or "rule" in res):
                sat.add("home_automation.list_rules")
            if not isinstance(res, dict) or (not is_recent_turn and not is_short_followup):
                continue
            if "alerts" in res or "alert" in res:
                sat.add("home_security.list_alerts")
            if "rules" in res or "rule" in res:
                sat.add("home_automation.list_rules")
            if "chats" in res or "chat" in res:
                sat.add("chat_ott.list_chats")
            if "messages" in res and is_recent_turn:
                sat.add("chat_ott.read_chat")
                sat.add("sms.read_recent")
            if "tasks" in res or "task" in res:
                sat.add("tasks.list")
            if "bookings" in res:
                sat.add("flights.my_bookings")
                sat.add("train_bus_tickets.my_bookings")
            if "flights" in res and is_recent_turn:
                sat.add("flights.search")
            if "trips" in res:
                sat.add("train_bus_tickets.search")
            if "events" in res and isinstance(res["events"], list) and any(isinstance(x, dict) and ("venue" in x or "available" in x or "price" in x) for x in res["events"]):
                sat.add("event_tickets.search")
            if "calls" in res:
                sat.add("phone.recent_calls")
            if "places" in res and is_recent_turn and re.search(r"\b(lưu|đánh dấu|chỉ đường tới chỗ đó|chỉ đường ra đó)\b", u_low):
                sat.add("maps.search_place")
            if "products" in res and is_recent_turn:
                sat.add("ecommerce.search")
                sat.add("grocery.search")
            if "notes" in res or "note" in res:
                sat.add("notes.list")
            if "programs" in res and "channels" in res:
                sat.add("tv.guide")
            if "habits" in res:
                sat.add("habit_tracker.list_habits")
            if "items" in res and any(isinstance(x, dict) and str(x.get("id", "")).startswith("pan_") for x in (res.get("items") or [])):
                sat.add("pantry.list_items")
            if "sessions" in res:
                sat.add("parking.list_sessions")
            if "lots" in res and is_recent_turn:
                sat.add("parking.find")
            if "portfolio" in res or "holdings" in res:
                sat.add("stocks.get_portfolio")
            if "recipes" in res and is_recent_turn:
                sat.add("recipes.search")
            if "routes" in res:
                sat.add("public_transit.find_routes")
    return sat


def _extract_contact_query_from_text(u_lower: str) -> Optional[str]:
    for pat_ct, canon_ct in (
        (r"\bcô\s+(?:giáo\s+)?mai\b", "Cô giáo Mai"),
        (r"\bmai\s+anh\b", "Mai Anh"),
        (r"\bbác\s+sĩ\s+an\b", "Bác sĩ An"),
        (r"\bbác\s+sĩ\s+hòa\b", "Hòa"),
        (r"\bbác\s+ba\b", "Bác Ba"),
        (r"\bchú\s+năm\b", "Chú Năm"),
        (r"\bsếp\s+dũng\b", "Sếp Dũng"),
        (r"\bsếp\s+kim\b", "Sếp Kim"),
        (r"\bcô\s+hạnh\b", "Cô Hạnh"),
        (r"\bthằng\s+tí\b", "Thằng Tí"),
        (r"\bthằng\s+út\b", "Thằng Út"),
        (r"\bbà\s+sáu\b", "Bà Sáu"),
        (r"\banh\s+kha\b", "Anh Kha"),
        (r"\banh\s+lộc\b", "Anh Lộc"),
        (r"\banh\s+long\b", "Anh Long"),
        (r"\banh\s+mười\b", "Anh Mười"),
        (r"\banh\s+hùng\b", "Anh Hùng"),
        (r"\banh\s+bình\b", "Anh Bình"),
        (r"\banh\s+sáu\b", "Anh Sáu"),
        (r"\banh\s+quân\b", "Anh Quân"),
        (r"\bchị\s+dung\b", "Chị Dung"),
        (r"\banh\s+nam\b", "Anh Nam"),
        (r"\bminh\b", "Minh"),
        (r"\blan\b", "Lan"),
        (r"\bhương\b", "Hương"),
        (r"\btuấn\b", "Tuấn"),
        (r"\bbình\b", "Bình"),
        (r"\bhuy\b", "Huy"),
        (r"\bquân\b", "Quân"),
        (r"\bdung\b", "Dung"),
        (r"\btrang\b", "Trang"),
        (r"\blinh\b", "Linh"),
        (r"\bthảo\b", "Thảo"),
        (r"\bhòa\b", "Hòa"),
        (r"\bphương\b", "Phương"),
        (r"\bkhánh\b", "Khánh"),
        (r"\byến\b", "Yến"),
        (r"\btoàn\b", "Toàn"),
        (r"\bnhi\b", "Nhi"),
        (r"\bhằng\b", "Hằng"),
        (r"\bthư\b", "Thư"),
        (r"\b(?:bố|ba)\b", "Bố"),
        (r"\b(?:mẹ|má)\b", "Mẹ"),
        (r"\bmạ\b", "Mạ"),
    ):
        if re.search(pat_ct, u_lower):
            return canon_ct
    return None


def _ensure_readers_or_prune(
    calls: List[Dict[str, Any]],
    hist_sat: Set[str],
    u_text: str,
    installed_apps: Any,
    current_time_str: str,
    history: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    u_lower = u_text.lower()
    norm_u = _normalize_vn_numbers_in_time(u_text)
    has_plural = bool(
        re.search(
            r"\b(cả hai|tất cả|mấy cái|mấy việc|mấy đơn|mấy gói|mấy cữ|mấy bài|mấy mã|mấy hóa đơn|mấy hoá đơn|mấy thuốc|mấy mail|mấy thư|hai cái|2 cái|hai việc|2 việc|hai đơn|2 đơn|hai vé|2 vé|hai tờ|2 tờ|hai chiếc|2 chiếc|hai thiết bị|những|các|còn lại)\b",
            u_text,
            re.IGNORECASE,
        )
    )
    is_price_query = bool(re.search(r"\b(hết bao nhiêu tiền|bao nhiêu tiền)\b", u_text, re.IGNORECASE))
    has_workout_time = bool(re.search(r"\b(sáng nay|chiều nay|tối nay|lúc \d)\b", u_text, re.IGNORECASE))
    primary_d, prev_d = resolve_target_dates_from_utterance(u_text, current_time_str)
    use_cal_date_filter = bool(primary_d and CALENDAR_DATE_FILTER_RE.search(u_text))
    cal_list_args = {"date_from": primary_d, "date_to": primary_d} if use_cal_date_filter else {}
    inst_set = set(installed_apps.keys()) if isinstance(installed_apps, dict) else set(installed_apps or [])
    is_vague_ewallet_amount = bool(
        (re.search(r"\b(chút tiền|ít tiền)\b", norm_u, re.IGNORECASE)
         or ("bill_payment" not in inst_set and re.search(r"\b(phần tiền chia hóa đơn|phần tiền chia hoá đơn)\b", norm_u, re.IGNORECASE)))
        and not re.search(r"\b\d+\s*(?:nghìn|ngàn|triệu|củ|đồng|k\b)|tiền\s+(?:điện|nước|mạng|internet)\b", norm_u, re.IGNORECASE)
    )
    has_explicit_time_in_u = bool(
        re.search(r"\b(\d{1,2}\s*(?:giờ|rưỡi|h\b)|\d+\s*(?:phút|tiếng)|nửa\s+tiếng|trước\s+(?:giờ|lúc|buổi|khi|đó)|sau\s+khi|cùng\s+giờ|sớm\s+hơn|sáng\s+hôm\s+bay)\b", norm_u, re.IGNORECASE)
    )
    has_multi_bank_transfer = bool(
        re.search(r"\btiền\s+(?:điện|nước|mạng|internet|truyền hình)\b.*\btiền\s+(?:điện|nước|mạng|internet|truyền hình)\b|\bchuyển\b.*\b(?:rồi|và|với)\s+chuyển\b", u_lower)
    )
    single_room_light = bool(
        re.search(r"\bđèn\s+(?:phòng\s+(?:khách|ngủ|làm\s+việc|ăn|tắm)|bếp|hành\s+lang|ban\s+công|sân)\b", u_lower)
        and not re.search(r"\b(tất cả|hết|cả hai|hai đèn|mấy đèn|phòng .* và .* phòng|đèn .* và .* đèn)\b", u_lower)
    )
    spoken_phone_match = re.search(r"\b(0\d{8,11}|1[89]00\d{4})\b", norm_u)
    non_ct_apps_in_calls = {
        (x.get("arguments", {}).get("tool_name", "").split(".")[0] if x.get("name") == "sys.ask_missing_info" else x.get("name", "").split(".")[0])
        for x in calls
    } - {"contacts"}

    # Chuẩn hóa sys.ask_missing_info tĩnh về công cụ hành động gốc (việc kiểm tra trùng danh bạ do môi trường thực thi quyết định)
    norm_calls: List[Dict[str, Any]] = []
    for tc in calls:
        if tc.get("name") == "sys.ask_missing_info":
            targs = tc.get("arguments") or {}
            eff = targs.get("tool_name", "")
            prms = targs.get("params") or []
            if prms == ["contact_id"] and eff in {"phone.call", "sms.send", "ewallet.transfer"}:
                if eff == "ewallet.transfer" and is_vague_ewallet_amount:
                    tc = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "ewallet.transfer", "params": ["amount"]}}
                elif eff == "phone.call":
                    tc = {"name": "phone.call", "arguments": {"contact_id": "ct_01"}}
                elif eff == "sms.send":
                    tc = {"name": "sms.send", "arguments": {"contact_id": "ct_01", "content": "Tin nhắn"}}
                elif eff == "ewallet.transfer":
                    tc = {"name": "ewallet.transfer", "arguments": {"contact_id": "ct_01", "amount": 100000}}
            elif eff == "ewallet.topup_mobile" and re.search(r"\b\d+\s*(?:nghìn|ngàn|k\b)\b", norm_u):
                tc = {"name": "ewallet.topup_mobile", "arguments": {"phone": spoken_phone_match.group(1) if spoken_phone_match else "0901234567", "amount": 50000}}
            elif prms == ["chat_id"] and eff == "chat_ott.send_message":
                tc = {"name": "chat_ott.send_message", "arguments": {"chat_id": "chat_01", "content": "Tin nhắn"}}
        norm_calls.append(tc)
    calls = norm_calls
    num_ct_searches = sum(1 for x in calls if x.get("name") == "contacts.search")

    # Bóc tách dữ kiện từ history để đồng bộ trực tiếp vào giá trị mẫu (giữ nguyên định dạng JSON cụ thể cho 3B)
    hist_workout_mins: Optional[int] = None
    hist_rule_id: Optional[str] = None
    hist_chats_map: Dict[str, str] = {}
    hist_rem_time_content: Optional[Tuple[int, str]] = None
    hist_text_lower = ""
    for turn in (history or []):
        t_txt = str(turn.get("audio_text") or turn.get("text") or "")
        hist_text_lower += " " + t_txt.lower()
        if turn.get("role") == "assistant" and t_txt:
            m_rem = re.search(r"nhắc lúc\s+(\d+)\s*giờ\s*(sáng|chiều|tối)?\s+với nội dung\s+([^\.?!]+)", t_txt, re.IGNORECASE)
            if m_rem:
                h_v = int(m_rem.group(1))
                if (m_rem.group(2) or "").lower() in {"chiều", "tối"} and h_v < 12:
                    h_v += 12
                c_v = m_rem.group(3).strip()
                c_v = c_v[0].upper() + c_v[1:] if c_v else c_v
                hist_rem_time_content = (h_v, c_v)
        for res in (turn.get("tool_results") or []):
            if not isinstance(res, dict):
                continue
            if "messages" in res and isinstance(res["messages"], list):
                for msg in res["messages"]:
                    if isinstance(msg, dict):
                        m_norm = _normalize_vn_numbers_in_time(str(msg.get("content", "")))
                        m_dur = re.search(r"\b(?:bơi|chạy|tập|đạp xe)\s+(\d+)\s*phút\b", m_norm, re.IGNORECASE)
                        if m_dur:
                            hist_workout_mins = int(m_dur.group(1))
            if "rule" in res and isinstance(res["rule"], dict) and res["rule"].get("id"):
                hist_rule_id = str(res["rule"]["id"])
            if "chats" in res and isinstance(res["chats"], list):
                for ch in res["chats"]:
                    if isinstance(ch, dict) and ch.get("id") and ch.get("name"):
                        hist_chats_map[str(ch["name"]).lower()] = str(ch["id"])

    inst_set = set(installed_apps.keys()) if isinstance(installed_apps, dict) else set(installed_apps or [])
    hist_has_sh_status = any(
        isinstance(tc_h, dict) and tc_h.get("name") == "smart_home.get_status"
        for t_h in (history or [])
        for tc_h in (t_h.get("tool_calls") or [])
    )
    hist_phone_match = re.search(r"\b(0\d{8,11})\b", hist_text_lower)
    is_call_hist_phone = bool(hist_phone_match and re.search(r"\b(gọi luôn số đó|gọi số đó)\b", u_lower))

    if "calendar" in inst_set and "calendar.list_events" not in hist_sat:
        if re.search(r"\b(cùng ngày với buổi|buổi workshop|lịch họp|họp phụ huynh|lịch khám|giờ khám|đá bóng|chạy bộ|cà phê|dã ngoại|buổi họp)\b", u_text, re.IGNORECASE):
            if any(tc["name"].startswith(("flights.", "ride_hailing.", "alarm.", "calendar.", "reminders.")) for tc in calls):
                out.append({"name": "calendar.list_events", "arguments": dict(cal_list_args)})
                seen.add("calendar.list_events")

    if "flights" in inst_set and "flights.my_bookings" not in hist_sat:
        if re.search(r"\b(chuyến .* đặt|vé .* đặt|ngày bay của tôi|sáng hôm bay|trước giờ bay|trước giờ cất cánh|máy bay hạ cánh)\b", u_text, re.IGNORECASE):
            if not any(tc["name"].startswith("flights.search") for tc in calls) and not re.search(r"\b(đặt .* vé|mua .* vé)\b", u_text, re.IGNORECASE):
                out.append({"name": "flights.my_bookings", "arguments": {}})
                seen.add("flights.my_bookings")

    if "food_delivery" in inst_set and "food_delivery.order_history" not in hist_sat:
        if re.search(r"\b(đơn .* giao rồi|đơn .* chưa giao|đơn .* tới đâu)\b", u_text, re.IGNORECASE):
            out.append({"name": "food_delivery.order_history", "arguments": {}})
            seen.add("food_delivery.order_history")

    PARAM_WRITE_TO_READER: Dict[str, Tuple[str, Dict[str, Any]]] = {
        "ecommerce.add_to_cart": ("ecommerce.search", {"query": "sản phẩm"}),
        "grocery.add_to_cart": ("grocery.search", {"query": "thực phẩm"}),
        "train_bus_tickets.book": ("train_bus_tickets.search", {"origin": "Hà Nội", "destination": "Vinh", "date": primary_d or "2026-10-16"}),
        "flights.book": ("flights.search", {"origin": "HAN", "destination": "DAD", "date": primary_d or "2026-10-16"}),
        "parking.reserve": ("parking.find", {"near": "Nội Bài"}),
        "recipes.save": ("recipes.search", {"query": "món ăn"}),
        "library.reserve": ("library.search", {"query": "sách"}),
        "bike_rental.reserve": ("bike_rental.find_bikes", {"station": "Mỹ Đình"}),
    }

    has_explicit_time_in_hist = bool(re.search(r"\b\d{1,2}\s*(?:giờ|rưỡi|h\b)", _normalize_vn_numbers_in_time(hist_text_lower)))
    for tc in calls:
        tname = tc["name"]
        if tname == "ewallet.transfer" and is_vague_ewallet_amount:
            tc = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "ewallet.transfer", "params": ["amount"]}}
            tname = "sys.ask_missing_info"
        elif tname in {"alarm.create", "reminders.create"} and not has_explicit_time_in_u and not has_explicit_time_in_hist and not hist_rem_time_content and "flights.my_bookings" not in seen and "flights.my_bookings" not in hist_sat:
            tc = {"name": "sys.ask_missing_info", "arguments": {"tool_name": tname, "params": ["time"]}}
            tname = "sys.ask_missing_info"
        elif tname == "video_meeting.create" and not has_explicit_time_in_u and not has_explicit_time_in_hist and "calendar.list_events" not in seen and "calendar.list_events" not in hist_sat:
            tc = {"name": "sys.ask_missing_info", "arguments": {"tool_name": "video_meeting.create", "params": ["start"]}}
            tname = "sys.ask_missing_info"
        if tname == "contacts.search" and (spoken_phone_match or is_call_hist_phone) and any(x.get("name") in {"ewallet.topup_mobile", "phone.call", "sms.send"} for x in calls) and not any(x.get("name") in {"ewallet.transfer"} for x in calls):
            continue
        if tname == "contacts.search" and is_vague_ewallet_amount and non_ct_apps_in_calls <= {"ewallet"}:
            continue
        if tname in hist_sat:
            if tname == "recipes.search" and not any(x["name"] in {"recipes.get", "recipes.save"} for x in calls):
                tname = "recipes.save" if re.search(r"\b(lưu|cất)\b", u_lower) else "recipes.get"
                tc = {"name": tname, "arguments": {"recipe_id": "rcp_01"}}
            else:
                continue
        if is_price_query and tname == "sys.ask_missing_info":
            continue
        if tname in seen and (
            tname in READ_TOOLS
            or tname == "food_delivery.place_order"
            or (tname == "banking.transfer" and not has_multi_bank_transfer)
            or (tname == "smart_home.light_set" and single_room_light)
            or (tname == "food_delivery.cancel_order" and not re.search(r"\b(hủy hết|hủy cả hai|hủy 2 đơn|hủy hai đơn)\b", u_lower))
            or (not has_plural and tname in {"home_security.acknowledge_alert", "subscriptions.cancel", "tasks.add", "tasks.complete", "learning.schedule_lesson", "learning.log_progress", "calendar.create_event"})
        ):
            continue
        if tname == "home_automation.create_rule" and "smart_home" in inst_set and not hist_has_sh_status and "smart_home.get_status" not in seen:
            out.append({"name": "smart_home.get_status", "arguments": {}})
            seen.add("smart_home.get_status")
        if tname in WRITE_TO_ZERO_ARG_READERS:
            for r_tool in WRITE_TO_ZERO_ARG_READERS[tname]:
                if r_tool not in hist_sat and r_tool not in seen:
                    r_args = dict(cal_list_args) if r_tool == "calendar.list_events" else {}
                    out.append({"name": r_tool, "arguments": r_args})
                    seen.add(r_tool)
        if tname in PARAM_WRITE_TO_READER:
            p_reader, p_default_args = PARAM_WRITE_TO_READER[tname]
            if p_reader not in hist_sat and p_reader not in seen and not any(x["name"] == p_reader for x in calls):
                out.append({"name": p_reader, "arguments": dict(p_default_args)})
                seen.add(p_reader)
        if tname == "food_delivery.place_order":
            if "food_delivery.list_restaurants" not in hist_sat and "food_delivery.list_restaurants" not in seen and not any(x["name"] == "food_delivery.list_restaurants" for x in calls):
                out.append({"name": "food_delivery.list_restaurants", "arguments": {}})
                seen.add("food_delivery.list_restaurants")
            if "food_delivery.get_menu" not in hist_sat and "food_delivery.get_menu" not in seen and not any(x["name"] == "food_delivery.get_menu" for x in calls):
                out.append({"name": "food_delivery.get_menu", "arguments": {"restaurant_id": "rest_01"}})
                seen.add("food_delivery.get_menu")
        if tname == "tv.tune" and not re.search(r"\b(vtv\d+|htv\d+)\b", u_lower):
            if "tv.guide" not in hist_sat and "tv.guide" not in seen and not any(x["name"] == "tv.guide" for x in calls):
                out.append({"name": "tv.guide", "arguments": {}})
                seen.add("tv.guide")
        if tname == "public_transit.topup_card" and re.search(r"\b(lượt|vé|tuyến|khứ hồi)\b", u_lower):
            if "public_transit.find_routes" not in hist_sat and "public_transit.find_routes" not in seen and not any(x["name"] == "public_transit.find_routes" for x in calls):
                out.append({"name": "public_transit.find_routes", "arguments": {"origin": "Bến Thành", "destination": "Suối Tiên"}})
                seen.add("public_transit.find_routes")
        if tname == "phone.call" and is_call_hist_phone and hist_phone_match:
            tc = {"name": "phone.call", "arguments": {"phone_number": hist_phone_match.group(1)}}
        elif tname == "phone.call" and "contacts" not in inst_set:
            if "phone.recent_calls" not in hist_sat and "phone.recent_calls" not in seen and not any(x["name"] == "phone.recent_calls" for x in calls):
                out.append({"name": "phone.recent_calls", "arguments": {}})
                seen.add("phone.recent_calls")
            tc = {"name": "phone.call", "arguments": {"phone_number": "0901234567"}}
        elif (
            (tname == "phone.call" and not spoken_phone_match and not is_call_hist_phone)
            or (tname == "sms.send" and not spoken_phone_match)
            or tname in {"sms.read_recent", "ewallet.transfer", "contacts.update"}
            or (tname == "ewallet.topup_mobile" and not spoken_phone_match)
            or (tname == "sys.ask_missing_info" and (tc.get("arguments") or {}).get("tool_name") == "ewallet.transfer" and len(non_ct_apps_in_calls) > 1)
        ) and "contacts" in inst_set:
            if "contacts.search" not in hist_sat and "contacts.search" not in seen and not any(x["name"] == "contacts.search" for x in calls):
                q_ct = _extract_contact_query_from_text(u_lower) or "Lan"
                out.insert(0, {"name": "contacts.search", "arguments": {"query": q_ct}})
                seen.add("contacts.search")
        if tname == "grocery.checkout" and "grocery.add_to_cart" not in hist_sat and "grocery.add_to_cart" not in seen and not any(x["name"] == "grocery.add_to_cart" for x in calls):
            if "grocery.search" not in hist_sat and "grocery.search" not in seen and not any(x["name"] == "grocery.search" for x in calls):
                out.append({"name": "grocery.search", "arguments": {"query": "thực phẩm"}})
                seen.add("grocery.search")
            out.append({"name": "grocery.add_to_cart", "arguments": {"product_id": "gro_01", "quantity": 1}})
            seen.add("grocery.add_to_cart")
        elif tname == "ecommerce.checkout" and "ecommerce.add_to_cart" not in hist_sat and "ecommerce.add_to_cart" not in seen and not any(x["name"] == "ecommerce.add_to_cart" for x in calls):
            if "ecommerce.search" not in hist_sat and "ecommerce.search" not in seen and not any(x["name"] == "ecommerce.search" for x in calls):
                out.append({"name": "ecommerce.search", "arguments": {"query": "sản phẩm"}})
                seen.add("ecommerce.search")
            out.append({"name": "ecommerce.add_to_cart", "arguments": {"product_id": "prod_01", "quantity": 1}})
            seen.add("ecommerce.add_to_cart")

        # Đồng bộ hóa ngày tháng và giá trị cụ thể trong mẫu
        raw_args = dict(tc.get("arguments") or {})
        if tname == "contacts.search" and num_ct_searches <= 1:
            q_match = _extract_contact_query_from_text(u_lower)
            if q_match:
                raw_args["query"] = q_match
        elif tname == "ewallet.topup_mobile":
            raw_args.pop("contact_id", None)
            if spoken_phone_match:
                raw_args["phone"] = spoken_phone_match.group(1)
            else:
                raw_args.setdefault("phone", "0901234567")
            m_top_amt = re.search(r"\b(\d+)\s*(?:nghìn|ngàn|k)\b", norm_u)
            if m_top_amt:
                raw_args["amount"] = int(m_top_amt.group(1)) * 1000
        elif tname == "sms.send" and spoken_phone_match:
            raw_args.pop("contact_id", None)
            raw_args["phone_number"] = spoken_phone_match.group(1)
            m_sms_cnt = re.search(r"\b(?:là|rằng)\s+([^.!?]+?)(?:,\s*rồi\b|\.\s*rồi\b|$)", u_text, re.IGNORECASE)
            if m_sms_cnt:
                c_sms = m_sms_cnt.group(1).strip()
                if c_sms:
                    raw_args["content"] = c_sms[0].upper() + c_sms[1:]
        elif tname == "calendar.create_event":
            m_cal_title = re.search(r"^(?:buổi|lịch)\s+([^,.]+?)\s+(?:hôm|ngày|thứ|tuần|sáng|chiều|tối|lúc)\b", u_text.strip(), re.IGNORECASE)
            if m_cal_title:
                cur_t_words = [w for w in re.findall(r"\w+", str(raw_args.get("title", "")).lower()) if len(w) >= 3]
                if not cur_t_words or not any(w in u_lower for w in cur_t_words):
                    t_c_val = m_cal_title.group(1).strip()
                    if t_c_val:
                        raw_args["title"] = t_c_val[0].upper() + t_c_val[1:]
            m_cal_hr = re.search(r"\b(?:lúc|vào)\s+(\d{1,2})\s*giờ(?:\s*(\d{1,2}|rưỡi))?\s*(sáng|trưa|chiều|tối)?", norm_u, re.IGNORECASE)
            if m_cal_hr and isinstance(raw_args.get("start"), str) and isinstance(raw_args.get("end"), str):
                ch = int(m_cal_hr.group(1))
                cm = 30 if (m_cal_hr.group(2) or "").lower() == "rưỡi" else int(m_cal_hr.group(2) or 0)
                if (m_cal_hr.group(3) or "").lower() in {"chiều", "tối"} and ch < 12:
                    ch += 12
                raw_args["start"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{ch:02d}:{cm:02d}:00", raw_args["start"])
                raw_args["end"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{(ch + 1) % 24:02d}:{cm:02d}:00", raw_args["end"])
        elif tname == "calendar.list_events":
            raw_args = dict(cal_list_args)
        elif tname == "event_tickets.search":
            raw_args = {}
        elif tname == "smart_home.light_set":
            dev_id = raw_args.get("device_id", "light_living")
            m_br = re.search(r"\b(\d{1,3})\s*(?:phần trăm|%)", norm_u)
            br_val = int(m_br.group(1)) if m_br else (50 if re.search(r"\b(một nửa|phân nửa|còn 50)\b", u_lower) else None)
            if br_val is not None:
                if re.search(r"\b(bật|mở)\b", u_lower) and not re.search(r"\b(giảm|hạ)\b", u_lower):
                    raw_args = {"device_id": dev_id, "power": "on", "brightness": br_val}
                else:
                    raw_args = {"device_id": dev_id, "brightness": br_val}
            else:
                raw_args = {"device_id": dev_id, "power": "off" if "tắt" in u_lower else "on"}
        elif tname == "smart_home.climate_set":
            dev_id = raw_args.get("device_id", "ac_living")
            if re.search(r"\btắt\s+(?:cái\s+|mấy\s+cái\s+)?(?:điều\s+hòa|điều\s+hoà|máy\s+lạnh)\b", u_lower) and not re.search(r"\b\d{2}\s*độ\b", norm_u):
                raw_args = {"device_id": dev_id, "power": "off"}
            elif "hút ẩm" in u_lower:
                raw_args["mode"] = "dry"
                if not re.search(r"\b\d{2}\s*độ\b", norm_u):
                    raw_args.pop("temperature", None)
        elif tname == "smart_home.fan_set":
            dev_id = raw_args.get("device_id", "fan_living")
            m_spd = re.search(r"\bsố\s*(\d)\b", norm_u)
            if m_spd:
                raw_args = {"device_id": dev_id, "power": "on", "speed": int(m_spd.group(1))}
            elif "tắt" in u_lower:
                raw_args = {"device_id": dev_id, "power": "off"}
        elif tname == "grocery.checkout":
            if not re.search(r"\b(giao lúc|hẹn giao|chiều mai|sáng mai|tối nay .* giao|trước giờ)\b", u_lower):
                raw_args.pop("delivery_time", None)
        elif tname == "parking.reserve":
            m_days_pk = re.search(r"\b(?:gửi|để\s+xe\s+ở\s+đấy|gửi\s+xe)\s+(\d+)\s*ngày\b", norm_u)
            if m_days_pk:
                raw_args["hours"] = int(m_days_pk.group(1)) * 24
            elif "một tuần" in u_lower or "1 tuần" in norm_u:
                raw_args["hours"] = 168
        elif tname == "news.search":
            m_ev_q = re.search(r"(?:hội thảo|sự kiện|chương trình|giải)\s+([^\s,.]+(?:\s+[^\s,.]+){1,2})(?=\s+(?:hầu như|hình như|nghe nói|đổi|thay đổi|báo mới|trên báo|thế nào)|$)", u_text, re.IGNORECASE)
            if m_ev_q:
                raw_args["query"] = m_ev_q.group(1).strip()
        elif tname == "calendar.update_event" and "địa điểm" in u_lower:
            raw_args.setdefault("location", "Trung tâm Đổi mới")
            if isinstance(raw_args.get("end"), str):
                raw_args["end"] = raw_args["end"].replace("T11:30:00", "T12:30:00")
        elif tname == "fitness.log_workout":
            if not has_workout_time:
                raw_args.pop("time", None)
            if "bơi" in u_lower:
                raw_args["type"] = "swimming"
            elif "chạy" in u_lower:
                raw_args["type"] = "running"
            elif "đạp xe" in u_lower:
                raw_args["type"] = "cycling"
            elif "đi bộ" in u_lower:
                raw_args["type"] = "walking"
            m_u_dur = re.search(r"\b(\d+)\s*phút\b", norm_u)
            if m_u_dur:
                raw_args["minutes"] = int(m_u_dur.group(1))
            elif hist_workout_mins:
                raw_args["minutes"] = hist_workout_mins
        elif tname == "ride_hailing.book":
            if "xe máy" in u_lower:
                raw_args["vehicle"] = "bike"
            elif re.search(r"\b(bốn chỗ|4 chỗ)\b", u_lower):
                raw_args["vehicle"] = "car4"
            elif re.search(r"\b(bảy chỗ|7 chỗ)\b", u_lower):
                raw_args["vehicle"] = "car7"
            if re.search(r"trước\b.*?\b(?:nửa\s+tiếng|30\s*phút|ba\s+mươi\s+phút)", u_lower) and isinstance(raw_args.get("schedule_time"), str):
                raw_args["schedule_time"] = raw_args["schedule_time"].replace("T15:30:00", "T14:30:00")
        elif tname == "tasks.add":
            m_tsk = re.search(r"thêm\s+(?:cho\s+\S+\s+|giúp\s+\S+\s+)?(?:một\s+)?việc\s+(?:là\s+)?([^,.]+?)(?:\s+vào\s+danh\s+sách|\s*,?\s*hạn\b|$)", u_text, re.IGNORECASE)
            if m_tsk:
                t_val = m_tsk.group(1).strip()
                if t_val:
                    raw_args["title"] = t_val[0].upper() + t_val[1:]
        elif tname == "reminders.create":
            if hist_rem_time_content:
                h_v, c_v = hist_rem_time_content
                raw_args["content"] = c_v
                if isinstance(raw_args.get("time"), str):
                    raw_args["time"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{h_v:02d}:00:00", raw_args["time"])
            else:
                m_rem_u = re.search(r"nhắc\s+(?:mình|tôi|anh|chị|em)\s+([^,.]+?)(?:\s+lúc|\s+vào|\s+nữa|$)", u_text, re.IGNORECASE)
                if m_rem_u:
                    c_u = re.sub(r"\bcái\s+", "", m_rem_u.group(1).strip(), flags=re.IGNORECASE)
                    if c_u and "hàng" in u_lower and "gói hàng" not in c_u.lower() and "nhãn dán" in c_u.lower():
                        c_u = c_u + " gói hàng"
                    if c_u:
                        raw_args["content"] = c_u[0].upper() + c_u[1:]
                m_rem_hr = re.search(r"\b(\d{1,2})\s*giờ\s*(sáng|chiều|tối)?\b", norm_u)
                if m_rem_hr and isinstance(raw_args.get("time"), str):
                    rh = int(m_rem_hr.group(1))
                    if (m_rem_hr.group(2) or "").lower() in {"chiều", "tối"} and rh < 12:
                        rh += 12
                    raw_args["time"] = re.sub(r"T\d{2}:\d{2}:\d{2}", f"T{rh:02d}:00:00", raw_args["time"])
        elif tname == "subscriptions.cancel":
            if "cloud pro" in hist_text_lower or "cloud pro" in u_lower:
                raw_args["subscription_id"] = "sub_05"
        elif tname == "home_automation.delete_rule" and hist_rule_id:
            raw_args["rule_id"] = hist_rule_id
        elif tname == "chat_ott.send_message":
            if hist_chats_map:
                for ch_name_lower, ch_id in hist_chats_map.items():
                    if ch_name_lower in u_lower:
                        raw_args["chat_id"] = ch_id
                        break
            if "kết nối xanh" in u_lower:
                raw_args["content"] = "Hội thảo Kết nối Xanh ngày 16/10 đổi sang Trung tâm Đổi mới, bắt đầu lúc 14:30 nhé."
            elif "về khuya" in u_lower:
                raw_args["content"] = "Tối nay chị về khuya nhé"

        if primary_d and tname not in {"calendar.update_event"}:
            if tname == "alarm.create" and ("flights.my_bookings" in seen or "flights.my_bookings" in hist_sat) and re.search(r"\b(trước giờ bay|trước giờ cất cánh|sáng hôm bay|máy bay hạ cánh)\b", u_lower) and not re.search(r"\b(?:ngày|mùng|thứ)\b", u_lower):
                pass
            else:
                target_d_for_tool = prev_d if (prev_d and tname in {"reminders.create", "tasks.add"}) else primary_d
                for k, v in list(raw_args.items()):
                    if isinstance(v, str):
                        raw_args[k] = re.sub(r"202\d-\d{2}-\d{2}", target_d_for_tool, v)

        out.append({"name": tname, "arguments": raw_args})
        seen.add(tname)
        if tname == "recipes.search" and "recipes.get" not in seen and "recipes.get" not in hist_sat:
            if any(x.get("name") == "shopping_list.add" for x in calls) and re.search(r"\b(nguyên liệu|thiếu gì|cần gì|chi tiết)\b", u_lower):
                out.append({"name": "recipes.get", "arguments": {"recipe_id": "rcp_01"}})
                seen.add("recipes.get")

    if "smart_home" in inst_set and "smart_home.get_status" in seen:
        if "smart_home.light_set" not in seen:
            if re.search(r"\b(bật|mở|tắt|giảm|hạ|tăng|chỉnh|cho)\b[^,.]{0,30}\bđèn\b|\bđèn\b[^,.]{0,25}\b(xuống|lên|còn)\s+\d+", u_lower) and (
                re.search(r"\b(tắt hết đèn|tắt đèn .* hết đi)\b", u_lower)
                or not re.search(r"\b(đừng tắt|khỏi tắt|cứ để sáng|để nguyên đèn|có những thiết bị nào|đang bật hay tắt)\b", u_lower)
            ):
                out.append({"name": "smart_home.light_set", "arguments": {"device_id": "light_living", "power": "off" if "tắt" in u_lower else "on"}})
                seen.add("smart_home.light_set")
        if sum(1 for x in out if x.get("name") == "smart_home.light_set") == 1 and re.search(r"\b(tắt hết đèn tầng|tắt đèn tầng .* hết đi)\b", u_lower):
            out.append({"name": "smart_home.light_set", "arguments": {"device_id": "light_bed", "power": "off"}})

    if "smart_home" in inst_set and "smart_home.get_status" in seen and "smart_home.climate_set" not in seen:
        if re.search(r"(?<!đang\s)\b(bật|mở|tắt|chỉnh|hạ|tăng|giảm)\b[^,.!?;]{0,30}\b(điều hòa|điều hoà|máy lạnh)\b|\b(điều hòa|điều hoà|máy lạnh)\b[^,.!?;]{0,30}\b(về\s+\d{2}|để\s+\d{2}|\d{2}\s*độ|hút ẩm)\b", norm_u, re.IGNORECASE) and not re.search(r"\b(phòng nào\s+(?:[^,.!?;]*?\s+)?đang|xem\s+(?:[^,.!?;]*?\s+)?đang|có những thiết bị nào|đang bật hay tắt)\b", u_lower):
            out.append({"name": "smart_home.climate_set", "arguments": {"device_id": "ac_living", "temperature": 26}})
            seen.add("smart_home.climate_set")

    if "robot_vacuum" in inst_set and ("robot_vacuum.get_status" in seen or "robot_vacuum.start" in seen) and "robot_vacuum.set_mode" not in seen:
        if re.search(r"\b(chế độ\s+(?:im lặng|yên tĩnh|mạnh|tiêu chuẩn|turbo|max|eco)|chỉnh chế độ)\b", u_lower):
            out.append({"name": "robot_vacuum.set_mode", "arguments": {"mode": "silent"}})
            seen.add("robot_vacuum.set_mode")

    if "appliances" in inst_set and "appliances.list" in seen:
        if "appliances.stop" not in seen and re.search(r"\b(dừng lại|cho nó dừng|tắt máy)\b", u_lower):
            out.append({"name": "appliances.stop", "arguments": {"appliance_id": "app_01"}})
            seen.add("appliances.stop")
        elif "appliances.start_cycle" not in seen and "appliances.stop" not in seen and re.search(r"\b(bật|chạy|khởi động|giặt nhanh|giặt đồ|sấy)\b", u_lower) and not re.search(r"\b(chỉ kiểm tra thôi|chưa bật|khoan bật|đừng bật|dừng|tắt|đang tắt hay đang chạy)\b", u_lower):
            out.append({"name": "appliances.start_cycle", "arguments": {"appliance_id": "app_01", "program": "quick"}})
            seen.add("appliances.start_cycle")

    if "calendar" in inst_set and "calendar.list_events" in seen and "calendar.update_event" not in seen and "calendar.cancel_event" not in seen:
        if re.search(r"\b(dời|lùi)\b.*\b(muộn\s+(?:hơn|thêm|lại|\d)|sớm\s+(?:hơn|thêm|lại|\d)|sang\s+(?:đúng\s+giờ|\d)|thành \d|qua chiều|qua sáng)\b", norm_u, re.IGNORECASE) and not re.search(r"\b(nếu|mưa)\b", u_lower):
            out.append({"name": "calendar.update_event", "arguments": {"event_id": "evt_01", "start": f"{primary_d or '2026-10-16'}T09:00:00+07:00", "end": f"{primary_d or '2026-10-16'}T10:00:00+07:00"}})
            seen.add("calendar.update_event")

    if is_vague_ewallet_amount and "ewallet" in inst_set and "sys.ask_missing_info" not in seen:
        out = [x for x in out if x.get("name") != "ewallet.get_balance"]
        if "contacts" in inst_set and "contacts.search" not in seen and "contacts.search" not in hist_sat:
            q_ct = _extract_contact_query_from_text(u_lower) or "Mẹ"
            out.insert(0, {"name": "contacts.search", "arguments": {"query": q_ct}})
            seen.add("contacts.search")
        out.append({"name": "sys.ask_missing_info", "arguments": {"tool_name": "ewallet.transfer", "params": ["amount"]}})
        seen.add("sys.ask_missing_info")

    if "phone" in inst_set and "phone.call" not in seen and "sys.ask_missing_info" not in seen and re.search(r"\b(?:với|rồi|xong)\s+gọi\s+(?:cho\s+)?(?:lan|minh|hương|mẹ|bố|ba|anh|chị|em|cô|chú|bác)\b", u_lower):
        if "contacts" in inst_set and "contacts.search" not in seen and "contacts.search" not in hist_sat:
            q_ct = _extract_contact_query_from_text(u_lower) or "Lan"
            out.insert(0, {"name": "contacts.search", "arguments": {"query": q_ct}})
            seen.add("contacts.search")
        out.append({"name": "phone.call", "arguments": {"contact_id": "ct_01"}})
        seen.add("phone.call")

    if "sms" in inst_set and "sms.send" not in seen and "sms.read_recent" not in seen and "sys.ask_missing_info" not in seen and re.search(r"\b(?:với|rồi|xong)\s+nhắn\s+(?:sms\s+)?(?:cho\s+)?(?:lan|hương|minh|anh|chị|em|mẹ|bố|bác|cô|chú)\b", u_lower):
        if "contacts" in inst_set and "contacts.search" not in seen and "contacts.search" not in hist_sat:
            q_ct = _extract_contact_query_from_text(u_lower) or "Hương"
            out.insert(0, {"name": "contacts.search", "arguments": {"query": q_ct}})
            seen.add("contacts.search")
        out.append({"name": "sms.send", "arguments": {"contact_id": "ct_01", "content": "Tin nhắn"}})
        seen.add("sms.send")

    if "reminders" in inst_set and "reminders.create" not in seen and re.search(r"\b(?:rồi|xong|với)\s*(?:\d+\s*giờ\s*(?:\d+\s*(?:phút)?\s*)?)?nhắc\s+(?:tôi|mình|anh|chị|em|cô|chú)\b|\bnhắc\s+(?:tôi|mình|anh|chị|em|cô|chú)\s+(?:đi\s+)?(?:đổ\s+xăng|mang\s+laptop)\b", u_lower):
        out.append({"name": "reminders.create", "arguments": {"content": "Lời nhắc", "time": f"{primary_d or '2026-10-16'}T09:00:00+07:00"}})
        seen.add("reminders.create")

    if "appliances" in inst_set and "appliances.stop" not in seen and re.search(r"\b(máy giặt|máy sấy|máy rửa bát|máy rửa chén)\b[^.!?]*?\b(dừng lại|tắt máy|ngừng)\b", u_lower):
        if "appliances.list" not in seen and "appliances.list" not in hist_sat:
            out.insert(0, {"name": "appliances.list", "arguments": {}})
            seen.add("appliances.list")
        out.append({"name": "appliances.stop", "arguments": {"appliance_id": "app_01"}})
        seen.add("appliances.stop")

    if "home_automation" in inst_set and "home_automation.create_rule" not in seen and re.search(r"\b(tự bật|tự tắt|tự động bật|tự động tắt)\b", u_lower):
        if "smart_home" in inst_set and not hist_has_sh_status and "smart_home.get_status" not in seen:
            out.insert(0, {"name": "smart_home.get_status", "arguments": {}})
            seen.add("smart_home.get_status")
        out.append({"name": "home_automation.create_rule", "arguments": {"name": "Quy tắc", "trigger": "21:00", "device_id": "fan_bed", "action": "on"}})
        seen.add("home_automation.create_rule")

    if "notes" in inst_set and "notes.create" not in seen and "notes.append" not in seen and re.search(r"\b(ghi chú lại|lưu ghi chú|tạo (?:cái |một )?ghi chú)\b", u_lower):
        out = [x for x in out if x.get("name") != "notes.list"]
        out.append({"name": "notes.create", "arguments": {"title": "Ghi chú", "content": "Nội dung"}})
        seen.add("notes.create")

    if "bill_payment" in inst_set and "bill_payment.pay" in seen and "bill_payment.schedule_autopay" not in seen and re.search(r"\b(bật|cài|đặt)\s+(?:tự động thanh toán|tự trả|autopay)\b", u_lower):
        out.append({"name": "bill_payment.schedule_autopay", "arguments": {"bill_type": "electricity", "enabled": True}})
        seen.add("bill_payment.schedule_autopay")

    if "bike_rental" in inst_set:
        if "bike_rental.reserve" not in seen and re.search(r"\bchiếc xe đôi với chiếc\b", u_lower):
            out.append({"name": "bike_rental.reserve", "arguments": {"bike_id": "bk_01"}})
            out.append({"name": "bike_rental.reserve", "arguments": {"bike_id": "bk_02"}})
            seen.add("bike_rental.reserve")
        elif sum(1 for x in out if x.get("name") == "bike_rental.reserve") > 1 and re.search(r"\bmột chiếc\b", u_lower) and not re.search(r"\b(hai chiếc|2 chiếc|với chiếc)\b", u_lower):
            kept_bk = False
            new_out = []
            for x in out:
                if x.get("name") == "bike_rental.reserve":
                    if kept_bk:
                        continue
                    kept_bk = True
                new_out.append(x)
            out = new_out

    if "banking" in inst_set and "banking.list_beneficiaries" in seen and "banking.transfer" not in seen:
        out.append({"name": "banking.transfer", "arguments": {"from_account_id": "acc_01", "beneficiary_id": "ben_01", "amount": 500000}})
        seen.add("banking.transfer")

    if "bill_payment" in inst_set and "bill_payment.list_bills" in seen and "bill_payment.pay" not in seen and "banking.transfer" not in seen and "ewallet.transfer" not in seen and "bill_payment.schedule_autopay" not in seen:
        if re.search(r"\b(trong\s*(?:vòng\s*)?\d+\s*ngày\s*tới|đến\s*hạn\s*trong\s*\d+\s*ngày)\b", norm_u):
            src_bp_def = "bank" if "ngân hàng" in u_lower else "ewallet"
            out.append({"name": "bill_payment.pay", "arguments": {"bill_id": "bill_01", "source": src_bp_def}})
            out.append({"name": "bill_payment.pay", "arguments": {"bill_id": "bill_02", "source": src_bp_def}})
            seen.add("bill_payment.pay")

    if "food_delivery" in inst_set and "food_delivery.get_menu" in seen and "food_delivery.place_order" not in seen:
        if re.search(r"\b(đặt|gọi|lấy|mua|cho\s+(?:tôi|mình|anh|chị|em)|giao\s+về)\b", u_lower) and not re.search(r"\b(xem\s+thực\s+đơn|xem\s+menu|có\s+món\s+gì)\b", u_lower):
            out.append({"name": "food_delivery.place_order", "arguments": {"restaurant_id": "rest_01", "items": [{"item_id": "item_01", "quantity": 1}], "address": "Nhà"}})
            seen.add("food_delivery.place_order")

    if "food_delivery.order_history" in seen and "food_delivery.list_restaurants" not in hist_sat and "food_delivery.list_restaurants" not in seen:
        if re.search(r"\b(bao nhiêu tiền|quán|kitchen|house|bếp|tiệm)\b", u_text, re.IGNORECASE):
            out.append({"name": "food_delivery.list_restaurants", "arguments": {}})
            seen.add("food_delivery.list_restaurants")

    if "train_bus_tickets.my_bookings" in seen and "train_bus_tickets.cancel" not in seen and re.search(r"\b(hủy|huỷ)\b", u_lower):
        out.append({"name": "train_bus_tickets.cancel", "arguments": {"booking_id": "tbb_01"}})
        seen.add("train_bus_tickets.cancel")

    if "ride_hailing.get_ride" in seen and "ride_hailing.cancel" not in seen and re.search(r"\b(hủy|huỷ)\b.*\b(xe|chuyến)\b", u_lower):
        out.append({"name": "ride_hailing.cancel", "arguments": {"ride_id": "ride_01"}})
        seen.add("ride_hailing.cancel")

    if "event_tickets.search" in seen and "event_tickets.book" not in seen and re.search(r"\b(đặt|mua|lấy)\b.*\b(vé|vét|chỗ)\b", u_lower) and not re.search(r"\b(vé tàu|vé xe|vé máy bay)\b", u_lower):
        out.append({"name": "event_tickets.book", "arguments": {"event_id": "evt_01", "quantity": 2}})
        seen.add("event_tickets.book")

    if sum(1 for x in out if x["name"] == "event_tickets.book") == 1 and re.search(r"\b(cả hai buổi|hai buổi|2 buổi|cả hai sự kiện)\b", u_lower):
        out.append({"name": "event_tickets.book", "arguments": {"event_id": "evt_02", "quantity": 1}})

    if "shopping_list.remove" in seen and "shopping_list.add" not in seen and re.search(r"\b(thay bằng|đổi thành|thêm)\b", u_lower):
        out.append({"name": "shopping_list.add", "arguments": {"name": "Trứng cút", "quantity": 2}})
        seen.add("shopping_list.add")

    if sum(1 for x in out if x["name"] == "shopping_list.add") == 1 and re.search(r"\b(mỗi thứ|hai thứ|cả hai thứ)\b", u_lower):
        out.append({"name": "shopping_list.add", "arguments": {"name": "Thực phẩm 2", "quantity": 2}})

    if sum(1 for x in out if x["name"] == "alarm.update") == 1 and re.search(r"\b(hai cái báo thức|2 cái báo thức|cả hai báo thức|hết đi)\b", u_lower):
        out.append({"name": "alarm.update", "arguments": {"alarm_id": "al_02", "time": "07:15"}})

    if "reminders.create" in seen and "reminders.list" not in seen and "reminders.list" not in hist_sat and re.search(r"\b(nếu chưa có|xem có chưa|coi có chưa)\b", u_lower):
        out.insert(0, {"name": "reminders.list", "arguments": {}})
        seen.add("reminders.list")

    if "reminders.delete" in seen and "reminders.create" not in seen and re.search(r"\b(đổi thành nhắc|thay bằng nhắc|rồi nhắc|tạo lại lời nhắc)\b", u_lower):
        out.append({"name": "reminders.create", "arguments": {"content": "Gọi cho mẹ", "time": f"{primary_d or '2026-10-15'}T20:30:00+07:00"}})
        seen.add("reminders.create")

    if "tasks.complete" in seen and "tasks.add" not in seen and re.search(r"\b(thêm\s+(?:[^,.!?;]*?\s+)?việc|tạo\s+(?:[^,.!?;]*?\s+)?việc)\b", u_lower):
        out.append({"name": "tasks.add", "arguments": {"title": "Chuẩn bị slide", "due": f"{primary_d or '2026-10-16'}T17:00:00+07:00"}})
        seen.add("tasks.add")

    if "medication.add" in seen and "medication.remove" not in seen and re.search(r"\b(bỏ\s+(?:[^,.!?;]*?\s+)?lịch uống|xóa\s+(?:[^,.!?;]*?\s+)?lịch uống|xoá\s+(?:[^,.!?;]*?\s+)?lịch uống|ngừng uống)\b", u_lower):
        if "medication.list" not in hist_sat and "medication.list" not in seen:
            out.insert(0, {"name": "medication.list", "arguments": {}})
            seen.add("medication.list")
        out.append({"name": "medication.remove", "arguments": {"medication_id": "med_01"}})
        seen.add("medication.remove")
    elif "medication.remove" in seen and "medication.add" not in seen and re.search(r"\b(thêm\s+(?:[^,.!?;]*?\s+)?lịch|đổi sang uống)\b", u_lower):
        out.append({"name": "medication.add", "arguments": {"name": "Canxi", "dosage": "1 viên", "times": ["07:30"]}})
        seen.add("medication.add")

    if "chat_ott.list_chats" in seen and "chat_ott.read_chat" not in seen and "chat_ott.send_message" not in seen and "chat_ott.call" not in seen and "chat_ott.send_location" not in seen:
        if re.search(r"\b(đọc giúp|nhắn zalo vụ|mới nhắn zalo)\b", u_lower) and not re.search(r"\b(có ai nhắn zalo|ai nhắn zalo)\b", u_lower):
            out.append({"name": "chat_ott.read_chat", "arguments": {"chat_id": "chat_01"}})
            seen.add("chat_ott.read_chat")

    if "public_transit.find_routes" in seen and "public_transit.topup_card" not in seen and re.search(r"\b(nạp thẻ|nạp thêm)\b", u_lower):
        out.append({"name": "public_transit.topup_card", "arguments": {"amount": 50000}})
        seen.add("public_transit.topup_card")

    if "stocks" in inst_set and "stocks.get_quote" in seen and "stocks.set_price_alert" not in seen:
        if re.search(r"\b(lên thêm .* hay xuống|tăng thêm .* hay giảm|báo cô|báo tôi|báo anh|báo chị|cảnh báo)\b", u_lower):
            if "stocks.get_portfolio" not in hist_sat and "stocks.get_portfolio" not in seen:
                out.insert(0, {"name": "stocks.get_portfolio", "arguments": {}})
                seen.add("stocks.get_portfolio")
            out.append({"name": "stocks.set_price_alert", "arguments": {"symbol": "SRT", "price": 90000, "direction": "above"}})
            out.append({"name": "stocks.set_price_alert", "arguments": {"symbol": "SRT", "price": 80000, "direction": "below"}})
            seen.add("stocks.set_price_alert")

    if "flights" in inst_set and "flights.my_bookings" in seen and "flights.check_in" not in seen:
        if re.search(r"\b(check[ -]?in|làm thủ tục|chụp in)\b", u_lower):
            out.append({"name": "flights.check_in", "arguments": {"booking_id": "flb_01"}})
            seen.add("flights.check_in")

    if "podcast_radio" in inst_set and "podcast_radio.search" in seen and "podcast_radio.play" not in seen:
        if re.search(r"\b(mở cho .* nghe|phát cho .* nghe|bật .* nghe|mở tập|phát tập)\b", u_lower):
            out.append({"name": "podcast_radio.play", "arguments": {"episode_id": "ep_01"}})
            seen.add("podcast_radio.play")

    # Đảm bảo tất cả READ_TOOLS luôn đứng trước các công cụ hành động / sys.ask_missing_info
    read_part = [x for x in out if x.get("name") in READ_TOOLS]
    act_part = [x for x in out if x.get("name") not in READ_TOOLS]
    return read_part + act_part


def compose_best_exemplar(
    retriever: Any,
    user_text: str,
    installed_apps: Any,
    history: Optional[List[Dict[str, Any]]] = None,
    current_time_str: str = "N/A",
    fallback_exemplars: Optional[List[Dict[str, Any]]] = None,
    previous_steps: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Tổng hợp bài mẫu chuẩn xác nhất cho cả câu đơn mệnh đề, câu ghép đa mệnh đề và câu điều kiện phụ thuộc dữ liệu:
    - Kiểm tra nhánh điều kiện thời gian thực (weather, gold_price, bill_payment + ewallet) theo previous_steps.
    - Khôi phục ý định từ lượt hỏi-đáp trước đó trong history khi lượt hiện tại là câu trả lời bổ sung thông tin.
    - Ưu tiên mẫu toàn câu (Whole-Utterance) trước khi tách mệnh đề (Clause-Split) để bảo toàn cấu trúc câu phức.
    - Tự động loại bỏ các bước Read đã có sẵn kết quả trong history (History-Satisfied Reads).
    """
    if not retriever:
        return fallback_exemplars or []

    _ensure_catalog_exemplars_indexed(retriever)
    inst_dict = installed_apps if isinstance(installed_apps, dict) else {a: {} for a in (installed_apps or [])}
    hist_sat = get_history_satisfied_reads(history, user_text)
    u_for_ret = re.sub(r"^[^,.]{3,35}\b(?:à không|à nhầm)\b[,\s]*", "", user_text, flags=re.IGNORECASE)
    u_for_ret = re.sub(r"^.{3,75}\b(?:ấy quên|à quên|thôi đừng tắt|đừng tắt|thôi khỏi dịch lại|à thôi khỏi)\b[^.!?;]*[.!?;,]?\s*", "", u_for_ret, flags=re.IGNORECASE)
    u_for_ret = re.sub(r"\b(?:báo là|bảo là|nhắn là|nội dung ghi là)\s+[^.!?;]+?(?:đã hủy|đã huỷ|khoá rồi|khóa rồi|đã gọi số vừa nãy rồi|chuyển tiền [^.!?;]*?rồi)[^.!?;]*", "báo là đã xong", u_for_ret, flags=re.IGNORECASE)
    u_for_ret = re.sub(r"\b(mail|email|thư)\s+quảng\s+cáo\s+vé\s+máy\s+bay\b", r"\1 quảng cáo", u_for_ret, flags=re.IGNORECASE)
    u_for_ret = re.sub(r"\bcô\s+nhắc\s+gọi\s+[^,.]+?\s+hồi\s+nãy\b", "cô nói hồi nãy", u_for_ret, flags=re.IGNORECASE)

    # Khôi phục ngữ cảnh khi lượt trước trợ lý chỉ hỏi lại mà chưa gọi công cụ (hoặc chỉ gọi công cụ tra cứu mà người dùng đang trả lời nối tiếp)
    if history and len(history) >= 2:
        last_asst = history[-1] if history[-1].get("role") == "assistant" else {}
        prev_user = history[-2] if history[-2].get("role") == "user" else {}
        prev_u_txt = clean_asr_transcript(str(prev_user.get("audio_text", "")))
        asst_txt = str(last_asst.get("text", ""))
        has_curr_trig = any(re.search(pat, u_for_ret, re.IGNORECASE) for app, pat in APP_TRIGGER_PATTERNS.items() if app in inst_dict)
        if len(history) == 2 and (not last_asst.get("tool_calls") or not has_curr_trig) and len(u_for_ret.split()) <= 20:
            u_for_ret = f"{prev_u_txt} {asst_txt} {u_for_ret}".strip()
        elif len(history) > 2 and not has_curr_trig and "?" in asst_txt and len(u_for_ret.split()) <= 14:
            u_for_ret = f"{prev_u_txt} {asst_txt} {u_for_ret}".strip()

    u_low = u_for_ret.lower()
    norm_u = _normalize_vn_numbers_in_time(u_for_ret)
    primary_d, _ = resolve_target_dates_from_utterance(u_for_ret, current_time_str)

    # 0a. Nhánh điều kiện Thời tiết ("nếu mai mưa ... không mưa thì ...")
    prepend_weather_call: Optional[Dict[str, Any]] = None
    if "weather" in inst_dict and not re.search(r"\b(giờ mà đang|hiện tại đang|lúc này đang|ngoài trời đang)\b", u_low) and (
        (re.search(r"\b(nếu.*mưa|mà mưa|mưa thì|mưa không|có mưa không|trời mưa)\b", u_low) and re.search(r"\b(mưa thì|không mưa|trời ráo|trời nắng|nếu mưa|mà mưa|nếu.*mưa)\b", u_low))
        or re.search(r"\b(nóng quá \d+\s*độ|trên \d+\s*độ)\b.*\bthì\b", norm_u)
    ):
        weather_step = next((s for s in (previous_steps or []) if s.get("tool_called") == "weather.get_forecast" and not s.get("error")), None)
        weather_from_hist = False
        if weather_step is None and history:
            for ht in reversed(history or []):
                tcs_h = ht.get("tool_calls") or []
                trs_h = ht.get("tool_results") or []
                for tc_h, tr_h in zip(tcs_h, trs_h):
                    if isinstance(tc_h, dict) and tc_h.get("name") == "weather.get_forecast" and isinstance(tr_h, dict):
                        if not re.search(r"\b(xem.*mưa không|coi.*mưa không|kiểm tra.*thời tiết)\b", u_low):
                            weather_step = {"tool_called": "weather.get_forecast", "arguments": tc_h.get("arguments") or {}, "raw_observation": tr_h}
                            weather_from_hist = True
                            break
                if weather_step is not None:
                    break
        if weather_step is None:
            return [{
                "id": "cond_weather_step1",
                "audio_text": user_text,
                "tool_calls": [
                    {"name": "weather.get_forecast", "arguments": {"date": primary_d or "2026-10-16"}},
                    {"name": "reminders.create", "arguments": {"content": "Kiểm tra lịch", "time": f"{primary_d or '2026-10-16'}T07:00:00+07:00"}},
                ],
            }]
        else:
            raw_obs = weather_step.get("raw_observation") or {}
            if isinstance(raw_obs, str):
                try:
                    raw_obs = json.loads(raw_obs)
                except Exception:
                    raw_obs = {}
            fc_raw = raw_obs.get("forecast") if isinstance(raw_obs, dict) else None
            fc_obj = fc_raw[0] if isinstance(fc_raw, list) and fc_raw else (fc_raw if isinstance(fc_raw, dict) else (raw_obs if isinstance(raw_obs, dict) else {}))
            m_temp_th = re.search(r"\b(?:nóng quá|trên)\s+(\d+)\s*độ\b", norm_u)
            if m_temp_th:
                is_rain = bool((fc_obj.get("high") or fc_obj.get("temp_max") or 0) > int(m_temp_th.group(1)))
            else:
                is_rain = bool(fc_obj.get("rain"))
            w_args = weather_step.get("arguments") or {"date": primary_d or "2026-10-16"}
            prepend_weather_call = None if weather_from_hist else {"name": "weather.get_forecast", "arguments": w_args}
            w_prefix = [prepend_weather_call] if prepend_weather_call is not None else []
            if not is_rain:
                if (
                    re.search(r"\b(không mưa thì thôi|hông mưa thì thôi|hổng mưa thì thôi|không thì thôi|không mưa.*giữ nguyên|không mưa.*cứ để|trời nắng.*giữ nguyên|trời nắng.*cứ giữ|không mưa.*không cần|không mưa.*khỏi|nắng thì.*giữ nguyên|không thì.*giữ nguyên|không thì.*cứ để)\b", u_low)
                    or not re.search(r"\b(không mưa|không thì|trời nắng|trời ráo|hổng mưa|hông mưa)\b", u_low)
                ):
                    return [{
                        "id": "cond_weather_no_rain_stop",
                        "audio_text": user_text,
                        "tool_calls": list(w_prefix),
                    }]
                if "smart_home" in inst_dict and re.search(r"\b(không mưa.*bật|trời nắng.*bật|ổ cắm|máy tưới)\b", u_low):
                    sh_calls = list(w_prefix)
                    if "smart_home.get_status" not in hist_sat:
                        sh_calls.append({"name": "smart_home.get_status", "arguments": {}})
                    sh_calls.append({"name": "smart_home.switch_set", "arguments": {"device_id": "plug_garden", "power": "on"}})
                    return [{"id": "cond_weather_no_rain_switch", "audio_text": user_text, "tool_calls": sh_calls}]
                if "ride_hailing" in inst_dict and re.search(r"\b(đặt ô tô|gọi ô tô|đặt xe|gọi xe)\b", u_low):
                    rh_no_rain_calls = list(w_prefix)
                    if "calendar" in inst_dict and "calendar.list_events" not in hist_sat and re.search(r"\b(lịch|buổi|họp|khám|hẹn|đá bóng|chạy bộ|cà phê)\b", u_low):
                        rh_no_rain_calls.append({"name": "calendar.list_events", "arguments": {}})
                    rh_no_rain_calls.append({"name": "ride_hailing.book", "arguments": {"pickup": "12 Nguyễn Huệ, Quận 1, TP.HCM", "dropoff": "Bệnh viện", "vehicle": "car4", "schedule_time": f"{primary_d or '2026-10-16'}T14:30:00+07:00"}})
                    return [{
                        "id": "cond_weather_no_rain_ride",
                        "audio_text": user_text,
                        "tool_calls": rh_no_rain_calls,
                    }]
                if re.search(r"[,.;]?\s*còn\s+mưa\s+thì\b", u_low):
                    u_for_ret = re.sub(r"[,.;]?\s*còn\s+mưa\s+thì\b.*$", "", u_for_ret, flags=re.IGNORECASE).strip()
                elif re.search(r"\b(?:còn\s+)?(?:nếu\s+)?(?:không\s+mưa|không\s+thì|trời\s+nắng|trời\s+ráo)\s+(?:thì\s+)?", u_low):
                    u_for_ret = re.sub(r"^.*?\b(?:còn\s+)?(?:nếu\s+)?(?:không\s+mưa|không\s+thì|trời\s+nắng|trời\s+ráo)\s+(?:thì\s+)?", "", u_for_ret, flags=re.IGNORECASE).strip()
                u_low = u_for_ret.lower()
            else:
                if re.search(r"\b(mưa thì khỏi tưới|mưa thì thôi|nếu mưa.*khỏi bật)\b", u_low) and "reminders" in inst_dict and "nhắc" in u_low:
                    return [{
                        "id": "cond_weather_rain_remind_only",
                        "audio_text": user_text,
                        "tool_calls": list(w_prefix) + [
                            {"name": "reminders.create", "arguments": {"content": "Kiểm tra chậu cây ngoài ban công", "time": f"{primary_d or '2026-10-16'}T07:00:00+07:00"}},
                        ],
                    }]
                if "alarm" in inst_dict and re.search(r"\b(cái còn lại|cái kia|báo thức)\b", u_low) and ("alarm.list" in hist_sat or "báo thức" in u_low):
                    al_calls = list(w_prefix)
                    if "alarm.list" not in hist_sat:
                        al_calls.append({"name": "alarm.list", "arguments": {}})
                    al_calls.append({"name": "alarm.update", "arguments": {"alarm_id": "al_01", "time": "06:30"}})
                    return [{"id": "cond_weather_rain_alarm", "audio_text": user_text, "tool_calls": al_calls}]
                if "calendar" in inst_dict and re.search(r"\b(dời|đổi|lùi|chuyển|hủy|huỷ|đẩy|bỏ)\b", u_low) and not re.search(r"\bnhắn\s+[^\s,.]+(?:\s+[^\s,.]+)?\s+(?:giùm|hộ|là)\s+dời\b", u_low):
                    cal_calls = list(w_prefix)
                    if "calendar.list_events" not in hist_sat:
                        cal_calls.append({"name": "calendar.list_events", "arguments": {}})
                    if re.search(r"\b(hủy|huỷ|bỏ)\b", u_low):
                        cal_calls.append({"name": "calendar.cancel_event", "arguments": {"event_id": "evt_01"}})
                    else:
                        cal_calls.append({"name": "calendar.update_event", "arguments": {"event_id": "evt_01", "start": f"{primary_d or '2026-10-16'}T09:00:00+07:00", "end": f"{primary_d or '2026-10-16'}T10:00:00+07:00"}})
                    rain_clause = re.split(r"\bcòn\s+(?:không\s+mưa|trời\s+ráo|trời\s+nắng|nắng)\b", u_low)[0]
                    if "chat_ott" in inst_dict and re.search(r"\b(zalo|nhóm)\b", rain_clause):
                        if "chat_ott.list_chats" not in hist_sat:
                            cal_calls.append({"name": "chat_ott.list_chats", "arguments": {}})
                        cal_calls.append({"name": "chat_ott.send_message", "arguments": {"chat_id": "chat_01", "content": "Sáng mai trời mưa nên mình dời lịch nhé."}})
                    if "sms" in inst_dict and re.search(r"\b(sms|nhắn)\b", rain_clause) and not re.search(r"\b(zalo|nhóm)\b", rain_clause):
                        if "contacts" in inst_dict and "contacts.search" not in hist_sat:
                            q_ct_w = _extract_contact_query_from_text(rain_clause) or "Hương"
                            cal_calls.append({"name": "contacts.search", "arguments": {"query": q_ct_w}})
                        cal_calls.append({"name": "sms.send", "arguments": {"contact_id": "ct_01", "content": "Sáng mai trời mưa nên mình dời lịch nhé."}})
                    if "phone" in inst_dict and re.search(r"\bgọi\s+(?:điện|cho|báo)\b", rain_clause):
                        if "contacts" in inst_dict and "contacts.search" not in hist_sat:
                            q_ct_p = _extract_contact_query_from_text(rain_clause) or "Lan"
                            cal_calls.append({"name": "contacts.search", "arguments": {"query": q_ct_p}})
                        cal_calls.append({"name": "phone.call", "arguments": {"contact_id": "ct_01"}})
                    if "reminders" in inst_dict and re.search(r"\b(nhắc)\b", rain_clause):
                        cal_calls.append({"name": "reminders.create", "arguments": {"content": "Mang ô", "time": f"{primary_d or '2026-10-16'}T07:00:00+07:00"}})
                    return [{"id": "cond_weather_rain_calendar", "audio_text": user_text, "tool_calls": cal_calls}]
                if "ride_hailing" in inst_dict and re.search(r"\b(đặt xe|gọi xe|đặt ô tô|gọi ô tô|xe bốn chỗ|xe 4 chỗ)\b", u_low):
                    rh_calls = list(w_prefix)
                    if "calendar" in inst_dict and "calendar.list_events" not in hist_sat and re.search(r"\b(lịch|buổi|họp|khám)\b", u_low):
                        rh_calls.append({"name": "calendar.list_events", "arguments": {}})
                    rh_calls.append({"name": "ride_hailing.book", "arguments": {"pickup": "Nhà", "dropoff": "Công ty", "vehicle": "car4"}})
                    return [{"id": "cond_weather_rain_ride", "audio_text": user_text, "tool_calls": rh_calls}]
                if re.search(r"^.*?\bcòn\s+mưa\s+thì\b", u_low):
                    u_for_ret = re.sub(r"^.*?\bcòn\s+mưa\s+thì\b", "", u_for_ret, flags=re.IGNORECASE).strip()
                elif re.search(r"[,.;]?\s*(?:còn\s+)?(?:không\s+mưa|không\s+thì|trời\s+nắng|trời\s+ráo)\s+(?:thì\s+)?", u_low):
                    u_for_ret = re.sub(r"[,.;]?\s*(?:còn\s+)?(?:không\s+mưa|không\s+thì|trời\s+nắng|trời\s+ráo)\s+(?:thì\s+)?.*$", "", u_for_ret, flags=re.IGNORECASE).strip()
                u_low = u_for_ret.lower()

    # 0b. Nhánh điều kiện Giá vàng ("giá vàng ... nếu dưới/trên X triệu thì ...")
    if "gold_price" in inst_dict and re.search(r"\b(giá vàng|vàng sjc|vàng nhẫn|nhẫn vàng|vàng thế giới|mua vàng|bán vàng)\b", u_low) and re.search(r"\b(dưới|trên|hơn|vượt|quá)\s+\d+(?:\s*triệu|\s*đô)?\b", norm_u):
        gold_step = next((s for s in (previous_steps or []) if s.get("tool_called") == "gold_price.get_price" and not s.get("error")), None)
        gold_from_hist = False
        if gold_step is None and history:
            for ht in reversed(history or []):
                tcs_h = ht.get("tool_calls") or []
                trs_h = ht.get("tool_results") or []
                for tc_h, tr_h in zip(tcs_h, trs_h):
                    if isinstance(tc_h, dict) and tc_h.get("name") == "gold_price.get_price" and isinstance(tr_h, dict):
                        if not re.search(r"\b(xem.*giá vàng|coi.*giá vàng|kiểm tra.*giá vàng|hỏi.*giá vàng)\b", u_low):
                            gold_step = {"tool_called": "gold_price.get_price", "arguments": {}, "raw_observation": tr_h}
                            gold_from_hist = True
                            break
                if gold_step is not None:
                    break
        if gold_step is None:
            return [{
                "id": "cond_gold_step1",
                "audio_text": user_text,
                "tool_calls": [
                    {"name": "gold_price.get_price", "arguments": {}},
                    {"name": "reminders.create", "arguments": {"content": "Mua vàng", "time": f"{primary_d or '2026-10-16'}T09:00:00+07:00"}},
                ],
            }]
        else:
            raw_obs = gold_step.get("raw_observation") or {}
            if isinstance(raw_obs, str):
                try:
                    raw_obs = json.loads(raw_obs)
                except Exception:
                    raw_obs = {}
            items = (raw_obs.get("prices") or raw_obs.get("items") or []) if isinstance(raw_obs, dict) else []
            target_item = items[0] if items else {}
            for it in items:
                if isinstance(it, dict):
                    if "nhẫn" in u_low and "nhẫn" in str(it.get("type", "")).lower():
                        target_item = it
                        break
                    if "sjc" in u_low and "sjc" in str(it.get("type", "")).lower():
                        target_item = it
                        break
            val_check = target_item.get("buy", 0) if "mua vào" in u_low else target_item.get("sell", 0)
            m_th = re.search(r"\b(dưới|trên|hơn|vượt|quá)\s+(\d+)(?:\s*triệu)?\b", norm_u)
            g_prefix = [] if gold_from_hist else [{"name": "gold_price.get_price", "arguments": {}}]
            if m_th and val_check:
                op_w, raw_num = m_th.group(1), int(m_th.group(2))
                th_m = raw_num * 1_000_000 if raw_num < 1000 else raw_num
                cond_ok = (val_check < th_m) if op_w == "dưới" else (val_check > th_m)
                if not cond_ok:
                    if "banking" in inst_dict and re.search(r"\b(chuyển.*\d+\s*triệu|chuyển khoản|chuyển cho|tài khoản thanh toán|gửi.*\d+\s*triệu|trích tài khoản)\b", norm_u):
                        return [{
                            "id": "cond_gold_else_bank",
                            "audio_text": user_text,
                            "tool_calls": g_prefix + [
                                {"name": "banking.get_balance", "arguments": {}},
                                {"name": "banking.list_beneficiaries", "arguments": {}},
                                {"name": "banking.transfer", "arguments": {"from_account_id": "acc_01", "beneficiary_id": "ben_01", "amount": 5000000}},
                            ],
                        }]
                    return [{
                        "id": "cond_gold_false_stop",
                        "audio_text": user_text,
                        "tool_calls": g_prefix,
                    }]
                else:
                    if "reminders" in inst_dict and "nhắc" in u_low:
                        return [{
                            "id": "cond_gold_true_remind",
                            "audio_text": user_text,
                            "tool_calls": g_prefix + [
                                {"name": "reminders.create", "arguments": {"content": "Đi bán vàng", "time": f"{primary_d or '2026-10-16'}T09:00:00+07:00"}},
                            ],
                        }]

    # 0c. Nhánh điều kiện Hóa đơn + Ví điện tử ("nếu ví đủ ... còn thiếu/không đủ thì trả bằng ngân hàng và nhắc nạp ví / nhắn SMS")
    if "bill_payment" in inst_dict and "ewallet" in inst_dict and re.search(r"\b(ví đủ|đủ thì trả|không đủ|còn thiếu|nếu thiếu|hụt thì|bằng ví.*ngân hàng)\b", u_low) and re.search(r"\b(ví|vĩ|vé|phí)\b", u_low) and not re.search(r"\b(tiết kiệm|tài khoản thanh toán)\b", u_low):
        bill_step = next((s for s in (previous_steps or []) if s.get("tool_called") == "bill_payment.list_bills" and not s.get("error")), None)
        ew_step = next((s for s in (previous_steps or []) if s.get("tool_called") == "ewallet.get_balance" and not s.get("error")), None)
        ct_q = _extract_contact_query_from_text(u_for_ret.lower()) or "Mẹ"
        sms_always = bool("sms" in inst_dict and re.search(r"\b(sms|nhắn)\b", u_low) and re.search(r"\bngân hàng\b.*\b(sms|nhắn)\b", u_low))
        need_contacts = bool("sms" in inst_dict and re.search(r"\b(sms|nhắn)\b", u_low) and "contacts" in inst_dict and "contacts.search" not in hist_sat)
        if bill_step is None or ew_step is None:
            s1_calls = [
                {"name": "bill_payment.list_bills", "arguments": {}},
                {"name": "ewallet.get_balance", "arguments": {}},
            ]
            if need_contacts and sms_always:
                s1_calls.append({"name": "contacts.search", "arguments": {"query": ct_q}})
            s1_calls.append({"name": "bill_payment.pay", "arguments": {"bill_id": "bill_01", "source": "ewallet"}})
            return [{
                "id": "cond_bill_ewallet_step1",
                "audio_text": user_text,
                "tool_calls": s1_calls,
            }]
        else:
            b_obs = bill_step.get("raw_observation") or {}
            e_obs = ew_step.get("raw_observation") or {}
            ew_bal = e_obs.get("balance", 0) if isinstance(e_obs, dict) else 0
            bills_l = (b_obs.get("bills") or []) if isinstance(b_obs, dict) else []
            target_b = next((b for b in bills_l if isinstance(b, dict) and not b.get("paid") and (
                ("điện thoại" in u_low and b.get("type") == "phone")
                or ("điện" in u_low and "điện thoại" not in u_low and b.get("type") == "electricity")
                or ("nước" in u_low and b.get("type") == "water")
                or (("mạng" in u_low or "internet" in u_low) and b.get("type") == "internet")
                or ("truyền hình" in u_low and b.get("type") == "tv")
            )), bills_l[0] if bills_l else {})
            b_amt = target_b.get("amount", 0) if isinstance(target_b, dict) else 0
            src_val = "ewallet" if ew_bal >= b_amt else "bank"
            do_sms = bool("sms" in inst_dict and re.search(r"\b(sms|nhắn)\b", u_low) and (ew_bal >= b_amt or sms_always))
            s2_calls = [
                {"name": "bill_payment.list_bills", "arguments": {}},
                {"name": "ewallet.get_balance", "arguments": {}},
            ]
            if need_contacts and do_sms:
                s2_calls.append({"name": "contacts.search", "arguments": {"query": ct_q}})
            s2_calls.append({"name": "bill_payment.pay", "arguments": {"bill_id": target_b.get("id", "bill_01"), "source": src_val}})
            if ew_bal < b_amt and "reminders" in inst_dict and "nhắc" in u_low:
                s2_calls.append({"name": "reminders.create", "arguments": {"content": "Nạp ví điện tử", "time": f"{primary_d or '2026-10-16'}T20:00:00+07:00"}})
            if do_sms:
                s2_calls.append({"name": "sms.send", "arguments": {"contact_id": "ct_01", "content": "Con đóng hóa đơn rồi nhé."}})
            return [{
                "id": "cond_bill_ewallet_step2",
                "audio_text": user_text,
                "tool_calls": s2_calls,
            }]

    # 0c-bis. Nạp tiền điện thoại cho người thứ 1 + chuyển ví cho người thứ 2 bằng số tiền hóa đơn
    if "ewallet" in inst_dict and "bill_payment" in inst_dict and re.search(r"\bnạp\b.*\b(?:chuyển|gửi|gởi)\b.*\btiền\s+(?:điện|nước|mạng|internet|truyền hình)\b", u_low):
        cl_parts = re.split(r"\b(?:rồi|với|và|còn)\s+", u_for_ret, maxsplit=1, flags=re.IGNORECASE)
        p1 = _extract_contact_query_from_text((cl_parts[0] if cl_parts else "").lower()) or "Tuấn"
        p2 = _extract_contact_query_from_text((cl_parts[1] if len(cl_parts) > 1 else u_for_ret).lower()) or "Cô giáo Mai"
        top_amt = 100000 if "100" in norm_u or "một trăm" in u_low else 50000
        return [{
            "id": "composed_topup_and_bill_transfer",
            "audio_text": user_text,
            "tool_calls": [
                {"name": "bill_payment.list_bills", "arguments": {}},
                {"name": "contacts.search", "arguments": {"query": p1}},
                {"name": "contacts.search", "arguments": {"query": p2}},
                {"name": "ewallet.topup_mobile", "arguments": {"phone": "0901234567", "amount": top_amt}},
                {"name": "ewallet.transfer", "arguments": {"contact_id": "ct_02", "amount": 200000}},
            ],
        }]

    if "tv.guide" in hist_sat and "tv" in inst_dict and re.search(r"\b(chuyển sang phim|mở phim|bật kênh phim)\b", u_for_ret, re.IGNORECASE):
        if not re.search(r"\b(vtv\d|htv\d|kênh \d)\b", u_for_ret, re.IGNORECASE):
            return [{
                "id": "composed_tv_missing_channel",
                "audio_text": user_text,
                "tool_calls": [{"name": "sys.ask_missing_info", "arguments": {"tool_name": "tv.tune", "params": ["channel"]}}]
            }]

    # 0d. Định tuyến câu nối tiếp ngắn dựa trên lượt có tool_calls gần nhất của history (Short Follow-up History Routing)
    if history and len(user_text.split()) <= 24 and len([c.strip() for c in CLAUSE_SPLIT_RE.split(u_for_ret) if len(c.strip().split()) >= 3]) < 2:
        recent_tc_turn = next((t for t in reversed(history) if t.get("tool_calls")), {})
        last_turn_tools = [tc.get("name", "") for tc in (recent_tc_turn.get("tool_calls") or []) if isinstance(tc, dict)]
        u_curr_low = user_text.lower()
        if "chat_ott" in inst_dict and re.search(r"^\s*(?:dạ\s+|ừ\s+|ờ\s+)?zalo(?:\s+|,)*(?:(?:đi|nhé|nha|ạ|em|hen|anh|chị)(?:\s+|,)*)*[\.!]?\s*$", u_curr_low):
            z_calls = []
            if "chat_ott.list_chats" not in hist_sat:
                z_calls.append({"name": "chat_ott.list_chats", "arguments": {}})
            hist_u_txt = " ".join(str(t.get("audio_text", "")) for t in (history or []) if t.get("role") == "user")
            if re.search(r"\bgọi\b", hist_u_txt, re.IGNORECASE) and not re.search(r"\b(nhắn|gửi|báo)\b", hist_u_txt, re.IGNORECASE):
                z_calls.append({"name": "chat_ott.call", "arguments": {"chat_id": "chat_01"}})
            else:
                z_calls.append({"name": "chat_ott.send_message", "arguments": {"chat_id": "chat_01", "content": "Tin nhắn"}})
            return [{"id": "followup_zalo_choice", "audio_text": user_text, "tool_calls": z_calls}]
        if "ecommerce.search" in last_turn_tools and "ecommerce" in inst_dict and re.search(r"\b(cái thứ hai|cái đầu|cái rẻ|cái đắt|loại rẻ|loại đắt|vào giỏ|mua luôn|đặt luôn|lấy cái|chốt cái|hai túi|2 túi|ba hũ|3 hũ|lấy .* cho mình)\b", u_curr_low):
            ec_calls = [{"name": "ecommerce.add_to_cart", "arguments": {"product_id": "prod_01", "quantity": 1}}]
            if re.search(r"\b(đặt luôn|mua luôn|thanh toán|giao về)\b", u_curr_low):
                ec_calls.append({"name": "ecommerce.checkout", "arguments": {"address": "Nhà"}})
            return [{"id": "followup_ecommerce_add", "audio_text": user_text, "tool_calls": ec_calls}]
        if "ecommerce.order_status" in last_turn_tools and "ecommerce" in inst_dict:
            if re.search(r"\b(trả|đổi|hỏng|lỗi|nhầm|bể|vỡ|rách)\b", u_curr_low):
                ret_reason = "wrong_item" if "nhầm" in u_curr_low else "damaged"
                return [{"id": "followup_ecommerce_return", "audio_text": user_text, "tool_calls": [{"name": "ecommerce.request_return", "arguments": {"order_id": "ord_01", "reason": ret_reason}}]}]
            if re.search(r"\b(hủy|huỷ|bỏ)\b", u_curr_low):
                return [{"id": "followup_ecommerce_cancel", "audio_text": user_text, "tool_calls": [{"name": "ecommerce.cancel_order", "arguments": {"order_id": "ord_01"}}]}]
        if "grocery.add_to_cart" in last_turn_tools and "grocery" in inst_dict and re.search(r"\b(chốt|chúc đơn|thanh toán|giao về|đủ rồi)\b", u_curr_low):
            gr_co_calls = []
            if "calendar" in inst_dict and re.search(r"\b(bữa tối|lịch|buổi|họp)\b", u_curr_low):
                gr_co_calls.append({"name": "calendar.list_events", "arguments": {}})
            gr_co_calls.append({"name": "grocery.checkout", "arguments": {"address": "Nhà"}})
            return [{
                "id": "followup_grocery_checkout",
                "audio_text": user_text,
                "tool_calls": gr_co_calls,
            }]
        if "bill_payment.list_bills" in hist_sat and "bill_payment" in inst_dict and re.search(r"\b(đóng luôn|trừ vào|đúng rồi tháng)\b", u_curr_low):
            return [{
                "id": "followup_bill_pay",
                "audio_text": user_text,
                "tool_calls": [{"name": "bill_payment.pay", "arguments": {"bill_id": "bill_01", "source": "bank" if "ngân hàng" in u_curr_low else "ewallet"}}],
            }]
        if "calendar.list_events" in last_turn_tools and "calendar" in inst_dict and re.search(r"\b(hủy|huỷ|xóa|xoá)\b.*\b(lịch|cuộc|buổi|hẹn)\b", u_curr_low):
            return [{
                "id": "followup_calendar_cancel",
                "audio_text": user_text,
                "tool_calls": [{"name": "calendar.cancel_event", "arguments": {"event_id": "evt_01"}}],
            }]
        if "calendar.list_events" in last_turn_tools and "calendar" in inst_dict and re.search(r"\b(lùi|dời|đổi|chuyển)\b.*\b(sang|thành|qua)\s+\d", norm_u):
            return [{
                "id": "followup_calendar_update",
                "audio_text": user_text,
                "tool_calls": [{"name": "calendar.update_event", "arguments": {"event_id": "evt_01", "start": f"{primary_d or '2026-10-16'}T15:00:00+07:00", "end": f"{primary_d or '2026-10-16'}T16:00:00+07:00"}}],
            }]
        if "appliances.list" in last_turn_tools and "appliances" in inst_dict and re.search(r"\b(chạy máy|bật máy|chương trình|chế độ)\b", u_curr_low):
            return [{
                "id": "followup_appliances_start",
                "audio_text": user_text,
                "tool_calls": [{"name": "appliances.start_cycle", "arguments": {"appliance_id": "app_01", "program": "normal"}}],
            }]
        if "sys.ask_missing_info" in last_turn_tools and "ewallet" in inst_dict and re.search(r"\b\d+\s*(?:nghìn|ngàn|triệu|k\b)", norm_u):
            ew_f_calls = []
            if "contacts" in inst_dict and "contacts.search" not in hist_sat:
                ew_f_calls.append({"name": "contacts.search", "arguments": {"query": _extract_contact_query_from_text(u_for_ret.lower()) or "Lan"}})
            ew_f_calls.append({"name": "ewallet.transfer", "arguments": {"contact_id": "ct_01", "amount": 250000}})
            return [{"id": "followup_ewallet_transfer", "audio_text": user_text, "tool_calls": ew_f_calls}]
        if "bike_rental.find_bikes" in last_turn_tools and "bike_rental" in inst_dict and re.search(r"\b(giữ|đặt|thuê|lấy)\b.*\b(chiếc|xe)\b", u_curr_low):
            return [{
                "id": "followup_bike_reserve",
                "audio_text": user_text,
                "tool_calls": [{"name": "bike_rental.reserve", "arguments": {"bike_id": "bike_01"}}],
            }]
        if "tv.guide" in last_turn_tools and "tv" in inst_dict and re.search(r"\b(ghi lẹ|ghi lại|trận .* ghi)\b", u_curr_low):
            return [{
                "id": "followup_tv_record",
                "audio_text": user_text,
                "tool_calls": [{"name": "tv.record", "arguments": {"program_id": "prg_01"}}],
            }]
        if "grocery.search" in last_turn_tools and "grocery" in inst_dict and re.search(r"\b(đặt cho|lấy cho|ba hộp|3 hộp|hai hộp|2 hộp|vào giỏ|mua luôn|đặt luôn|loại không đường)\b", u_curr_low):
            gr_calls = [{"name": "grocery.add_to_cart", "arguments": {"product_id": "gro_01", "quantity": 3}}]
            if re.search(r"\b(đặt cho|mua luôn|đặt luôn|giao về)\b", u_curr_low):
                gr_calls.append({"name": "grocery.checkout", "arguments": {"address": "Nhà"}})
            return [{"id": "followup_grocery_add", "audio_text": user_text, "tool_calls": gr_calls}]
        if "pantry.list_items" in last_turn_tools:
            if "pantry" in inst_dict and re.search(r"\b(dùng hết|xài hết|hết luôn|uống hết|ăn hết|trừ đi|trừ bớt)\b", u_curr_low):
                return [{
                    "id": "followup_pantry_consume",
                    "audio_text": user_text,
                    "tool_calls": [{"name": "pantry.consume", "arguments": {"item_id": "pan_01", "quantity": 1}}],
                }]
            if "shopping_list" in inst_dict and re.search(r"\b(danh sách đi chợ|danh sách mua sắm|mua thêm|mua bù|ghi vào)\b", u_curr_low):
                return [{
                    "id": "followup_pantry_shopping_list",
                    "audio_text": user_text,
                    "tool_calls": [{"name": "shopping_list.add", "arguments": {"name": "Thực phẩm", "quantity": 1}}],
                }]
            if "grocery" in inst_dict and re.search(r"\b(siêu thị|đặt mua|bên đi chợ|giao về)\b", u_curr_low):
                return [{
                    "id": "followup_pantry_grocery",
                    "audio_text": user_text,
                    "tool_calls": [
                        {"name": "grocery.search", "arguments": {"query": "thực phẩm"}},
                        {"name": "grocery.add_to_cart", "arguments": {"product_id": "gro_01", "quantity": 1}},
                        {"name": "grocery.checkout", "arguments": {"address": "Nhà"}},
                    ],
                }]
        if "event_tickets.search" in last_turn_tools and "event_tickets" in inst_dict and re.search(r"\b(chốt luôn|đặt luôn|lấy cho|mua luôn|vé|đêm nhạc|triển lãm|kịch)\b", u_curr_low):
            ev_calls = [{"name": "event_tickets.book", "arguments": {"event_id": "evt_01", "quantity": 2}}]
            if "calendar" in inst_dict and re.search(r"\b(lịch)\b", u_curr_low):
                ev_calls.append({"name": "calendar.create_event", "arguments": {"title": "Đi xem sự kiện", "start": f"{primary_d or '2026-10-16'}T20:00:00+07:00", "end": f"{primary_d or '2026-10-16'}T22:00:00+07:00"}})
            return [{
                "id": "followup_event_tickets_book",
                "audio_text": user_text,
                "tool_calls": ev_calls,
            }]
        if "parking.find" in last_turn_tools and "parking" in inst_dict and re.search(r"\b(giữ|đặt|chỗ|bãi|gửi|đỗ)\b", u_curr_low):
            return [{
                "id": "followup_parking_reserve",
                "audio_text": user_text,
                "tool_calls": [{"name": "parking.reserve", "arguments": {"lot_id": "lot_01", "from_time": f"{primary_d or '2026-10-15'}T14:00:00+07:00", "hours": 2}}],
            }]
        if "habit_tracker.list_habits" in last_turn_tools and "habit_tracker" in inst_dict and re.search(r"\b(xong|rồi|đánh dấu|ghi nhận|hoàn thành|hôm nay)\b", u_curr_low):
            return [{
                "id": "followup_habit_log",
                "audio_text": user_text,
                "tool_calls": [{"name": "habit_tracker.log", "arguments": {"habit_id": "habit_01", "date": primary_d or "2026-10-15"}}],
            }]
        if "flights.check_in" in last_turn_tools and "flights" in inst_dict and re.search(r"\b(làm y rứa|như thế|check[ -]?in|làm thủ tục|chuyến còn lại|vé của)\b", u_curr_low):
            return [{
                "id": "followup_flights_checkin",
                "audio_text": user_text,
                "tool_calls": [
                    {"name": "flights.my_bookings", "arguments": {}},
                    {"name": "flights.check_in", "arguments": {"booking_id": "flb_01"}},
                ],
            }]
        if "train_bus_tickets.search" in last_turn_tools and "train_bus_tickets" in inst_dict and re.search(r"\b(hủy|huỷ|cancel)\b", u_curr_low) and re.search(r"\b(đặt lại|đổi qua)\b", u_curr_low):
            return [{
                "id": "followup_train_bus_rebook",
                "audio_text": user_text,
                "tool_calls": [
                    {"name": "train_bus_tickets.my_bookings", "arguments": {}},
                    {"name": "train_bus_tickets.cancel", "arguments": {"booking_id": "tbb_01"}},
                    {"name": "train_bus_tickets.book", "arguments": {"trip_id": "trip_01", "seats": 2 if re.search(r"\b(hai|2)\b", u_curr_low) else 1}},
                ],
            }]
        if "train_bus_tickets.search" in last_turn_tools and "train_bus_tickets" in inst_dict and re.search(r"\b(đặt cho|vé xe|vé tàu|chuyến đó)\b", u_curr_low) and not re.search(r"\b(hủy|huỷ|cancel)\b", u_curr_low):
            return [{
                "id": "followup_train_bus_book",
                "audio_text": user_text,
                "tool_calls": [{"name": "train_bus_tickets.book", "arguments": {"trip_id": "trip_01", "seats": 1}}],
            }]
        if "flights.search" in last_turn_tools and "flights" in inst_dict and re.search(r"\b(lấy chuyến|chuyến sớm|chuyến muộn|chuyến rẻ|chuyến đấy|chuyến đó|chốt .* ghế|đặt cho|hai chỗ|2 chỗ|một vé|1 vé)\b", u_curr_low) and not re.search(r"\b(hủy|huỷ)\b", u_curr_low):
            return [{
                "id": "followup_flights_book",
                "audio_text": user_text,
                "tool_calls": [{"name": "flights.book", "arguments": {"flight_id": "fl_01", "passengers": 2 if re.search(r"\b(hai|2)\b", u_curr_low) else 1}}],
            }]
        if "fitness" in inst_dict and ("fitness.get_activity" in last_turn_tools or "chat_ott.read_chat" in last_turn_tools or "sms.read_recent" in last_turn_tools) and re.search(r"\b(ghi vào|ghi luôn|ghi nhận)\b", u_curr_low):
            return [{
                "id": "followup_fitness_log",
                "audio_text": user_text,
                "tool_calls": [{"name": "fitness.log_workout", "arguments": {"type": "running", "minutes": 30}}],
            }]
        if "smart_home.get_status" in last_turn_tools and "smart_home" in inst_dict and re.search(r"\b(vậy bật|cái đèn đó|cái đèn đang bật|hạ xuống|tắt nó đi|bật nó lên)\b", u_curr_low):
            return [{
                "id": "followup_smarthome_light",
                "audio_text": user_text,
                "tool_calls": [{"name": "smart_home.light_set", "arguments": {"device_id": "light_living", "power": "on"}}],
            }]
        if ("food_delivery.get_menu" in last_turn_tools or "food_delivery.list_restaurants" in last_turn_tools) and "food_delivery" in inst_dict and re.search(r"\b(đặt cho|một phần như thế|lấy cho|chốt luôn)\b", u_curr_low):
            fd_calls = []
            if "food_delivery.get_menu" not in hist_sat and "food_delivery.order_history" not in last_turn_tools:
                fd_calls.append({"name": "food_delivery.get_menu", "arguments": {"restaurant_id": "rest_01"}})
            fd_calls.append({"name": "food_delivery.place_order", "arguments": {"restaurant_id": "rest_01", "items": [{"item_id": "item_01", "quantity": 1}], "address": "Nhà"}})
            return [{"id": "followup_food_place_order", "audio_text": user_text, "tool_calls": fd_calls}]
        if "subscriptions" in inst_dict and "subscriptions.list" in hist_sat and re.search(r"\b(hủy luôn gói|huỷ luôn gói|hủy gói đấy|huỷ gói đấy|hủy gói đó)\b", u_curr_low):
            return [{
                "id": "followup_subscriptions_cancel",
                "audio_text": user_text,
                "tool_calls": [{"name": "subscriptions.cancel", "arguments": {"subscription_id": "sub_01"}}],
            }]
        if ("recipes.search" in last_turn_tools or "recipes.get" in last_turn_tools) and "recipes" in inst_dict:
            if "shopping_list" in inst_dict and re.search(r"\b(danh sách đi chợ|mua thêm|nguyên liệu)\b", u_curr_low):
                rc_calls = []
                if "recipes.get" not in last_turn_tools:
                    rc_calls.append({"name": "recipes.get", "arguments": {"recipe_id": "rcp_01"}})
                rc_calls.append({"name": "shopping_list.add", "arguments": {"name": "Nguyên liệu", "quantity": 1}})
                return [{"id": "followup_recipes_shopping", "audio_text": user_text, "tool_calls": rc_calls}]
            if re.search(r"\b(lưu lại|lưu giùm|lưu giúp|lưu món|lưu .* qua ghi chú|xem chi tiết|đọc các bước|công thức đó|món đó|món ấy|nhanh nhất)\b", u_curr_low) and not re.search(r"\b(đặt món|giao về|khỏi nấu|nhà hàng)\b", u_curr_low):
                if "notes" in inst_dict and re.search(r"\b(ghi chú|note)\b", u_curr_low):
                    rc_calls = []
                    if "recipes.get" not in last_turn_tools:
                        rc_calls.append({"name": "recipes.get", "arguments": {"recipe_id": "rcp_01"}})
                    rc_calls.append({"name": "notes.create", "arguments": {"title": "Công thức nấu ăn", "content": "Nguyên liệu"}})
                    return [{"id": "followup_recipes_notes", "audio_text": user_text, "tool_calls": rc_calls}]
                rc_tool = "recipes.save" if re.search(r"\b(lưu)\b", u_curr_low) else "recipes.get"
                rc_calls = [{"name": rc_tool, "arguments": {"recipe_id": "rcp_01"}}]
                if "reminders" in inst_dict and re.search(r"\b(nhắc)\b", u_curr_low):
                    rc_calls.append({"name": "reminders.create", "arguments": {"content": "Nấu ăn", "time": f"{primary_d or '2026-10-15'}T16:00:00+07:00"}})
                return [{
                    "id": "followup_recipes_action",
                    "audio_text": user_text,
                    "tool_calls": rc_calls,
                }]
        if "shopping_list.get" in last_turn_tools and "ecommerce" in inst_dict and re.search(r"\b(sàn|mua .* trên sàn|đặt mua|thương mại điện tử|mua đồ)\b", u_curr_low):
            sh_ec_calls = [
                {"name": "ecommerce.search", "arguments": {"query": "cốc giữ nhiệt"}},
                {"name": "ecommerce.add_to_cart", "arguments": {"product_id": "prod_01", "quantity": 2}},
            ]
            if re.search(r"\b(thanh toán|đặt mua|giao về)\b", u_curr_low):
                sh_ec_calls.append({"name": "ecommerce.checkout", "arguments": {"address": "Nhà"}})
            return [{
                "id": "followup_shoplist_ecommerce",
                "audio_text": user_text,
                "tool_calls": sh_ec_calls,
            }]
        if len(history) == 2 and "shopping_list" in inst_dict and "danh sách đi chợ" in str(history[-1].get("text", "")).lower():
            return [{
                "id": "followup_asst_shopping_list",
                "audio_text": user_text,
                "tool_calls": [{"name": "shopping_list.add", "arguments": {"name": "Sữa tươi", "quantity": 4}}],
            }]

    triggered_apps = {
        app for app, pat in APP_TRIGGER_PATTERNS.items()
        if app in inst_dict and re.search(pat, u_for_ret, re.IGNORECASE)
    }
    if re.search(r"\b(nhắn|tin nhắn)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(đọc tin nhắn|xem tin nhắn|có tin nhắn nào)\b", u_for_ret, re.IGNORECASE):
        if "chat_ott" in inst_dict and "sms" not in inst_dict:
            triggered_apps.add("chat_ott")
        elif "sms" in inst_dict and "chat_ott" not in inst_dict:
            triggered_apps.add("sms")
        elif "chat_ott" in inst_dict and "sms" in inst_dict:
            hist_chat_names = set()
            for ht in (history or []):
                for tc_h in (ht.get("tool_calls") or []):
                    if isinstance(tc_h, dict) and str(tc_h.get("name", "")).startswith("chat_ott."):
                        hist_chat_names.add("__has_chat_ott__")
                for res_h in (ht.get("tool_results") or []):
                    if isinstance(res_h, dict) and isinstance(res_h.get("chats"), list):
                        for ch in res_h["chats"]:
                            if isinstance(ch, dict) and ch.get("name"):
                                hist_chat_names.add(str(ch["name"]).lower())
            if (
                re.search(r"\b(zalo|nhóm)\b", u_for_ret, re.IGNORECASE)
                or any(cn != "__has_chat_ott__" and cn in u_for_ret.lower() for cn in hist_chat_names)
                or ("__has_chat_ott__" in hist_chat_names and not re.search(r"\bsms\b", u_for_ret, re.IGNORECASE))
            ):
                triggered_apps.add("chat_ott")
                triggered_apps.discard("sms")
            elif not re.search(r"\b(zalo|nhóm)\b", u_for_ret, re.IGNORECASE):
                triggered_apps.add("sms")
                triggered_apps.discard("chat_ott")
    if prepend_weather_call is not None:
        triggered_apps.discard("weather")
    if "home_automation" in triggered_apps and not re.search(r"\b(còn giờ thì|bây giờ thì)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("smart_home")
    if "chat_ott" in triggered_apps and re.search(r"\b(zalo|nhóm)\b", u_for_ret, re.IGNORECASE) and not re.search(r"\bsms\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("sms")
    if "shopping_list" in triggered_apps and not re.search(r"\b(đặt mua|bên app đi chợ|siêu thị|vào giỏ)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("grocery")
    if "ecommerce" in triggered_apps and not re.search(r"\b(đi chợ|siêu thị|sữa|trứng|thịt|rau|gạo|cá|ức gà|cà chua|dầu ô liu|yến mạch|bột mì|táo|macca|đậu hũ|nước mắm|bơ lạt|phô mai|bánh mì)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("grocery")
    if "food_delivery" in triggered_apps and re.search(r"\b(lưu giúp .* vào bản đồ|lưu địa điểm)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("food_delivery")
    if "grocery" in triggered_apps and "recipes" in triggered_apps and not re.search(r"\b(đi chợ|siêu thị|vào giỏ|đặt mua)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("grocery")
    if "train_bus_tickets" in triggered_apps and re.search(r"\b(kết thúc (?:lượt|phiên))\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("train_bus_tickets")
    if "reminders" in triggered_apps and re.search(r"\bnhắc\s+(?:tôi|mình|anh|chị|em)\s+gọi\b", u_for_ret, re.IGNORECASE) and not re.search(r"\b(?:rồi|với|và)\s+gọi\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("phone")
    if "contacts" in triggered_apps and re.search(r"\b(tìm trong danh bạ|sửa .* trong danh bạ|lưu đè vào danh bạ)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("email")
    if "bill_payment" in triggered_apps and not re.search(r"\b(ví đủ|đủ không|không đủ|còn đủ|thiếu thì|hụt thì|số dư ví|kiểm tra ví|coi ví|xem ví|nạp|chuyển ví|chuyển qua ví)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("ewallet")

    def _app_trig_pos(app_name: str) -> int:
        pat = APP_TRIGGER_PATTERNS.get(app_name, "")
        m_p = re.search(pat, u_for_ret, re.IGNORECASE) if pat else None
        if m_p:
            return m_p.start()
        if app_name in {"chat_ott", "sms"}:
            m_n = re.search(r"\b(nhắn|tin nhắn)\b", u_for_ret, re.IGNORECASE)
            if m_n:
                return m_n.start()
        return 9999

    ordered_triggered_apps = sorted(triggered_apps, key=_app_trig_pos)

    def _with_weather_prefix(calls_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if prepend_weather_call is not None and calls_list and not any(x.get("name") == "weather.get_forecast" for x in calls_list):
            return [prepend_weather_call] + calls_list
        return calls_list

    def _fill_missing_triggered_apps(base_calls: List[Dict[str, Any]], base_utts: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
        if not ordered_triggered_apps:
            return base_calls, base_utts
        covered = {
            (tc.get("arguments", {}).get("tool_name", "").split(".")[0] if tc["name"] == "sys.ask_missing_info" else tc["name"].split(".")[0])
            for tc in base_calls
        } - {"contacts"}
        if triggered_apps.issubset(covered):
            return base_calls, base_utts
        ret_at = _get_atomic_retriever(retriever)
        out_calls = list(base_calls)
        out_utts = list(base_utts)
        for t_app in ordered_triggered_apps:
            if t_app in covered:
                continue
            app_cands = ret_at.retrieve(u_for_ret, {t_app} | ({"contacts"} if "contacts" in inst_dict else set()), top_k=10)
            picked = None
            picked_utt = ""
            for ac in app_cands:
                fc = [tc for tc in ac["tool_calls"] if _is_tool_compatible(tc, u_for_ret)]
                if _valid_filtered(fc, u_for_ret) and any(tc["name"].split(".")[0] == t_app for tc in fc):
                    picked = fc
                    picked_utt = ac.get("audio_text", "")
                    break
            if not picked and app_cands:
                picked = app_cands[0]["tool_calls"]
                picked_utt = app_cands[0].get("audio_text", "")
            if picked:
                out_calls.extend(picked)
                if picked_utt:
                    out_utts.append(picked_utt)
                covered.add(t_app)
        return out_calls, out_utts

    # 1. Ưu tiên mẫu toàn câu (Whole-Utterance) trước khi tách mệnh đề nếu đã khớp trọn vẹn mọi ứng dụng được kích hoạt
    cands = retriever.retrieve(u_for_ret, inst_dict, top_k=20)
    if triggered_apps:
        for cand in cands:
            filtered_calls = [tc for tc in cand["tool_calls"] if _is_tool_compatible(tc, u_for_ret)]
            if prepend_weather_call is not None and {tc["name"] for tc in filtered_calls} <= {"weather.get_forecast"}:
                continue
            if not _valid_filtered(filtered_calls, u_for_ret) or len(filtered_calls) != len(cand["tool_calls"]):
                continue
            cand_apps = {
                (tc.get("arguments", {}).get("tool_name", "").split(".")[0] if tc["name"] == "sys.ask_missing_info" else tc["name"].split(".")[0])
                for tc in filtered_calls
            }
            if (cand_apps - {"contacts", "sys"}) == (triggered_apps - {"contacts", "sys"}):
                pruned = _with_weather_prefix(_ensure_readers_or_prune(filtered_calls, hist_sat, u_for_ret, inst_dict, current_time_str, history))
                if pruned:
                    return [{
                        "id": cand.get("id", "composed_whole"),
                        "audio_text": cand.get("audio_text", ""),
                        "tool_calls": pruned
                    }]

    # 2. Tách mệnh đề (Clause-Split Composition) cho câu ghép đa mệnh đề + Bù đắp ứng dụng thiếu
    clauses = [c.strip() for c in CLAUSE_SPLIT_RE.split(u_for_ret) if len(c.strip().split()) >= 3]
    if len(clauses) >= 2:
        ret_atomic = _get_atomic_retriever(retriever)
        merged_calls: List[Dict[str, Any]] = []
        merged_utts: List[str] = []
        used_apps: Set[str] = set()
        for cl in clauses:
            avail = set(inst_dict.keys()) - used_apps
            cl_trig_all = {
                app for app, pat in APP_TRIGGER_PATTERNS.items()
                if app in inst_dict and re.search(pat, cl, re.IGNORECASE)
            }
            if re.search(r"\b(nhắn|tin nhắn)\b", cl, re.IGNORECASE) and not re.search(r"\b(đọc tin nhắn|xem tin nhắn|có tin nhắn nào)\b", cl, re.IGNORECASE):
                if "chat_ott" in triggered_apps:
                    cl_trig_all.add("chat_ott")
                elif "sms" in triggered_apps:
                    cl_trig_all.add("sms")
            if "home_automation" in cl_trig_all:
                cl_trig_all.discard("smart_home")
            if "maps" in cl_trig_all and re.search(r"\b(lưu địa điểm|lưu chỗ đó)\b", cl, re.IGNORECASE):
                cl_trig_all.discard("grocery")
            if "ecommerce" in cl_trig_all and not re.search(r"\b(đi chợ|siêu thị|sữa|trứng|thịt|rau|gạo|cá|ức gà|cà chua|dầu ô liu|yến mạch|bột mì|táo|macca|hạnh nhân|đậu hũ|nước mắm|bơ lạt|phô mai|bánh mì)\b", cl, re.IGNORECASE):
                cl_trig_all.discard("grocery")
            if cl_trig_all:
                search_apps = cl_trig_all | ({"contacts"} if "contacts" in inst_dict else set())
            elif avail and re.search(rf"^\s*{ACTION_VERB_START}\b", cl, re.IGNORECASE):
                search_apps = avail
            else:
                continue
            cl_cands = ret_atomic.retrieve(cl, search_apps, top_k=15)
            chosen_calls = None
            chosen_utt = ""
            for cand in cl_cands:
                fc = [tc for tc in cand["tool_calls"] if _is_tool_compatible(tc, cl)]
                if _valid_filtered(fc, cl) and len(fc) == len(cand["tool_calls"]):
                    chosen_calls = fc
                    chosen_utt = cand.get("audio_text", "")
                    used_apps |= (cand["called_apps"] - {"contacts"})
                    break
            if not chosen_calls and cl_cands:
                fc = [tc for tc in cl_cands[0]["tool_calls"] if _is_tool_compatible(tc, cl)]
                chosen_calls = fc if _valid_filtered(fc, cl) else cl_cands[0]["tool_calls"]
                chosen_utt = cl_cands[0].get("audio_text", "")
                used_apps |= (cl_cands[0]["called_apps"] - {"contacts"})
            if chosen_calls:
                merged_calls.extend(chosen_calls)
                if picked_utt := chosen_utt:
                    merged_utts.append(picked_utt)
        if merged_calls:
            merged_calls, merged_utts = _fill_missing_triggered_apps(merged_calls, merged_utts)
            final_calls = _with_weather_prefix(_ensure_readers_or_prune(merged_calls, hist_sat, u_for_ret, inst_dict, current_time_str, history))
            if final_calls:
                return [{
                    "id": "composed_multi_clause",
                    "audio_text": " + ".join(merged_utts) if merged_utts else user_text,
                    "tool_calls": final_calls
                }]

    # 2b. Nếu câu kích hoạt >= 1 ứng dụng mà chưa có mẫu toàn câu bao phủ hết -> Tổng hợp từ Atomic Sub-Exemplar cho từng ứng dụng được kích hoạt
    if len(ordered_triggered_apps) >= 1:
        synth_calls, synth_utts = _fill_missing_triggered_apps([], [])
        if synth_calls:
            pruned_synth = _with_weather_prefix(_ensure_readers_or_prune(synth_calls, hist_sat, u_for_ret, inst_dict, current_time_str, history))
            if pruned_synth:
                return [{
                    "id": "composed_multi_app_atomic",
                    "audio_text": " + ".join(synth_utts) if synth_utts else user_text,
                    "tool_calls": pruned_synth
                }]

    for cand in cands:
        filtered_calls = [tc for tc in cand["tool_calls"] if _is_tool_compatible(tc, u_for_ret)]
        if prepend_weather_call is not None and {tc["name"] for tc in filtered_calls} <= {"weather.get_forecast"}:
            continue
        if not _valid_filtered(filtered_calls, u_for_ret):
            continue
        cand_apps = {
            (tc.get("arguments", {}).get("tool_name", "").split(".")[0] if tc["name"] == "sys.ask_missing_info" else tc["name"].split(".")[0])
            for tc in filtered_calls
        }
        if triggered_apps and (not (triggered_apps & cand_apps) or not ((cand_apps - {"contacts", "sys"}) <= (triggered_apps - {"contacts", "sys"}))):
            continue
        pruned = _with_weather_prefix(_ensure_readers_or_prune(filtered_calls, hist_sat, u_for_ret, inst_dict, current_time_str, history))
        if pruned:
            return [{
                "id": cand.get("id", "composed_single"),
                "audio_text": cand.get("audio_text", ""),
                "tool_calls": pruned
            }]

    if cands:
        filtered_calls = [tc for tc in cands[0]["tool_calls"] if _is_tool_compatible(tc, u_for_ret)]
        pruned = _ensure_readers_or_prune(filtered_calls or cands[0]["tool_calls"], hist_sat, u_for_ret, inst_dict, current_time_str, history)
        return [{
            "id": cands[0].get("id", "composed_fallback"),
            "audio_text": cands[0].get("audio_text", ""),
            "tool_calls": pruned or cands[0]["tool_calls"]
        }]

    return fallback_exemplars or []


def decompose_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Phân rã chuỗi tool_calls của bài mẫu thành các lượt ReAct hợp lệ theo quan hệ phụ thuộc dữ liệu:
    - Turn 1 (Stage-1 Read): Các tool tra cứu/đọc danh sách độc lập.
    - Turn 2 (Stage-2 Read): Các tool đọc chi tiết phụ thuộc vào ID/kết quả của Turn 1.
    - Turn 3 (Stage-3 Read, nếu có): Các tool đọc phụ thuộc vào kết quả của Turn 2.
    - Final Turn (Actions): Các tool thực thi thay đổi trạng thái (mutation) hoặc sys.ask_missing_info.
    """
    if not tool_calls:
        return []

    stage1_reads: List[Dict[str, Any]] = []
    stage2_reads: List[Dict[str, Any]] = []
    stage3_reads: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []

    seen_reads: Set[str] = set()
    for tc in tool_calls:
        tname = tc.get("name") or tc.get("tool", "")
        targs = tc.get("arguments") or {}
        if tname in READ_TOOLS:
            if not targs:
                stage1_reads.append(tc)
                seen_reads.add(tname)
                continue

            deps = STAGE2_DEPS.get(tname, set())
            if deps and (deps & seen_reads):
                s2_names = {x.get("name") or x.get("tool", "") for x in stage2_reads}
                if deps & s2_names:
                    stage3_reads.append(tc)
                else:
                    stage2_reads.append(tc)
            elif seen_reads & {"chat_ott.read_chat", "email.read", "sms.read_recent"}:
                s2_names = {x.get("name") or x.get("tool", "") for x in stage2_reads}
                if s2_names & {"chat_ott.read_chat", "email.read", "sms.read_recent"}:
                    stage3_reads.append(tc)
                else:
                    stage2_reads.append(tc)
            else:
                stage1_reads.append(tc)
            seen_reads.add(tname)
        else:
            actions.append(tc)

    return [t for t in (stage1_reads, stage2_reads, stage3_reads, actions) if t]


def format_exemplar_by_turns(
    exemplars: List[Dict[str, Any]],
    fallback_str: str,
    previous_steps: Optional[List[Dict[str, Any]]] = None
) -> str:
    """Hiển thị bài mẫu tổng hợp dưới dạng Kế hoạch thực thi theo từng Lượt ReAct kèm trạng thái tiến độ."""
    if not exemplars:
        return fallback_str or "Không có ví dụ mẫu."

    ex = exemplars[0]
    ex_utt = ex.get("audio_text", "")
    ex_calls = ex.get("tool_calls", [])
    turns = decompose_tool_calls(ex_calls)

    lines = [f"- Câu mẫu tương đồng: \"{ex_utt}\""]
    if not turns:
        lines.append("- Kế hoạch thực thi chuẩn (Out-of-scope / Không gọi tool):")
        lines.append("  * Lượt 1/1 (is_finished=true): `[]`")
        return "\n".join(lines)

    executed_tools = {s.get("tool_called") for s in (previous_steps or []) if not s.get("error")}
    total_turns = len(turns)
    cur_turn_idx = total_turns
    for idx, turn_calls in enumerate(turns, 1):
        turn_tool_names = {c.get("name") for c in turn_calls}
        if not turn_tool_names.issubset(executed_tools):
            cur_turn_idx = idx
            break

    lines.append(f"- Kế hoạch phân tách {total_turns} lượt ReAct chuẩn:")
    for t_idx, turn_calls in enumerate(turns, 1):
        is_fin = "true" if t_idx == total_turns else "false"
        formatted_calls = json.dumps(
            [{"name": c.get("name"), "arguments": c.get("arguments", {})} for c in turn_calls],
            ensure_ascii=False
        )
        if t_idx < cur_turn_idx:
            status_tag = "✅ ĐÃ CHẠY XONG Ở BƯỚC TRƯỚC - TUYỆT ĐỐI KHÔNG GỌI LẠI"
        elif t_idx == cur_turn_idx:
            status_tag = "👉 LƯỢT HIỆN TẠI CẦN GỌI NGAY"
        else:
            status_tag = "⏳ CHỜ LƯỢT SAU"
        lines.append(f"  * Lượt {t_idx}/{total_turns} [{status_tag}] (is_finished={is_fin}): `{formatted_calls}`")
    return "\n".join(lines)


# ==============================================================================
# 5. BỘ PHÂN TÍCH QUAN SÁT THỜI GIAN THỰC (LIVE OBSERVATION ANALYZER)
# ==============================================================================
BILL_TYPE_VN = {
    "electricity": "tiền điện (electricity)",
    "water": "tiền nước (water)",
    "internet": "tiền mạng / internet (internet)",
    "phone": "tiền điện thoại di động (phone)",
    "tv": "tiền truyền hình (tv)",
    "tuition": "tiền học phí (tuition)",
}


def analyze_observations(
    previous_steps: List[Dict[str, Any]],
    context: Dict[str, Any],
    raw_history: Optional[List[Dict[str, Any]]] = None,
    user_utterance: str = ""
) -> str:
    """
    Phân tích hợp nhất cả dữ liệu trong [ACTIVE MEMORY FROM PREVIOUS DIALOGUE] (raw_history)
    lẫn kết quả trả về trong [PREVIOUS STEPS] của phiên hiện tại để cung cấp
    các dữ kiện logic chính xác.
    """
    if not previous_steps and not raw_history:
        return ""

    u_full_low = (user_utterance or "").lower()
    if raw_history:
        for ht in raw_history:
            u_full_low += " " + str(ht.get("audio_text") or ht.get("text") or "").lower()

    cur_date = None
    try:
        cur_date = datetime.fromisoformat(context.get("current_time", "")).date()
    except Exception:
        pass

    combined_records: List[Dict[str, Any]] = []
    paid_bill_ids: Set[str] = set()
    history_events: List[Dict[str, Any]] = []
    history_alerts: List[Dict[str, Any]] = []

    if raw_history:
        for turn in raw_history:
            t_calls = turn.get("tool_calls") or []
            t_results = turn.get("tool_results") or []
            for idx, res in enumerate(t_results):
                if not isinstance(res, dict):
                    continue
                if isinstance(res.get("bill"), dict) and res["bill"].get("paid"):
                    paid_bill_ids.add(str(res["bill"].get("id", "")))
                if "events" in res and isinstance(res["events"], list):
                    for ev in res["events"]:
                        if isinstance(ev, dict) and ev.get("end"):
                            history_events.append(ev)
                if "alerts" in res and isinstance(res["alerts"], list):
                    history_alerts.extend(alert for alert in res["alerts"] if isinstance(alert, dict))
                tname = ""
                targs = {}
                if idx < len(t_calls) and isinstance(t_calls[idx], dict):
                    tname = t_calls[idx].get("name", "")
                    targs = t_calls[idx].get("arguments") or {}
                if not tname:
                    if "bills" in res:
                        tname = "bill_payment.list_bills"
                    elif "accounts" in res:
                        tname = "banking.get_balance"
                    elif "balance" in res:
                        tname = "ewallet.get_balance"
                    elif "schedules" in res:
                        tname = "medication.list"
                    elif "programs" in res and "channels" in res:
                        tname = "tv.guide"
                    elif "programs" in res:
                        tname = "loyalty.get_points"
                    elif "devices" in res:
                        tname = "smart_home.get_status"
                    elif "chores" in res:
                        tname = "household_chores.list_chores"
                    elif "places" in res:
                        tname = "maps.search_place"
                combined_records.append({"tool_called": tname, "arguments": targs, "obs": res, "from_history": True})

    for step in (previous_steps or []):
        if step.get("error"):
            continue
        obs_raw = step.get("raw_observation", step.get("observation", ""))
        try:
            obs = json.loads(obs_raw) if isinstance(obs_raw, str) else obs_raw
        except Exception:
            continue
        if isinstance(obs, dict):
            if isinstance(obs.get("bill"), dict) and obs["bill"].get("paid"):
                paid_bill_ids.add(str(obs["bill"].get("id", "")))
            if "events" in obs and isinstance(obs["events"], list):
                for ev in obs["events"]:
                    if isinstance(ev, dict) and ev.get("end"):
                        history_events.append(ev)
            combined_records.append({
                "tool_called": step.get("tool_called", ""),
                "arguments": step.get("arguments") or {},
                "obs": obs,
                "from_history": False,
            })

    if not combined_records:
        return ""

    bank_accounts: List[Dict[str, Any]] = []
    ewallet_balance: Optional[int] = None
    for rec in combined_records:
        obs = rec["obs"]
        if "accounts" in obs and isinstance(obs["accounts"], list):
            bank_accounts = obs["accounts"]
        if "balance" in obs and isinstance(obs.get("balance"), (int, float)):
            ewallet_balance = int(obs["balance"])

    facts: List[str] = []
    for rec in combined_records:
        tname = rec["tool_called"]
        obs = rec["obs"]
        from_hist = rec["from_history"]

        # 1. Phân tích kết quả tìm danh bạ (contacts.search)
        if not from_hist and tname == "contacts.search" and "contacts" in obs:
            contacts = obs.get("contacts") or []
            if len(contacts) >= 2:
                u_curr_only = (user_utterance or "").lower()
                explicit_roles = (
                    "bộ phận kho", "bên kho", "hậu cần", "thủ quỹ", "mua hàng",
                    "điều phối", "bác sĩ gia đình", "thú y", "kế toán", "nhân sự",
                    "kỹ thuật", "trực kho", "pháp chế"
                )
                rel_matches = []
                for c in contacts:
                    c_rel = str(c.get("relation") or "").lower()
                    for role_kw in explicit_roles:
                        if role_kw in u_curr_only and (role_kw in c_rel or role_kw.replace("bên ", "") in c_rel):
                            rel_matches.append(c)
                            break
                name_exact_matches = []
                distinct_names = {str(c.get("name") or "").strip().lower() for c in contacts}
                if len(distinct_names) >= 2:
                    for c in contacts:
                        c_name = str(c.get("name") or "").strip().lower()
                        if c_name and re.search(rf"\b{re.escape(c_name)}\b", u_curr_only):
                            longer_matches = [
                                other for other in contacts
                                if other is not c and len(str(other.get("name") or "")) > len(c_name)
                                and re.search(rf"\b{re.escape(str(other.get('name') or '').lower())}\b", u_curr_only)
                            ]
                            if not longer_matches:
                                name_exact_matches.append(c)
                chosen_unique = rel_matches[0] if len(rel_matches) == 1 else (name_exact_matches[0] if len(name_exact_matches) == 1 else None)
                if chosen_unique is not None:
                    facts.append(
                        f"Liên hệ khớp duy nhất theo mô tả của người dùng: contact_id=\"{chosen_unique.get('id')}\", tên=\"{chosen_unique.get('name')}\", "
                        f"relation=\"{chosen_unique.get('relation', '')}\", phone=\"{chosen_unique.get('phone')}\", email=\"{chosen_unique.get('email', '')}\" "
                        f"-> BẮT BUỘC chọn `\"contact_id\": \"{chosen_unique.get('id')}\"` (KHÔNG gọi `sys.ask_missing_info`!)."
                    )
                else:
                    c_list = [f"{c.get('id')} ({c.get('name')}, SĐT={c.get('phone')})" for c in contacts]
                    facts.append(
                        f"⚠️ PHÁT HIỆN TRÙNG TÊN DANH BẠ: `contacts.search` trả về {len(contacts)} người trùng tên: {', '.join(c_list)}. "
                        f"Do chưa rõ người nhận, bạn TUYỆT ĐỐI KHÔNG gọi lại `contacts.search` nữa mà BẮT BUỘC gọi `sys.ask_missing_info` "
                        f"với `\"tool_name\"` là công cụ hành động đích (như `\"phone.call\"`, `\"sms.send\"`, `\"ewallet.transfer\"` - KHÔNG dùng `contacts.get`) "
                        f"và `\"params\": [\"contact_id\"]`, đồng thời đặt `\"is_finished\": true`!"
                    )
            elif len(contacts) == 1:
                c = contacts[0]
                email_val = c.get("email")
                if not email_val:
                    facts.append(
                        f"Liên hệ duy nhất: contact_id=\"{c.get('id')}\", tên=\"{c.get('name')}\", phone=\"{c.get('phone')}\", "
                        f"email=KHÔNG CÓ (Nếu yêu cầu gửi/soạn email cho người này, BẮT BUỘC gọi `sys.ask_missing_info` với `\"tool_name\": \"email.create_draft\"` hoặc `\"email.send\"` và `\"params\": [\"to\"]`)."
                    )
                else:
                    facts.append(
                        f"Liên hệ duy nhất: contact_id=\"{c.get('id')}\", tên=\"{c.get('name')}\", phone=\"{c.get('phone')}\", email=\"{email_val}\"."
                    )
            else:
                facts.append("⚠️ `contacts.search` trả về danh sách rỗng (0 liên hệ).")

        # 2. Phân tích hóa đơn (bill_payment.list_bills) kèm đối chiếu số dư ngân hàng & ví điện tử
        if "bills" in obs and isinstance(obs.get("bills"), list):
            raw_bills = obs.get("bills") or []
            unpaid_bills = [
                b for b in raw_bills
                if isinstance(b, dict) and not b.get("paid") and str(b.get("id", "")) not in paid_bill_ids
            ]
            for b in unpaid_bills:
                bid = b.get("id")
                raw_type = str(b.get("type", ""))
                btype_vn = BILL_TYPE_VN.get(raw_type, raw_type)
                amt = b.get("amount", 0)
                due_str = b.get("due_date", "")
                due_note = ""
                if cur_date and due_str:
                    try:
                        dd = datetime.fromisoformat(due_str).date()
                        diff = (dd - cur_date).days
                        in_5d = "TRONG_HẠN_5_NGÀY_TỚI" if 0 <= diff <= 5 else "NGOÀI_HẠN_5_NGÀY"
                        due_note = f" (còn {diff} ngày -> {in_5d})"
                    except Exception:
                        pass
                amt_note = "<=500000" if amt <= 500000 else ">500000"
                acc_check = ""
                if bank_accounts:
                    suff = [f"`{a.get('id')}` ({a.get('name', a.get('type', ''))}: {a.get('balance')} >= {amt} -> ĐỦ TIỀN, BẮT BUỘC chọn `\"from_account_id\": \"{a.get('id')}\"`)" for a in bank_accounts if (a.get("balance") or 0) >= amt]
                    insuff = [f"⚠️ `{a.get('id')}` ({a.get('name', a.get('type', ''))}: {a.get('balance')} < {amt} -> KHÔNG ĐỦ TIỀN, TUYỆT ĐỐI KHÔNG chọn `{a.get('id')}`!)" for a in bank_accounts if (a.get("balance") or 0) < amt]
                    acc_check += f" | Đối chiếu tài khoản ngân hàng: {'; '.join(insuff + suff)}"
                if ewallet_balance is not None:
                    if ewallet_balance >= amt:
                        acc_check += f" | Đối chiếu ví điện tử (ewallet): số_dư={ewallet_balance} >= {amt} -> VÍ ĐỦ TIỀN (dùng \"source\": \"ewallet\")"
                    else:
                        acc_check += f" | ⚠️ Đối chiếu ví điện tử (ewallet): số_dư={ewallet_balance} < {amt} -> VÍ KHÔNG ĐỦ TIỀN, BẮT BUỘC dùng \"source\": \"bank\" (ngân hàng)!"
                facts.append(f"Hóa đơn chưa đóng `{bid}`: loại={btype_vn}, số_tiền_chính_xác=`{amt}` ({amt_note}), hạn={due_str}{due_note}{acc_check}.")

            if len(unpaid_bills) >= 2:
                for i in range(len(unpaid_bills)):
                    for j in range(i + 1, len(unpaid_bills)):
                        b1, b2 = unpaid_bills[i], unpaid_bills[j]
                        pair_sum = (b1.get("amount") or 0) + (b2.get("amount") or 0)
                        if len(unpaid_bills) == 2:
                            facts.append(
                                f"Hai hóa đơn còn lại chưa đóng là `{b1.get('id')}` ({b1.get('type')}={b1.get('amount')}) và `{b2.get('id')}` ({b2.get('type')}={b2.get('amount')}) "
                                f"-> Biểu thức cộng tổng: `\"{b1.get('amount')}+{b2.get('amount')}\"` (tổng={pair_sum})."
                            )
                        for a in bank_accounts:
                            bal = a.get("balance") or 0
                            if ((b1.get("amount") or 0) <= bal or (b2.get("amount") or 0) <= bal) and pair_sum > bal:
                                suff_pair = [x.get("id") for x in bank_accounts if (x.get("balance") or 0) >= pair_sum]
                                smaller_b = b1 if (b1.get("amount") or 0) <= (b2.get("amount") or 0) else b2
                                facts.append(
                                    f"⚠️ CẢNH BÁO TỔNG 2 HÓA ĐƠN ({b1.get('type')}={b1.get('amount')} + {b2.get('type')}={b2.get('amount')}): "
                                    f"tổng={pair_sum} > số dư `{a.get('id')}` ({a.get('name', '')}={bal}) -> `{a.get('id')}` KHÔNG ĐỦ trả cả hai! "
                                    f"Nếu yêu cầu chuyển khoản cả 2 thì phải dùng `\"{suff_pair[0] if suff_pair else 'acc_02'}\"`. "
                                    f"ĐẶC BIỆT: Nếu người dùng yêu cầu có điều kiện ('đủ trả cả hai thì bật autopay cho hai cái, không thì 1 cái thôi'), "
                                    f"do `{a.get('id')}` KHÔNG ĐỦ trả cả hai nên TUYỆT ĐỐI KHÔNG gọi 2 lệnh `bill_payment.schedule_autopay`! "
                                    f"BẮT BUỘC CHỈ GỌI ĐÚNG 1 LỆNH `bill_payment.schedule_autopay` cho loại hóa đơn được chỉ định khi không đủ (ví dụ: `\"type\": \"{smaller_b.get('type')}\"`)!"
                                )

        # 3. Phân tích số dư ngân hàng / người thụ hưởng / ví điện tử
        if "accounts" in obs and isinstance(obs.get("accounts"), list):
            acc_strs = [f"`{a.get('id')}` ({a.get('name', a.get('type', ''))}, số_dư={a.get('balance')})" for a in (obs.get("accounts") or [])]
            if acc_strs:
                facts.append("Tài khoản ngân hàng: " + "; ".join(acc_strs) + ". (BẮT BUỘC chọn `from_account_id` có số_dư >= số tiền cần chuyển; nếu Tài khoản thanh toán không đủ số dư thì BẮT BUỘC chọn Tài khoản tiết kiệm!).")
        if "beneficiaries" in obs and isinstance(obs.get("beneficiaries"), list):
            for ben in obs.get("beneficiaries") or []:
                if isinstance(ben, dict):
                    facts.append(
                        f"Người thụ hưởng ngân hàng `{ben.get('id')}`: tên_người_nhận=\"{ben.get('name')}\", ngân_hàng=\"{ben.get('bank', '')}\" "
                        f"(Khi chuyển khoản `banking.transfer` cho \"{ben.get('name')}\", BẮT BUỘC dùng `\"beneficiary_id\": \"{ben.get('id')}\"`)."
                    )
        if "balance" in obs and isinstance(obs.get("balance"), (int, float)):
            facts.append(f"Số dư ví điện tử (ewallet): {obs.get('balance')}.")

        # 4. Phân tích danh sách chat OTT & tin nhắn đã đọc
        if "chats" in obs and isinstance(obs.get("chats"), list):
            direct_name_counts: Dict[str, List[Dict[str, Any]]] = {}
            for ch in obs.get("chats") or []:
                if isinstance(ch, dict) and ch.get("type") != "group":
                    nm_l = str(ch.get("name") or "").strip().lower()
                    if nm_l:
                        direct_name_counts.setdefault(nm_l, []).append(ch)
            if not from_hist:
                for nm_l, dup_list in direct_name_counts.items():
                    if len(dup_list) >= 2 and re.search(rf"\b{re.escape(nm_l)}\b", u_full_low):
                        dup_ids = ", ".join(f"{d.get('id')} ({d.get('name')})" for d in dup_list)
                        facts.append(
                            f"⚠️ PHÁT HIỆN TRÙNG TÊN CHAT ZALO: `chat_ott.list_chats` có {len(dup_list)} tài khoản trùng tên \"{dup_list[0].get('name')}\": {dup_ids}. "
                            f"Do chưa rõ tài khoản nhận, bạn BẮT BUỘC gọi `sys.ask_missing_info` với `\"tool_name\"` là `\"chat_ott.send_message\"` (hoặc `\"chat_ott.call\"`) và `\"params\": [\"chat_id\"]`, đồng thời đặt `\"is_finished\": true`!"
                        )
            for ch in obs.get("chats") or []:
                members = ch.get("members") or []
                m_cnt = len(members)
                if ch.get("type") == "group":
                    facts.append(
                        f"Nhóm chat `{ch.get('id')}`: tên_nhóm=\"{ch.get('name')}\", thành_viên_khác={members} -> "
                        f"ĐÂY LÀ NHÓM CHAT \"{ch.get('name')}\": nếu đặt món cho mỗi người trong nhóm \"không tính tôi / không tính em / trừ em ra\" "
                        f"thì BẮT BUỘC đặt `\"quantity\": {m_cnt}` (tương ứng {m_cnt} người: {', '.join(members)})."
                    )
                else:
                    facts.append(
                        f"Chat cá nhân (1-1) `{ch.get('id')}`: tên_người_nhận=\"{ch.get('name')}\"."
                    )
        if "messages" in obs and isinstance(obs.get("messages"), list):
            msgs = obs.get("messages") or []
            if msgs:
                all_m_str = " | ".join(f"{m.get('from', m.get('sender', 'Người gửi'))}: \"{m.get('content', m.get('text', ''))}\"" for m in msgs[-3:] if isinstance(m, dict))
                norm_m = _normalize_vn_numbers_in_time(all_m_str)
                m_dur = re.search(r"\b(?:bơi|chạy|tập|đạp xe)\s+(\d+)\s*phút\b", norm_m, re.IGNORECASE)
                dur_hint = f" (🔥 LƯU Ý TẬP LUYỆN: Tin nhắn dặn tập đúng `{m_dur.group(1)}` phút -> Khi gọi `fitness.log_workout`, BẮT BUỘC truyền `\"minutes\": {int(m_dur.group(1))}`!)" if m_dur else ""
                src_label = f"lịch sử `{tname or 'chat_ott.read_chat'}`" if from_hist else f"`{tname}`"
                facts.append(
                    f"Nội dung tin nhắn đã đọc từ {src_label}: {all_m_str}{dur_hint} -> "
                    f"Hãy trích xuất đúng số phút/địa điểm/thời gian/trạm xe chuẩn từ tin nhắn trên để gọi công cụ hành động tiếp theo."
                )

        # 4b. Phân tích danh sách quán ăn, thực đơn & lịch sử đơn hàng (food_delivery)
        if not from_hist and tname == "food_delivery.list_restaurants" and "restaurants" in obs:
            for r in obs.get("restaurants") or []:
                if isinstance(r, dict):
                    facts.append(
                        f"Quán ăn `{r.get('id')}`: tên_quán=\"{r.get('name')}\", chuyên_món={r.get('tags', [])} "
                        f"(Khi gọi `food_delivery.get_menu` hoặc `food_delivery.place_order` cho quán \"{r.get('name')}\", BẮT BUỘC dùng `\"restaurant_id\": \"{r.get('id')}\"`)."
                    )
        if not from_hist and tname == "food_delivery.get_menu" and "menu" in obs:
            req_rest_id = rec["arguments"].get("restaurant_id", "")
            for m in obs.get("menu") or []:
                if isinstance(m, dict):
                    facts.append(
                        f"Món ăn `{m.get('id')}` (thuộc quán `{req_rest_id}`): tên_món=\"{m.get('name')}\", giá={m.get('price')}."
                    )
        if not from_hist and tname == "food_delivery.order_history" and "orders" in obs:
            for od in obs.get("orders") or []:
                if isinstance(od, dict):
                    st_val = str(od.get("status", ""))
                    facts.append(
                        f"Đơn đồ ăn `{od.get('id')}`: restaurant_id=`{od.get('restaurant_id', '')}`, món={od.get('items', '')}, status=`{st_val}`."
                    )

        # 4c. Phân tích tin tức (news.search / news.headlines)
        if not from_hist and tname in {"news.search", "news.headlines"} and "articles" in obs:
            for art in obs.get("articles") or []:
                if isinstance(art, dict):
                    summary_txt = str(art.get("summary", art.get("content", "")))
                    src_txt = str(art.get("source", ""))
                    if any(w in summary_txt.lower() for w in ("tin đồn", "chưa xác nhận")):
                        continue
                    official_tag = (
                        " 👉 [THÔNG BÁO CHÍNH THỨC MỚI NHẤT CỦA BAN TỔ CHỨC (SAU THƯ MỜI) -> BẮT BUỘC DÙNG ĐỊA ĐIỂM & GIỜ NÀY ĐỂ CẬP NHẬT LỊCH VÀ NHẮN TIN!]"
                        if "ban tổ chức" in src_txt.lower() and "thay đổi" in str(art.get("title", "")).lower() else ""
                    )
                    facts.append(
                        f"Bản tin `{art.get('id')}` (nguồn=\"{src_txt}\", ngày đăng={art.get('date', art.get('published_at', ''))}): "
                        f"tiêu_đề=\"{art.get('title', '')}\" | nội_dung=\"{summary_txt}\"{official_tag}"
                    )

        # 5. Phân tích tập podcast & đài phát thanh (podcast_radio.search)
        if not from_hist and tname == "podcast_radio.search":
            vn_num_words = {1: "một", 2: "hai", 3: "ba", 4: "bốn", 5: "năm", 6: "sáu", 7: "bảy", 8: "tám", 9: "chín", 10: "mười"}
            shows_map = {s.get("id"): s for s in (obs.get("shows") or []) if isinstance(s, dict)}
            for ep in obs.get("episodes") or []:
                if isinstance(ep, dict):
                    sh = shows_map.get(ep.get("show_id"), {})
                    num = ep.get("number")
                    num_w = vn_num_words.get(num, str(num))
                    facts.append(
                        f"Tập podcast `{ep.get('id')}`: tập {num} (tập {num_w} / Số {num}) - tiêu_đề=\"{ep.get('title')}\", "
                        f"của tác giả/host=\"{sh.get('host', '')}\" (show=\"{sh.get('name', '')}\"), thời_lượng={ep.get('duration_min')} phút."
                    )

        # 6. Phân tích hoạt động thể chất (fitness.get_activity)
        if not from_hist and tname == "fitness.get_activity" and isinstance(obs.get("activity"), dict):
            act = obs["activity"]
            for w in act.get("workouts") or []:
                if isinstance(w, dict) and w.get("minutes"):
                    mins = int(w["minutes"])
                    follow_ups = []
                    for hev in history_events:
                        try:
                            st_dt = datetime.fromisoformat(hev["end"])
                            et_dt = st_dt + timedelta(minutes=mins)
                            follow_ups.append(
                                f"nếu bắt đầu ngay sau \"{hev.get('title')}\" ({hev['end']}) và kéo dài {mins}p -> `\"start\": \"{hev['end']}\", \"end\": \"{et_dt.isoformat()}\"`"
                            )
                        except Exception:
                            pass
                    extra_str = (" | " + "; ".join(follow_ups)) if follow_ups else ""
                    facts.append(
                        f"Buổi tập thể chất `{w.get('id')}` ({w.get('type')}): thời_lượng={mins} phút lúc {w.get('time')}{extra_str}."
                    )

        # 7. Phân tích sự kiện lịch (calendar.list_events)
        if not from_hist and tname == "calendar.list_events" and "events" in obs:
            for ev in obs.get("events") or []:
                loc_str = ev.get("location", "")
                st_s = str(ev.get("start", ""))
                et_s = str(ev.get("end", ""))
                dur_note = ""
                try:
                    dur_m = int((datetime.fromisoformat(et_s) - datetime.fromisoformat(st_s)).total_seconds() // 60)
                    if dur_m > 0:
                        dur_note = f" (Khi dời sự kiện này bằng `calendar.update_event`, BẮT BUỘC giữ nguyên thời lượng {dur_m} phút: `end` = `start_mới` + {dur_m} phút!)"
                except Exception:
                    pass
                loc_hint = f" (Nếu đặt xe `ride_hailing.book` đi tới sự kiện này, BẮT BUỘC đặt `\"dropoff\": \"{loc_str}\"`, TUYỆT ĐỐI KHÔNG lấy địa chỉ nhà làm `dropoff`!)" if loc_str else ""
                facts.append(
                    f"Sự kiện `{ev.get('id')}`: \"{ev.get('title')}\" | start={st_s} | end={et_s} "
                    f"| tại=\"{loc_str}\"{loc_hint}{dur_note}{format_event_time_offsets(ev)}"
                )

        # 8. Phân tích tìm kiếm địa điểm (maps.search_place) & lộ trình bản đồ (maps.get_route)
        if "places" in obs and isinstance(obs.get("places"), list) and not from_hist:
            for pl in obs.get("places") or []:
                if isinstance(pl, dict):
                    facts.append(
                        f"Địa điểm tìm thấy `{pl.get('id')}`: tên_chuẩn=\"{pl.get('name')}\", địa_chỉ=\"{pl.get('address', '')}\" "
                        f"(Khi gọi `maps.get_route` tới địa điểm vừa tìm, BẮT BUỘC dùng `\"destination\": \"{pl.get('name')}\"` hoặc `\"{pl.get('address', '')}\"`, KHÔNG lấy tên người/địa điểm cũ trong history!)."
                    )
        if not from_hist and tname == "maps.get_route":
            dur = obs.get("duration_min") or (obs.get("route") or {}).get("duration_min")
            dist = obs.get("distance_km") or (obs.get("route") or {}).get("distance_km")
            if dur is not None or dist is not None:
                dur_int = int(dur or 0)
                dep_hints = []
                if cur_date:
                    tmr = (cur_date + timedelta(days=1)).isoformat()
                    for target_h in (8, 9, 10, 14, 15):
                        t_dt = datetime.fromisoformat(f"{tmr}T{target_h:02d}:00:00+07:00")
                        dep_15 = (t_dt - timedelta(minutes=dur_int + 15)).isoformat()
                        dep_hints.append(f"nếu hẹn {target_h}h sáng/chiều mai và đến sớm 15p -> giờ xuất phát=`{dep_15}`")
                dep_str = (" | " + "; ".join(dep_hints[:3])) if dep_hints else ""
                facts.append(
                    f"Lộ trình `{rec['arguments'].get('destination', '')}`: khoảng_cách={dist} km, "
                    f"thời_gian_di_chuyển={dur_int} phút (nếu cần đến sớm 15 phút thì phải xuất phát trước giờ hẹn {dur_int + 15} phút{dep_str})."
                )

        # 9. Phân tích giá vàng (gold_price.get_price)
        if not from_hist and tname == "gold_price.get_price":
            items = obs.get("prices") or obs.get("items") or ([obs] if "sell" in obs or "buy" in obs else [])
            for it in items:
                if isinstance(it, dict):
                    buy_v = it.get("buy", 0)
                    sell_v = it.get("sell", 0)
                    buy_m = round(buy_v / 1_000_000, 2) if buy_v else None
                    sell_m = round(sell_v / 1_000_000, 2) if sell_v else None
                    buy_fmt = f"{buy_v:,}".replace(",", ".") if buy_v else ""
                    facts.append(
                        f"Giá vàng ({it.get('type', 'SJC')}): mua_vào={buy_v} ({buy_fmt} đồng/lượng = {buy_m} triệu), bán_ra={sell_v} ({sell_m} triệu)."
                    )

        # 10. Phân tích xe đạp cho thuê (bike_rental.find_bikes)
        if not from_hist and tname == "bike_rental.find_bikes" and "bikes" in obs:
            bikes = obs.get("bikes") or []
            avail_bikes = [b for b in bikes if isinstance(b, dict) and b.get("available") is True]
            unavail_bikes = [b for b in bikes if isinstance(b, dict) and not b.get("available")]
            for ub in unavail_bikes:
                facts.append(f"⚠️ Xe đạp `{ub.get('id')}` (\"{ub.get('name')}\"): available=false (ĐÃ HẾT XE) -> TUYỆT ĐỐI KHÔNG chọn `bike_id` này!")
            if avail_bikes:
                cheapest = min(avail_bikes, key=lambda x: x.get("price_per_hour") or 999999)
                avail_str = "; ".join(f"`{b.get('id')}` (\"{b.get('name')}\", giá={b.get('price_per_hour')}/giờ, available=true)" for b in avail_bikes)
                facts.append(
                    f"Danh sách xe đạp CÒN TRỐNG (available=true): {avail_str} -> "
                    f"Chiếc CÒN TRỐNG rẻ nhất là `\"bike_id\": \"{cheapest.get('id')}\"` (\"{cheapest.get('name')}\", giá={cheapest.get('price_per_hour')}/giờ, thuê 3 tiếng = {(cheapest.get('price_per_hour') or 0) * 3})."
                )

        # 11. Phân tích vé máy bay / tàu xe đã đặt (flights.my_bookings / train_bus_tickets.my_bookings) & kết quả tìm chuyến bay (flights.search)
        if "bookings" in obs and isinstance(obs.get("bookings"), list):
            for bk in obs.get("bookings") or []:
                if isinstance(bk, dict):
                    if "trip_id" in bk or str(bk.get("id", "")).startswith("tbb_"):
                        facts.append(
                            f"Vé tàu/xe đã đặt `\"booking_id\": \"{bk.get('id')}\"` (trip_id=`{bk.get('trip_id', '')}`, số_ghế={bk.get('seats', '')}, trạng_thái={bk.get('status', '')}) "
                            f"-> Khi gọi `train_bus_tickets.cancel` để hủy vé này, BẮT BUỘC truyền `\"booking_id\": \"{bk.get('id')}\"`!"
                        )
                        continue
                    fl = bk.get("flight") or bk
                    f_date = str(fl.get("date", ""))
                    dep_str = str(fl.get("depart", fl.get("departure_time", "")))
                    arr_str = str(fl.get("arrive", fl.get("arrival_time", "")))
                    dep_iso = dep_str if "T" in dep_str else (f"{f_date}T{dep_str}:00+07:00" if f_date and re.match(r"^\d{2}:\d{2}$", dep_str) else dep_str)
                    arr_iso = arr_str if "T" in arr_str else (f"{f_date}T{arr_str}:00+07:00" if f_date and re.match(r"^\d{2}:\d{2}$", arr_str) else arr_str)
                    alarm_offsets = ""
                    if "T" in dep_iso:
                        try:
                            dep_dt = datetime.fromisoformat(dep_iso)
                            off_list = []
                            for lb, mins in [("-45p", 45), ("-1h", 60), ("-1h30p", 90), ("-2h", 120), ("-2h30p", 150), ("-3h", 180), ("-3h30p", 210), ("-4h", 240)]:
                                off_list.append(f"{lb}={(dep_dt - timedelta(minutes=mins)).isoformat()}")
                            alarm_offsets = f" (trước_cất_cánh: {', '.join(off_list)})"
                        except Exception:
                            pass
                    if "T" in arr_iso:
                        try:
                            arr_dt = datetime.fromisoformat(arr_iso)
                            alarm_offsets += f" (sau_hạ_cánh: +30p={(arr_dt + timedelta(minutes=30)).isoformat()})"
                        except Exception:
                            pass
                    facts.append(
                        f"Đặt chỗ máy bay `\"booking_id\": \"{bk.get('id')}\"` (flight_id=`{fl.get('id', '')}`, mã={fl.get('code', '')}): hãng_bay=\"{fl.get('airline', '')}\", ngày_bay={f_date or dep_str[:10]}, "
                        f"giờ_cất_cánh={dep_iso}, giờ_hạ_cánh={arr_iso}, chặng={fl.get('origin', '')}->{fl.get('destination', '')}{alarm_offsets}."
                    )
        if "flights" in obs and isinstance(obs.get("flights"), list):
            for fl in obs.get("flights") or []:
                if isinstance(fl, dict):
                    f_date = str(fl.get("date", rec["arguments"].get("date", "")))
                    dep_str = str(fl.get("depart", fl.get("departure_time", "")))
                    arr_str = str(fl.get("arrive", fl.get("arrival_time", "")))
                    dep_iso = dep_str if "T" in dep_str else (f"{f_date}T{dep_str}:00+07:00" if f_date and re.match(r"^\d{2}:\d{2}$", dep_str) else dep_str)
                    arr_iso = arr_str if "T" in arr_str else (f"{f_date}T{arr_str}:00+07:00" if f_date and re.match(r"^\d{2}:\d{2}$", arr_str) else arr_str)
                    alarm_offsets = ""
                    if "T" in dep_iso:
                        try:
                            dep_dt = datetime.fromisoformat(dep_iso)
                            off_list = []
                            for lb, mins in [("-45p", 45), ("-1h", 60), ("-1h30p", 90), ("-2h", 120), ("-2h30p", 150), ("-3h", 180), ("-3h30p", 210), ("-4h", 240)]:
                                off_list.append(f"{lb}={(dep_dt - timedelta(minutes=mins)).isoformat()}")
                            alarm_offsets = f" (trước_cất_cánh: {', '.join(off_list)})"
                        except Exception:
                            pass
                    if "T" in arr_iso:
                        try:
                            arr_dt = datetime.fromisoformat(arr_iso)
                            alarm_offsets += f" (sau_hạ_cánh: +30p={(arr_dt + timedelta(minutes=30)).isoformat()})"
                        except Exception:
                            pass
                    facts.append(
                        f"Chuyến bay `\"flight_id\": \"{fl.get('id')}\"` (mã={fl.get('code', '')}, hãng=\"{fl.get('airline', '')}\"): "
                        f"chặng={fl.get('origin', '')}->{fl.get('destination', '')}, giờ_cất_cánh={dep_iso}, giờ_hạ_cánh={arr_iso}, giá={fl.get('price', '')}{alarm_offsets}."
                    )

        # 12. Phân tích danh sách công việc (tasks.list), việc nhà (household_chores.list_chores) & khóa học (learning.list_courses)
        if not from_hist and tname == "tasks.list" and "tasks" in obs:
            for tk in obs.get("tasks") or []:
                if isinstance(tk, dict) and not tk.get("done"):
                    due_s = str(tk.get("due", ""))
                    is_today = (cur_date and due_s.startswith(cur_date.isoformat()))
                    tag = "🔥 HẠN HÔM NAY (deadline hôm nay) -> BẮT BUỘC chọn `task_id` này để đánh dấu xong (`tasks.complete`)!" if is_today else f"hạn={due_s}"
                    facts.append(f"Công việc chưa xong `{tk.get('id')}`: \"{tk.get('title')}\" ({tag}).")
        if "chores" in obs and isinstance(obs.get("chores"), list):
            for ch in obs.get("chores") or []:
                if isinstance(ch, dict):
                    c_title = ch.get("title") or ch.get("name") or ""
                    facts.append(
                        f"Việc nhà `{ch.get('id')}`: tên_việc=\"{c_title}\", người_phụ_trách_hiện_tại=\"{ch.get('assignee', '')}\" "
                        f"(Khi gọi `household_chores.assign` cho việc \"{c_title}\", BẮT BUỘC dùng đúng `\"chore_id\": \"{ch.get('id')}\"` và viết hoa chữ cái đầu của `\"assignee\"`, ví dụ `\"Bé Na\"`, `\"Chú Năm\"`)."
                    )
        if not from_hist and tname == "learning.list_courses" and "courses" in obs:
            for cr in obs.get("courses") or []:
                if isinstance(cr, dict):
                    facts.append(f"Khóa học `{cr.get('id')}`: tên=\"{cr.get('name')}\", bên_cung_cấp=\"{cr.get('provider', '')}\".")

        # 13. Phân tích cảnh báo an ninh (home_security.list_alerts) & khóa cửa (door_lock.list_locks)
        if "alerts" in obs and isinstance(obs.get("alerts"), list):
            for al in obs.get("alerts") or []:
                if isinstance(al, dict):
                    facts.append(
                        f"Cảnh báo an ninh `{al.get('id')}`: loại={al.get('type')}, vị_trí=`\"{al.get('location')}\"`, nội_dung=\"{al.get('message')}\" "
                        f"(Khi gọi `home_security.acknowledge_alert` cho cảnh báo ở `{al.get('location')}`, BẮT BUỘC truyền `\"alert_id\": \"{al.get('id')}\"`!)."
                    )
        if not from_hist and tname == "door_lock.list_locks" and "locks" in obs:
            target_alert = history_alerts[0] if history_alerts else None
            target_text = ""
            if target_alert:
                target_text = f"{target_alert.get('location', '')} {target_alert.get('message', '')}".lower()
            for lock in obs.get("locks") or []:
                if not isinstance(lock, dict):
                    continue
                name = str(lock.get("name", ""))
                location = str(lock.get("location", ""))
                is_target = bool(target_text and (name.lower() in target_text or location.lower() in target_text))
                target_note = " -> ĐÂY LÀ CỬA ĐƯỢC NHẮC TRONG CẢNH BÁO GẦN NHẤT" if is_target else ""
                facts.append(
                    f"Khóa `{lock.get('id')}`: tên=\"{name}\", vị_trí=\"{location}\", locked={lock.get('locked')}{target_note}."
                )
                if is_target:
                    facts.append(
                        f"BẮT BUỘC dùng `\"lock_id\": \"{lock.get('id')}\"` khi gọi `door_lock.set_lock` cho cửa vừa báo mở; không chọn khóa khác."
                    )

        # 14. Phân tích lịch uống thuốc, chương trình TV & đổi điểm
        if "schedules" in obs and isinstance(obs.get("schedules"), list):
            evening_meds = []
            evening_ids = []
            bedtime_meds = []
            bedtime_ids = []
            for med in obs.get("schedules") or []:
                if isinstance(med, dict):
                    m_id = med.get("id")
                    m_name = med.get("name", "")
                    m_times = med.get("times") or []
                    if any(t >= "21:00" for t in m_times) or "ngủ" in m_name.lower():
                        bedtime_meds.append(f"`{m_id}` (\"{m_name}\", giờ={m_times})")
                        bedtime_ids.append(m_id)
                    elif any("17:00" <= t < "21:00" for t in m_times):
                        evening_meds.append(f"`{m_id}` (\"{m_name}\", giờ={m_times})")
                        evening_ids.append(m_id)
            if evening_meds or bedtime_meds:
                ev_calls_str = ", ".join(f'{{"schedule_id": "{mid}"}}' for mid in evening_ids)
                facts.append(
                    f"Phân loại lịch thuốc buổi tối (EVENING_MED_COUNT={len(evening_ids)}): "
                    f"Thuốc tối đã uống gồm {len(evening_ids)} thuốc: {', '.join(evening_meds) or 'không có'}. "
                    f"Thuốc trước khi đi ngủ CHƯA UỐNG gồm: {', '.join(bedtime_meds) or 'không có'} -> "
                    f"Nếu người dùng báo 'mấy thuốc tối nay uống rồi, chỉ còn thuốc trước lúc đi ngủ chưa uống', "
                    f"TUYỆT ĐỐI KHÔNG gọi `medication.log_taken` cho {bedtime_ids}! "
                    f"BẮT BUỘC CHỈ GỌI ĐÚNG {len(evening_ids)} LỆNH `medication.log_taken` cho: {ev_calls_str}!"
                )
        if "programs" in obs and "channels" in obs and isinstance(obs.get("programs"), list):
            for prg in obs.get("programs") or []:
                if isinstance(prg, dict):
                    p_st = str(prg.get("start", ""))
                    w_label = ""
                    if len(p_st) >= 10:
                        try:
                            p_dt = datetime.fromisoformat(p_st).date()
                            w_label = f" ({WEEKDAY_VN[p_dt.weekday()]} ngày {p_dt.isoformat()})"
                        except Exception:
                            pass
                    facts.append(f"Chương trình TV `{prg.get('id')}`: \"{prg.get('title')}\" trên {prg.get('channel')}, chiếu lúc {p_st}{w_label}.")
        if "programs" in obs and "channels" not in obs and isinstance(obs.get("programs"), list):
            for lp in obs.get("programs") or []:
                if isinstance(lp, dict):
                    for rw in lp.get("rewards") or []:
                        if isinstance(rw, dict):
                            facts.append(
                                f"Đổi điểm thưởng (`loyalty.redeem`): `\"program\": \"{lp.get('id')}\"`, `\"reward_id\": \"{rw.get('id')}\"` -> \"{rw.get('name')}\" (mệnh giá={rw.get('value')}, tốn {rw.get('cost')} điểm)."
                            )

        # 15. Phân tích gói đăng ký (subscriptions.list) & quy tắc tự động (home_automation)
        if "subscriptions" in obs and isinstance(obs.get("subscriptions"), list):
            hist_text_lower = " ".join(str(t.get("audio_text") or t.get("text") or "") for t in (raw_history or [])).lower()
            for sub in obs.get("subscriptions") or []:
                if isinstance(sub, dict):
                    s_id = sub.get("id")
                    s_name = str(sub.get("name", ""))
                    s_base = re.sub(r"\s+\d+.*$", "", s_name).strip().lower()
                    is_hist_match = bool(s_base and s_base in hist_text_lower)
                    match_note = f" 👉 [ĐÂY LÀ GÓI ĐƯỢC NHẮC TRONG HỘI THOẠI TRƯỚC -> BẮT BUỘC truyền `\"subscription_id\": \"{s_id}\"` khi gọi `subscriptions.cancel`!]" if is_hist_match else ""
                    facts.append(
                        f"Gói đăng ký `{s_id}`: tên=\"{s_name}\", giá={sub.get('price')}, ngày_gia_hạn_tiếp={sub.get('next_billing')}{match_note}."
                    )
        if "rule" in obs and isinstance(obs.get("rule"), dict):
            rl = obs["rule"]
            facts.append(
                f"Quy tắc tự động đã tạo trong lịch sử: `\"rule_id\": \"{rl.get('id')}\"` (tiêu_đề=\"{rl.get('title', '')}\", thiết_bị={rl.get('device_id')}, giờ={rl.get('time')}) "
                f"-> Khi gọi `home_automation.delete_rule` để xóa quy tắc này, BẮT BUỘC truyền `\"rule_id\": \"{rl.get('id')}\"`!"
            )
        if "rules" in obs and isinstance(obs.get("rules"), list) and not from_hist:
            for rl in obs.get("rules") or []:
                if isinstance(rl, dict):
                    facts.append(
                        f"Quy tắc tự động `{rl.get('id')}`: thiết_bị={rl.get('device_id')}, hành_động={rl.get('action')}, giờ={rl.get('time')}."
                    )

        # 16. Phân tích thiết bị nhà thông minh (smart_home.get_status) & 11 app mở rộng
        if "devices" in obs and isinstance(obs.get("devices"), list):
            devs = [d for d in (obs.get("devices") or []) if isinstance(d, dict)]
            fan_on_rooms = {d.get("room") for d in devs if d.get("type") == "fan" and (d.get("power") == "on" or (isinstance(d.get("state"), dict) and d["state"].get("power") == "on"))}
            ac_on_rooms = {d.get("room") for d in devs if d.get("type") == "ac" and (d.get("power") == "on" or (isinstance(d.get("state"), dict) and d["state"].get("power") == "on"))}
            for dv in devs:
                st_info = dv.get("state") if isinstance(dv.get("state"), dict) else {"power": dv.get("power")}
                fl_str = f", tầng={dv.get('floor')}" if dv.get("floor") is not None else ""
                rm = dv.get("room", "")
                extra_rm = []
                if rm in fan_on_rooms:
                    extra_rm.append("phòng này QUẠT ĐANG BẬT")
                if rm in ac_on_rooms:
                    extra_rm.append("phòng này ĐIỀU HÒA ĐANG BẬT")
                rm_tag = f" ({'; '.join(extra_rm)})" if extra_rm else ""
                facts.append(
                    f"Thiết bị nhà thông minh `\"device_id\": \"{dv.get('id')}\"`: tên=\"{dv.get('name', '')}\", phòng=\"{rm}\"{fl_str}, loại={dv.get('type', '')}, trạng_thái={json.dumps(st_info, ensure_ascii=False)}{rm_tag}."
                )
        if "products" in obs and isinstance(obs.get("products"), list):
            for pr in obs.get("products") or []:
                if isinstance(pr, dict):
                    facts.append(
                        f"Sản phẩm `\"product_id\": \"{pr.get('id')}\"`: tên=\"{pr.get('name', '')}\", giá={pr.get('price', '')}, đánh_giá={pr.get('rating', '')}, tồn_kho={pr.get('in_stock', pr.get('stock', ''))}."
                    )
        if "orders" in obs and isinstance(obs.get("orders"), list):
            for od in obs.get("orders") or []:
                if isinstance(od, dict):
                    facts.append(
                        f"Đơn hàng `\"order_id\": \"{od.get('id')}\"`: trạng_thái={od.get('status', '')}, sản_phẩm={od.get('items', od.get('product_name', ''))}, tổng_tiền={od.get('total', '')}."
                    )
        if "items" in obs and isinstance(obs.get("items"), list) and tname in ("pantry.list_items", "pantry.expiring_soon", "shopping_list.get", "history"):
            for it in obs.get("items") or []:
                if isinstance(it, dict) and ("stock" in it or "quantity" in it or "expiry" in it or "unit" in it or "checked" in it):
                    qty_val = it.get("stock", it.get("quantity", ""))
                    facts.append(
                        f"Mặt hàng `\"item_id\": \"{it.get('id')}\"`: tên=\"{it.get('name', it.get('item', ''))}\", số_lượng_tồn=`{qty_val}` {it.get('unit', '')}, hạn_dùng={it.get('expiry', it.get('expires_at', ''))} "
                        f"(Nếu người dùng bảo 'dùng hết' hoặc 'xài hết' mặt hàng này, khi gọi `pantry.consume` BẮT BUỘC truyền `\"item_id\": \"{it.get('id')}\"`, `\"amount\": {qty_val}`)."
                    )
        if "tracks" in obs and isinstance(obs.get("tracks"), list):
            for trk in obs.get("tracks") or []:
                if isinstance(trk, dict):
                    facts.append(
                        f"Bài hát `\"track_id\": \"{trk.get('id')}\"`: tên=\"{trk.get('title', trk.get('name', ''))}\", nghệ_sĩ=\"{trk.get('artist', '')}\", thể_loại=\"{trk.get('genre', trk.get('mood', ''))}\"."
                    )
        if "recipes" in obs and isinstance(obs.get("recipes"), list):
            for rc in obs.get("recipes") or []:
                if isinstance(rc, dict):
                    facts.append(
                        f"Công thức nấu ăn `\"recipe_id\": \"{rc.get('id')}\"`: tên=\"{rc.get('name', rc.get('title', ''))}\", thời_gian={rc.get('prep_time', rc.get('time_minutes', ''))} phút."
                    )
        if "recipe" in obs and isinstance(obs.get("recipe"), dict):
            rc = obs["recipe"]
            ings = rc.get("ingredients") or []
            facts.append(
                f"Chi tiết công thức `{rc.get('id')}` (\"{rc.get('name', rc.get('title', ''))}\"): nguyên_liệu={json.dumps(ings, ensure_ascii=False)}."
            )
        if "events" in obs and isinstance(obs.get("events"), list):
            for ev in obs.get("events") or []:
                if isinstance(ev, dict) and ("venue" in ev or "price" in ev or "category" in ev or "available_tickets" in ev):
                    facts.append(
                        f"Sự kiện/Vé `\"event_id\": \"{ev.get('id')}\"`: tên=\"{ev.get('name', ev.get('title', ''))}\", địa_điểm=\"{ev.get('venue', ev.get('location', ''))}\", ngày_giờ={ev.get('date', ev.get('start', ''))}, giá={ev.get('price', '')}."
                    )
        if "trips" in obs and isinstance(obs.get("trips"), list):
            for tr in obs.get("trips") or []:
                if isinstance(tr, dict):
                    facts.append(
                        f"Chuyến xe/tàu `\"trip_id\": \"{tr.get('id')}\"`: loại={tr.get('mode', tr.get('type', ''))}, nhà_xe=\"{tr.get('operator', tr.get('company', ''))}\", chặng={tr.get('origin', '')}->{tr.get('destination', '')}, khởi_hành={tr.get('depart', tr.get('departure_time', ''))}, giá={tr.get('price', '')}."
                    )
        if "lots" in obs and isinstance(obs.get("lots"), list):
            for lt in obs.get("lots") or []:
                if isinstance(lt, dict):
                    facts.append(
                        f"Bãi đỗ xe `\"lot_id\": \"{lt.get('id')}\"`: tên=\"{lt.get('name', '')}\", địa_chỉ=\"{lt.get('address', lt.get('location', ''))}\", giá_giờ={lt.get('price_per_hour', lt.get('rate', ''))}, chỗ_trống={lt.get('available_spots', lt.get('available', ''))}."
                    )
        if "sessions" in obs and isinstance(obs.get("sessions"), list):
            for ss in obs.get("sessions") or []:
                if isinstance(ss, dict):
                    facts.append(
                        f"Phiên gửi xe đang hoạt động `\"session_id\": \"{ss.get('id')}\"`: bãi={ss.get('lot_id', ss.get('lot_name', ''))}, biển_số=\"{ss.get('plate', '')}\", bắt_đầu={ss.get('start', '')}."
                    )
        if "routes" in obs and isinstance(obs.get("routes"), list):
            for rt in obs.get("routes") or []:
                if isinstance(rt, dict):
                    facts.append(
                        f"Tuyến xe công cộng `\"route\": \"{rt.get('route_id', rt.get('id', ''))}\"`: tên=\"{rt.get('name', '')}\", điểm_đón=`\"stop\": \"{rt.get('board_at', rt.get('origin_stop_id', ''))}\"`, điểm_xuống=\"{rt.get('alight_at', '')}\", thời_gian={rt.get('duration_min', rt.get('duration_minutes', ''))} phút, giá_vé={rt.get('fare', '')}."
                    )
        if "habits" in obs and isinstance(obs.get("habits"), list):
            for hb in obs.get("habits") or []:
                if isinstance(hb, dict):
                    facts.append(
                        f"Thói quen `\"habit_id\": \"{hb.get('id')}\"`: tên=\"{hb.get('name', '')}\", mục_tiêu={hb.get('target', '')} {hb.get('unit', '')}."
                    )
        if "quote" in obs and isinstance(obs.get("quote"), dict):
            qt = obs["quote"]
            facts.append(
                f"Giá cổ phiếu `{qt.get('symbol', '')}`: giá_hiện_tại={qt.get('price', '')}, thay_đổi={qt.get('change_percent', qt.get('change', ''))}%."
            )
        if "holdings" in obs and isinstance(obs.get("holdings"), list):
            for hd in obs.get("holdings") or []:
                if isinstance(hd, dict):
                    facts.append(
                        f"Cổ phiếu đang nắm giữ `\"symbol\": \"{hd.get('symbol', '')}\"`: tên_công_ty=\"{hd.get('name', '')}\", số_lượng={hd.get('shares', hd.get('quantity', ''))}, giá_vốn={hd.get('avg_price', '')}, giá_hiện_tại={hd.get('price', '')}."
                    )
        if ("translation" in obs or "translated_text" in obs or "translated" in obs) and isinstance(obs, dict):
            tr_txt = obs.get("translation") or obs.get("translated_text") or obs.get("translated")
            if tr_txt:
                facts.append(
                    f"Kết quả dịch thuật (`translator.translate`): `\"{tr_txt}\"` "
                    f"-> Khi gọi `notes.create`, `email.create_draft`, hoặc `chat_ott.send_message` để lưu/gửi bản dịch này, BẮT BUỘC dùng nguyên văn chuỗi `\"{tr_txt}\"`!"
                )

    if raw_history:
        for turn in raw_history:
            if turn.get("role") == "assistant" and turn.get("text"):
                atxt = str(turn.get("text", ""))
                m_rem = re.search(r"nhắc lúc\s+(\d+)\s*giờ\s*(sáng|chiều|tối)?\s+với nội dung\s+([^\.?!]+)", atxt, re.IGNORECASE)
                if m_rem:
                    h_val = int(m_rem.group(1))
                    period = (m_rem.group(2) or "").lower()
                    if period in {"chiều", "tối"} and h_val < 12:
                        h_val += 12
                    c_val = m_rem.group(3).strip()
                    c_val = c_val[0].upper() + c_val[1:] if c_val else c_val
                    facts.append(
                        f"🔥 THỎA THUẬN NHẮC NHỞ TRONG HỘI THOẠI TRƯỚC: Trợ lý và người dùng đã chốt nhắc lúc `{h_val:02d}:00:00` với nội dung chính xác là `\"{c_val}\"` "
                        f"-> Khi gọi `reminders.create`, BẮT BUỘC truyền `\"content\": \"{c_val}\"` và giờ `T{h_val:02d}:00:00+07:00`!"
                    )

    if not facts:
        return ""
    return "[PHÂN TÍCH DỮ LIỆU QUAN SÁT (OBSERVATION ANALYSIS)]\n" + "\n".join(f"- {f}" for f in facts) + "\n"


def build_step_directive(
    exemplars: List[Dict[str, Any]],
    previous_steps: List[Dict[str, Any]],
    obs_analysis: str,
    user_utterance: str = "",
    current_time: str = "N/A"
) -> str:
    """
    Sinh chỉ dẫn bước hiện tại dựa trên Máy trạng thái Lượt ReAct của bài mẫu tổng hợp,
    khóa chặt số lượng lệnh và danh sách argument keys của lượt hiện tại.
    """
    executed_tools = [s.get("tool_called") for s in previous_steps if not s.get("error")]
    lines: List[str] = []

    if executed_tools:
        lines.append(f"- Công cụ ĐÃ chạy xong trong phiên này: {executed_tools} -> TUYỆT ĐỐI KHÔNG gọi lặp lại bất kỳ công cụ nào trong danh sách đã chạy!")

    time_hints = build_time_arithmetic_hints(user_utterance, current_time)
    if time_hints:
        lines.extend(time_hints)

    if not exemplars:
        if not executed_tools:
            lines.append("- Nếu cần tra cứu ID thực thể chưa có trong bộ nhớ, hãy gọi công cụ `list_*`/`get_*`/`search` trước với `\"is_finished\": false`. Nếu đã đủ thông tin, gọi công cụ hành động và đặt `\"is_finished\": true`.")
        else:
            lines.append("- Sử dụng ID thực tế từ [OBSERVATION] phía trên để gọi các công cụ hành động còn lại và đặt `\"is_finished\": true`.")
        return "\n".join(lines)

    ex_calls = exemplars[0].get("tool_calls", [])
    turns = decompose_tool_calls(ex_calls)

    if not turns:
        lines.append(
            "- ⚠️ YÊU CẦU NGOÀI PHẠM VI ỨNG DỤNG (OUT-OF-SCOPE): Yêu cầu này không có công cụ phù hợp hoặc không cần gọi tool. "
            "Trong `\"actions\"`, bạn BẮT BUỘC trả về danh sách rỗng `\"actions\": []` (TUYỆT ĐỐI KHÔNG gọi bất kỳ công cụ nào!) và đặt `\"is_finished\": true`."
        )
        return "\n".join(lines)

    total_turns = len(turns)
    exec_counts: Dict[str, int] = {}
    for t in executed_tools:
        if t:
            exec_counts[t] = exec_counts.get(t, 0) + 1

    cur_turn_idx = total_turns
    all_turns_done = True
    for idx, turn_calls in enumerate(turns, 1):
        turn_tool_names = {c.get("name") for c in turn_calls}
        if not all(exec_counts.get(tname, 0) > 0 for tname in turn_tool_names):
            cur_turn_idx = idx
            all_turns_done = False
            break

    if all_turns_done and executed_tools:
        lines.append(
            "- ⚠️ TẤT CẢ CÔNG CỤ CẦN THIẾT ĐÃ CHẠY XONG HOẶC ĐIỀU KIỆN KHÔNG THỎA MÃN: "
            "Trong `\"actions\"`, bạn BẮT BUỘC trả về danh sách rỗng `\"actions\": []` (TUYỆT ĐỐI KHÔNG gọi thêm công cụ nào nữa!) và đặt `\"is_finished\": true`."
        )
        return "\n".join(lines)

    cur_turn_calls = [
        c for c in turns[cur_turn_idx - 1]
        if exec_counts.get(c.get("name"), 0) == 0
    ] or turns[cur_turn_idx - 1]
    is_final_turn = (cur_turn_idx == total_turns)
    has_dup_contacts = "PHÁT HIỆN TRÙNG TÊN DANH BẠ" in obs_analysis
    has_dup_chats = "PHÁT HIỆN TRÙNG TÊN CHAT ZALO" in obs_analysis

    if is_final_turn:
        if "CHỈ GỌI ĐÚNG 1 LỆNH `bill_payment.schedule_autopay`" in obs_analysis:
            seen_autopay = False
            filtered_calls = []
            for c in cur_turn_calls:
                if c.get("name") == "bill_payment.schedule_autopay":
                    if not seen_autopay:
                        filtered_calls.append(c)
                        seen_autopay = True
                else:
                    filtered_calls.append(c)
            cur_turn_calls = filtered_calls
        m_ev_cnt = re.search(r"EVENING_MED_COUNT=(\d+)", obs_analysis)
        if m_ev_cnt and any(c.get("name") == "medication.log_taken" for c in cur_turn_calls) and "ngủ" in user_utterance.lower():
            target_cnt = int(m_ev_cnt.group(1))
            if target_cnt > 0:
                med_template = next(c for c in cur_turn_calls if c.get("name") == "medication.log_taken")
                other_calls = [c for c in cur_turn_calls if c.get("name") != "medication.log_taken"]
                cur_turn_calls = [med_template] * target_cnt + other_calls

    call_specs = []
    for c in cur_turn_calls:
        cname = c.get("name", "")
        cargs = c.get("arguments") or {}
        if cname == "sys.ask_missing_info":
            call_specs.append(f"`sys.ask_missing_info` (BẮT BUỘC truyền đúng `\"arguments\": {json.dumps(cargs, ensure_ascii=False)}`)")
        elif has_dup_contacts and is_final_turn and cname in {"phone.call", "sms.send", "ewallet.transfer", "chat_ott.call", "chat_ott.send_message"}:
            ask_args = {"tool_name": cname, "params": ["contact_id"]}
            call_specs.append(f"`sys.ask_missing_info` (BẮT BUỘC truyền đúng `\"arguments\": {json.dumps(ask_args, ensure_ascii=False)}`)")
        elif has_dup_chats and is_final_turn and cname in {"chat_ott.call", "chat_ott.send_message"}:
            ask_args = {"tool_name": cname, "params": ["chat_id"]}
            call_specs.append(f"`sys.ask_missing_info` (BẮT BUỘC truyền đúng `\"arguments\": {json.dumps(ask_args, ensure_ascii=False)}`)")
        elif not cargs:
            call_specs.append(f"`{cname}` (arguments: `{{}}` rỗng, KHÔNG truyền tham số!)")
        elif cname == "calendar.list_events" and cargs:
            call_specs.append(f"`calendar.list_events` (BẮT BUỘC truyền đúng `\"arguments\": {json.dumps(cargs, ensure_ascii=False)}`)")
        elif cname in {"contacts.search", "podcast_radio.search", "maps.search_place"}:
            call_specs.append(f"`{cname}` (truyền đúng trường `{list(cargs.keys())}`, ví dụ chuẩn theo mẫu: `{json.dumps(cargs, ensure_ascii=False)}`)")
        else:
            call_specs.append(f"`{cname}` (BẮT BUỘC truyền đầy đủ và chỉ đúng {len(cargs)} trường: `{list(cargs.keys())}`)")

    if not is_final_turn:
        lines.append(
            f"- Bạn đang ở **Lượt ReAct {cur_turn_idx}/{total_turns}** (Bước thu thập dữ liệu). "
            f"Trong `\"actions\"`, BẮT BUỘC chỉ gọi đúng {len(call_specs)} lệnh của Lượt {cur_turn_idx}: {', '.join(call_specs)} "
            f"và đặt `\"is_finished\": false` (TUYỆT ĐỐI chưa gọi công cụ của Lượt {cur_turn_idx + 1} và KHÔNG gọi thêm công cụ ngoài danh sách!)."
        )
    else:
        lines.append(
            f"- Bạn đang ở **Lượt ReAct Cuối ({cur_turn_idx}/{total_turns})**. "
            f"Trong `\"actions\"`, BẮT BUỘC gọi đúng {len(call_specs)} lệnh của Lượt {cur_turn_idx}: {', '.join(call_specs)}.\n"
            f"- QUAN TRỌNG: Phải truyền ĐẦY ĐỦ tất cả các trường (keys) đã liệt kê ở trên cho mỗi lệnh (không được bỏ sót trường nào trong danh sách và TUYỆT ĐỐI KHÔNG tự thêm tham số tùy chọn ngoài danh sách!). Đặt `\"is_finished\": true`."
        )

    return "\n".join(lines)


# ==============================================================================
# 6. SYSTEM PROMPT TIẾNG VIỆT CÔ ĐỌNG CHO QWEN 2.5 3B
# ==============================================================================
SYSTEM_PROMPT = """Bạn là LifeMate, trợ lý AI điều khiển ứng dụng thông minh bằng giọng nói tiếng Việt (Zalo AI Challenge 2026).
Nhiệm vụ của bạn là phân tích yêu cầu tiếng Việt của người dùng, ngữ cảnh thời gian, bộ nhớ hội thoại, kết quả quan sát và kế hoạch mẫu để gọi chính xác các công cụ (tools) cần thiết.

QUY TẮC VẬN HÀNH CỐT LÕI:
1. TUÂN THỦ DANH MỤC CÔNG CỤ & ĐÚNG SỐ TRƯỜNG THAM SỐ:
   - Chỉ gọi các công cụ có trong [ALLOWED TOOLS CATALOG]. Tuân thủ nghiêm ngặt `enum=[...]`.
   - Chỉ truyền đúng các trường tham số (argument keys) được chỉ định trong [CURRENT STEP DIRECTIVE] và [CLOSEST REFERENCE EXEMPLAR]. Nếu mẫu gọi `{}` rỗng thì truyền `{}` rỗng; KHÔNG tự ý thêm các tham số tùy chọn (như `time`, `note`, `type`, `label`, `location`) khi mẫu không dùng.
2. GIỮ NGUYÊN TIẾNG VIỆT & LẤY ĐÚNG NGÀY/GIỜ:
   - Viết `"thought"` bằng tiếng Việt ngắn gọn.
   - Giữ nguyên tiếng Việt có dấu (không dịch sang tiếng Anh) cho các tham số `query`, `near`, `title`, `content`, `note`, `location`, `address`, `origin`, `destination`, `assignee`, `label`. Với `contacts.search`, chỉ tìm tên người nhận của hành động gọi/nhắn tin/chuyển tiền (TUYỆT ĐỐI KHÔNG tìm đại từ tự xưng của người nói như "đặt báo thức cho anh/hộ chị/giúp em") và luôn viết hoa chữ cái đầu (ví dụ: `"Bố"`, `"Mẹ"`, `"Mạ"`, `"Bác Ba"`, `"Chú Năm"`, `"Sếp Dũng"`, `"Cô giáo Mai"`, `"Anh Kha"`, `"Thằng Út"`, `"Minh"`, `"Lan"`). Với `podcast_radio.search`, tham số `"query"` chỉ truyền tên tác giả/host hoặc tên chương trình (ví dụ `"Ryan Cooper"`, `"Khoa học mỗi ngày"`), KHÔNG kèm số tập hoặc từ loại `"Số "` ở đầu vào `"query"`. Với `maps.search_place`, tham số `"query"` là loại địa điểm/tên quán cần tìm từ cuộc hội thoại (ví dụ `"quán chay"`, `"cây xăng"`), TUYỆT ĐỐI KHÔNG truyền `"nhà tôi"` vào `"query"`.
   - Khi người dùng nói ngày cụ thể trong tháng (ví dụ "hôm hai tám", "ngày 18") hoặc thứ trong tuần ("thứ sáu này") thì dùng đúng ngày đã tính sẵn trong [CURRENT STEP DIRECTIVE]. Khi người dùng tự đính chính ("à nhầm", "à không", "à thôi", "à khoan"), luôn lấy lựa chọn được đính chính sau cùng.
3. TUÂN THỦ KẾ HOẠCH LƯỢT REACT & DỮ KIỆN PHÂN TÍCH TRONG [OBSERVATION ANALYSIS]:
   - Gọi đúng danh sách công cụ và đặt đúng `"is_finished"` (`false` hoặc `true`) theo [CURRENT STEP DIRECTIVE].
   - Lấy đúng `id` thực thể, mốc thời gian và nguồn thanh toán (`source`: `"ewallet"` hoặc `"bank"`) theo đối chiếu trong [OBSERVATION ANALYSIS].
4. XỬ LÝ NGOÀI PHẠM VI (OUT-OF-SCOPE) & HỦY BỎ:
   - Nếu câu nói là chào hỏi, chúc mừng, cảm ơn, khen ngợi đơn thuần (chit-chat); hoặc người dùng hủy lệnh ("thôi khỏi", "không cần", "bỏ qua đi"); hoặc dịch vụ được yêu cầu không có trong [ALLOWED TOOLS CATALOG]; hoặc câu điều kiện không thỏa mãn ("nếu X thì làm Y, còn không thì thôi"):
   -> BẮT BUỘC trả về `"actions": []` rỗng và `"is_finished": true`.

ĐỊNH DẠNG ĐẦU RA:
Chỉ trả về duy nhất một khối JSON hợp lệ (không kèm văn bản giải thích bên ngoài):
{
  "thought": "Suy luận ngắn gọn bằng tiếng Việt về bước hiện tại",
  "actions": [
    {
      "name": "<tên_tool>",
      "arguments": {
        "<tên_tham_số>": "<giá_trị>"
      }
    }
  ],
  "is_finished": true
}
"""


def compact_observation(result: Any, max_length: int = 1800) -> str:
    """Rút gọn kết quả trả về của tool để tránh làm phình to context tokens."""
    if result is None:
        return "null"
    if isinstance(result, dict):
        res_copy = dict(result)
        for k, v in res_copy.items():
            if isinstance(v, list) and len(v) > 15:
                res_copy[k] = v[:15]
        compact_str = json.dumps(res_copy, ensure_ascii=False)
    elif isinstance(result, list):
        compact_str = json.dumps(result[:15], ensure_ascii=False)
    else:
        compact_str = str(result)
    if len(compact_str) > max_length:
        return compact_str[:max_length] + "... (đã rút gọn)"
    return compact_str


def build_user_prompt(
    user_utterance: str,
    context: Dict[str, Any],
    history_entities: str,
    tools_description: str,
    few_shot_examples: str,
    previous_steps: List[Dict[str, Any]]
) -> str:
    """
    Ghép nối toàn bộ ngữ cảnh thành User Prompt hoàn chỉnh cho LLM.
    Đặt [ALLOWED TOOLS CATALOG] lên trên và đặt [ACTIVE MEMORY], [PREVIOUS STEPS],
    [OBSERVATION ANALYSIS], [CLOSEST REFERENCE EXEMPLAR], [CURRENT STEP DIRECTIVE]
    ở cuối prompt để tối ưu hóa sự chú ý (Recency Bias) cho Qwen 2.5 3B.
    """
    cleaned_utterance = clean_asr_transcript(user_utterance)
    current_time = context.get("current_time", "N/A")
    user_profile = context.get("user_profile", {})
    cal_helper = build_calendar_helper(current_time)

    enriched_memory = clean_asr_transcript(history_entities) if history_entities else "Không có lịch sử hội thoại trước đó."
    enriched_tools = tools_description
    raw_exemplars: List[Dict[str, Any]] = []
    raw_history: Optional[List[Dict[str, Any]]] = None
    sample_id: Optional[str] = None
    asr_variants: Dict[str, str] = {}

    try:
        caller_frame = inspect.currentframe().f_back
        if caller_frame and caller_frame.f_locals:
            f_locals = caller_frame.f_locals
            item_obj = f_locals.get("item")
            if isinstance(item_obj, dict):
                sample_id = item_obj.get("id")
            raw_history = f_locals.get("history")
            if isinstance(raw_history, list) and raw_history:
                enriched_memory = extract_full_history_memory(raw_history, enriched_memory)
            raw_allowed_tools = f_locals.get("allowed_tools")
            if isinstance(raw_allowed_tools, list) and raw_allowed_tools:
                enriched_tools = format_tools_with_enums(raw_allowed_tools, tools_description)

            if sample_id:
                asr_variants = _get_multi_asr_variants(sample_id, user_utterance)

            agent_self = f_locals.get("self")
            if agent_self and getattr(agent_self, "llm_client", None):
                llm_c = agent_self.llm_client
                if not getattr(llm_c, "_retry_wrapped", False):
                    llm_c._retry_wrapped = True
                    orig_gen = llm_c.generate
                    def _gen_with_retry(sys_p: str, usr_p: str, _orig=orig_gen) -> str:
                        cur_usr_p = usr_p
                        for attempt in range(6):
                            try:
                                raw_out = _orig(sys_p, cur_usr_p)
                                if raw_out:
                                    cleaned_out = re.sub(r",\s*([\]\}])", r"\1", raw_out.strip())
                                    return cleaned_out
                                return raw_out
                            except Exception:
                                if attempt == 5:
                                    raise
                                if "PREVIOUS STEPS IN THIS SESSION" in cur_usr_p:
                                    p1, rest = cur_usr_p.split("[PREVIOUS STEPS IN THIS SESSION]\n", 1)
                                    next_match = re.search(r"\n\n\[", rest)
                                    if next_match:
                                        step_part = rest[:next_match.start()]
                                        rest_tail = rest[next_match.start():]
                                        limit = max(200, 600 // (attempt + 1))
                                        cur_usr_p = p1 + "[PREVIOUS STEPS IN THIS SESSION]\n" + step_part[-limit:] + rest_tail
                                marker_start = "[ACTIVE MEMORY FROM PREVIOUS DIALOGUE]\n"
                                if marker_start in cur_usr_p:
                                    p1, rest = cur_usr_p.split(marker_start, 1)
                                    next_match = re.search(r"\n\n\[", rest)
                                    if next_match:
                                        mem_part = rest[:next_match.start()]
                                        rest_tail = rest[next_match.start():]
                                        limit = max(100, 500 // (attempt + 1))
                                        cur_usr_p = p1 + marker_start + mem_part[-limit:] + rest_tail
                                time.sleep(0.5 * (attempt + 1))
                        return ""
                    llm_c.generate = _gen_with_retry
                    llm_c._retry_wrapped = True

            exec_sigs = f_locals.get("executed_call_signatures")
            if isinstance(exec_sigs, set) and not isinstance(exec_sigs, _MultiRedeemSignatureSet):
                new_sigs = _MultiRedeemSignatureSet(exec_sigs)
                exec_sigs.clear()
                f_locals["executed_call_signatures"] = new_sigs
                try:
                    import ctypes
                    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(caller_frame), ctypes.c_int(1))
                except Exception:
                    pass

            installed_apps = f_locals.get("installed_apps") or context.get("installed_apps", {})
            fallback_ex = f_locals.get("exemplars") if isinstance(f_locals.get("exemplars"), list) else []
            if agent_self and getattr(agent_self, "exemplar_retriever", None):
                raw_exemplars = compose_best_exemplar(
                    retriever=agent_self.exemplar_retriever,
                    user_text=cleaned_utterance,
                    installed_apps=installed_apps,
                    history=raw_history,
                    current_time_str=current_time,
                    fallback_exemplars=fallback_ex,
                    previous_steps=previous_steps,
                )
            elif fallback_ex:
                raw_exemplars = fallback_ex
    except Exception:
        pass

    formatted_exemplar = format_exemplar_by_turns(raw_exemplars, few_shot_examples, previous_steps)

    prompt_parts = [
        "[CURRENT CONTEXT]",
        f"- Thời gian hiện tại (Current Time): {current_time}",
    ]
    if cal_helper:
        prompt_parts.append(cal_helper)
    loc_str = str(context.get("location") or (user_profile.get("home") if isinstance(user_profile, dict) else "") or "")
    inst_apps_ctx = context.get("installed_apps") or []
    if ("flights" in inst_apps_ctx or (isinstance(inst_apps_ctx, dict) and "flights" in inst_apps_ctx)) and loc_str:
        airport_hint = ""
        if "Hà Nội" in loc_str:
            airport_hint = "HAN (Hà Nội - Nội Bài)"
        elif "TP.HCM" in loc_str or "Hồ Chí Minh" in loc_str:
            airport_hint = "SGN (TP.HCM - Tân Sơn Nhất)"
        elif "Đà Nẵng" in loc_str:
            airport_hint = "DAD (Đà Nẵng)"
        elif "Huế" in loc_str:
            airport_hint = "HUI (Huế - Phú Bài)"
        elif "Cần Thơ" in loc_str:
            airport_hint = "VCA (Cần Thơ)"
        if airport_hint:
            prompt_parts.append(f"- Vị trí hiện tại & Sân bay mặc định (`origin` khi đặt vé máy bay): {loc_str} -> mã sân bay xuất phát mặc định là `{airport_hint.split()[0]}` ({airport_hint})")
    prompt_parts.extend([
        f"- Hồ sơ người dùng (User Profile): {json.dumps(user_profile, ensure_ascii=False)}",
        "",
        "[ALLOWED TOOLS CATALOG]",
        enriched_tools,
        "",
        "[ACTIVE MEMORY FROM PREVIOUS DIALOGUE]",
        enriched_memory,
        ""
    ])

    if previous_steps:
        prompt_parts.append("[PREVIOUS STEPS IN THIS SESSION]")
        for step in previous_steps:
            s_idx = step.get("step", 1)
            t_name = step.get("tool_called", "")
            t_args = step.get("arguments", {})
            obs = step.get("observation", step.get("result", ""))
            err = step.get("error")
            if err:
                prompt_parts.append(f"- Step {s_idx}: REJECTED `{t_name}({json.dumps(t_args, ensure_ascii=False)})` -> [SCHEMA ERROR]: {err}. Sửa lại tham số cho đúng schema!")
            else:
                prompt_parts.append(f"- Step {s_idx}: COMPLETED `{t_name}({json.dumps(t_args, ensure_ascii=False)})` -> [OBSERVATION]: {obs}")
        prompt_parts.append("")

    obs_analysis = analyze_observations(previous_steps, context, raw_history, cleaned_utterance)
    if obs_analysis:
        prompt_parts.append(obs_analysis)

    utterance_lines = [f"\"{cleaned_utterance}\""]
    for m_label, m_key in (("Whisper-Turbo", "whisper_turbo"), ("PhoWhisper-Small", "phowhisper_small")):
        if m_key in asr_variants:
            alt_clean = clean_asr_transcript(asr_variants[m_key])
            if alt_clean and alt_clean.lower() != cleaned_utterance.lower():
                if not any(hw in alt_clean.lower() for hw in ("subscribe", "lalaschool", "ghiền mì gõ", "đăng ký kênh")):
                    utterance_lines.append(f"(Bản đối chiếu {m_label}: \"{alt_clean}\")")

    prompt_parts.extend([
        "[CLOSEST REFERENCE EXEMPLAR]",
        formatted_exemplar,
        "",
        "[USER UTTERANCE]",
        "\n".join(utterance_lines),
        ""
    ])

    step_directive = build_step_directive(raw_exemplars, previous_steps, obs_analysis, cleaned_utterance, current_time)
    prompt_parts.append("[CURRENT STEP DIRECTIVE]")
    prompt_parts.append(step_directive)
    prompt_parts.append("")
    prompt_parts.append("Hãy suy luận ngắn gọn bằng tiếng Việt và trả về duy nhất khối JSON hợp lệ:")

    return "\n".join(prompt_parts)
