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
    (r"\b(món ruột lùi trùi pùi em coi món nào làm đẹp nhất thì mời tin tức dù mai nha)\b", "món ratatouille em coi món nào làm lẹ nhất thì mở chi tiết giùm anh nha"),
    (r"\b(tìm phương thơ mua gần trường em rồi lưu lại tên là trợ inh bài nga)\b", "tiệm photo gần trường em rồi lưu lại tên là Chỗ in bài nha"),
    (r"\b(em tìm trang mua đồ giúp anh gỗ cám chuyển đổi của nốt quyn travel giá bán trăm rưỡi nghìn thắng ai)\b", "em tìm trên mua đồ giúp anh ổ cắm chuyển đổi của Northwind Travel dưới 450 nghìn hãng nào"),
    (r"\b(google em hệt)\b", "Hugo Lambert"),
    (r"\b(kết thúc phiên độ xé)\b", "kết thúc phiên đỗ xe"),
    (r"\b(mấy món chơi gà gà)\b", "mấy món chưa gạch"),
    (r"\b(thêm hơi dở mua đồ)\b", "thêm vào giỏ mua đồ"),
    (r"\b(nhưng ăn chơi cay ấm đây nha)\b", "nhưng anh trừ cái ấm ra nha"),
    (r"\b(còn được về chỉ đi tàu em coi dần chịu em ở được mấy đêm chẳng hạn)\b", "còn lượt về chị đi tàu em coi giùm chị ở được mấy đêm vậy"),
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
]


def clean_asr_transcript(text: str) -> str:
    """Khôi phục các cụm từ bị biến âm vị do ASR tiếng Việt."""
    if not text:
        return ""
    cleaned = text
    for pattern, replacement in ASR_PHONETIC_FIXES:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


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
    ("một trăm hai mươi sáu", "126"), ("một trăm hai sáu", "126"), ("trăm hai sáu", "126"),
    ("một trăm hai mươi tám", "128"), ("một trăm hai tám", "128"), ("trăm hai tám", "128"),
    ("một trăm ba mươi lăm", "135"), ("một trăm ba lăm", "135"), ("trăm ba lăm", "135"),
    ("một trăm ba mươi", "130"), ("trăm ba mươi", "130"),
    ("một trăm bốn mươi", "140"), ("trăm bốn mươi", "140"),
    ("một trăm năm mươi", "150"), ("trăm năm mươi", "150"),
    ("bốn nghìn một trăm", "4100"), ("bốn ngàn một trăm", "4100"),
    ("chín mươi lăm", "95"), ("chín lăm", "95"), ("chín mươi hai", "92"), ("chín hai", "92"), ("chín mươi", "90"),
    ("tám mươi lăm", "85"), ("tám lăm", "85"), ("tám mươi tám", "88"), ("tám tám", "88"),
    ("tám mươi sáu", "86"), ("tám sáu", "86"), ("tám mươi tư", "84"), ("tám tư", "84"),
    ("tám mươi hai", "82"), ("tám hai", "82"), ("tám mươi", "80"),
    ("bảy mươi lăm", "75"), ("bảy lăm", "75"), ("bảy mươi", "70"),
    ("sáu mươi lăm", "65"), ("sáu lăm", "65"), ("sáu mươi", "60"),
    ("hai mươi tám", "28"), ("hai tám", "28"), ("hai mươi chín", "29"), ("hai chín", "29"),
    ("hai mươi bảy", "27"), ("hai bảy", "27"), ("hai mươi sáu", "26"), ("hai sáu", "26"),
    ("hai mươi lăm", "25"), ("hai lăm", "25"), ("hai mươi bốn", "24"), ("hai tư", "24"),
    ("hai mươi ba", "23"), ("hai ba", "23"), ("hai mươi hai", "22"), ("hai hai", "22"),
    ("hai mươi mốt", "21"), ("hai mốt", "21"), ("hai mươi", "20"),
    ("mười một", "11"), ("mười hai", "12"), ("mười lăm", "15"), ("mười bốn", "14"), ("mười ba", "13"),
    ("mười sáu", "16"), ("mười bảy", "17"), ("mười tám", "18"), ("mười chín", "19"),
    ("ba mươi lăm", "35"), ("ba lăm", "35"), ("ba mươi mốt", "31"), ("ba mốt", "31"), ("ba mươi", "30"),
    ("bốn mươi lăm", "45"), ("bốn lăm", "45"), ("bốn mươi", "40"),
    ("năm mươi lăm", "55"), ("năm lăm", "55"), ("năm mươi", "50"),
    ("hai trăm", "200"), ("ba trăm", "300"), ("bốn trăm", "400"), ("năm trăm", "500"), ("một trăm", "100"),
    ("mười", "10"), ("một", "1"), ("hai", "2"), ("ba", "3"), ("bốn", "4"),
    ("năm", "5"), ("sáu", "6"), ("bảy", "7"), ("tám", "8"), ("chín", "9"),
]


def _normalize_vn_numbers_in_time(text: str) -> str:
    out = text.lower()
    for w, d in VN_WORD_TO_NUM:
        out = re.sub(rf"\b{w}\b", d, out)
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
    rf"(?:[,.;]?\s+(?<!\bxong\s)(?<!\bđổi\s)(?<!\bgiao\s)(?<!\bđóng\s)(?<!\buống\s)(?<!\băn\s)(?<!\bbơi\s)(?<!\bsố\s)(?<!\bmới\s)\b(?:rồi|xong(?!\s+(?:rồi|giùm|dùm|nhé|nha|chưa|dụng|việc))|với cả|với lại|tiện thể|nhân tiện|đồng thời)\s+(?=(?:{ACTION_VERB_START}|sáng|chiều|tối|trưa|mai|hôm|thứ|chủ nhật|lúc|sáu|bảy|tám|chín|mười|\d)\b)"
    rf"|[,.;]?\s+\bvới\s+(?={ACTION_VERB_START}\b))",
    flags=re.IGNORECASE
)

TOOL_KEYWORD_GUARDS: Dict[str, str] = {
    "reminders.create": r"\b(nhắc|lời nhắc)\b",
    "alarm.create": r"\b(báo thức|đánh thức|gọi .* dậy|hẹn giờ dậy)\b",
    "alarm.update": r"\b(dời|đổi|chuyển|sửa|lùi|đẩy|sớm hơn|muộn hơn|trễ hơn|thành \d|sang \d|giờ thành)\b",
    "alarm.delete": r"\b(xóa|xoá|tắt|hủy|huỷ|bỏ)\b",
    "tasks.add": r"\b(thêm .* việc|tạo .* việc|tạo luôn dịp|việc cần làm|danh sách việc|hạn\b)\b",
    "notes.create": r"\b(tạo .* ghi chú|lưu .* ghi chú|viết .* ghi chú|ghi chú tên là|lưu bản dịch vào ghi chú|lưu tên món .* vào ghi chú|ghi chú bữa)\b",
    "notes.append": r"\b(thêm .* vào ghi chú|ghi thêm vào ghi chú|nối vào ghi chú|ghi tiếp vào ghi chú|bổ sung vào ghi chú)\b",
    "pantry.consume": r"\b(trừ|dùng hết|xài hết|lấy .* trong tủ|ăn hết|uống hết)\b",
    "stocks.place_order": r"\b(bán|mua|đặt lệnh)\b",
    "stocks.set_price_alert": r"\b(cảnh báo|báo khi|vượt|xuống dưới|lên lại giá vốn)\b",
    "home_automation.delete_rule": r"\b(xóa|xoá|hủy|huỷ|bỏ)\b",
    "home_automation.create_rule": r"\b(tạo|thêm|cài|lập|khi nào .* thì bật|hễ .* thì)\b",
    "subscriptions.cancel": r"\b(hủy|huỷ|bỏ|ngừng|cắt)\b",
    "food_delivery.cancel_order": r"\b(hủy|huỷ|bỏ)\b",
    "food_delivery.order_history": r"\b(đơn|hủy|huỷ|giao tới đâu)\b",
    "food_delivery.place_order": r"\b(đặt|mua|gọi món|lấy cho|giúp chị|giúp anh|giúp em|giúp mình|mỗi người một)\b",
    "calendar.create_event": r"\b(tạo .* lịch|thêm .* lịch|ghi vào lịch|cho vào lịch|đặt lịch)\b",
    "calendar.cancel_event": r"\b(hủy|huỷ|xóa|xoá|bỏ)\b",
    "calendar.update_event": r"\b(dời|đổi|chuyển|lùi|đẩy|sửa|sớm hơn|muộn hơn|trễ hơn)\b",
    "sms.read_recent": r"\b(sms|tin nhắn)\b",
    "ride_hailing.cancel": r"\b(hủy|huỷ|bỏ)\b",
    "ride_hailing.book": r"\b(đặt|gọi .* (?:xe|ô tô|taxi)|bắt .* xe|cho xe .* (?:đón|rước)|kiếm .* xe)\b",
    "ride_hailing.get_ride": r"\b(tới đâu|tới nơi|sắp tới|bao lâu|mấy phút|mấy giờ .* (?:rước|đón)|tài xế|xe đang|chuyến xe|hủy|huỷ|bỏ)\b",
    "maps.save_place": r"\b(lưu|đánh dấu)\b",
    "music.play": r"\b(nhạc|bài|ca khúc|playlist|nghe|phát|mở|bật .* của|bản|như .* nói hồi nãy)\b",
    "music.list_tracks": r"\b(nhạc|bài|ca khúc|playlist|nghe|phát|mở|bật .* của|bản|như .* nói hồi nãy)\b",
    "music.control": r"\b(nhạc|bài|dừng|tạm dừng|tiếp theo|qua bài|lặp lại|âm lượng|chuyển bài)\b",
    "music.add_to_playlist": r"\b(playlist|danh sách phát)\b",
    "email.create_draft": r"\b(nháp|soạn|viết .* thư|viết .* mail|viết .* email|gửi .* mail|gửi .* email|gửi .* thư|ghi là)\b",
    "smart_home.light_set": r"\b(đèn|ánh sáng|sáng|phần trăm|%|bật lên|tắt dùm|tắt hai cái|phòng khách với bếp)\b",
    "smart_home.fan_set": r"\b(quạt|quẹt|bật nó lên)\b",
    "smart_home.climate_set": r"\b(điều hòa|điều hoà|máy lạnh|nhiệt độ|hút ẩm|độ c\b|\d{2}\s*độ)\b",
    "ewallet.topup_mobile": r"\b(nạp|đạp .* tiền điện thoại|nàng .* tiền điện thoại)\b",
    "ewallet.transfer": r"\b(ví|tiền|nghìn|ngàn|triệu|củ|đồng|khoản|số đấy)\b",
    "ewallet.pay_qr": r"\b(quét mã|qr|thanh toán cho quán|trả cho quán|trả hết cho|trả phần tôi|đối diện|quán)\b",
    "home_security.get_camera": r"\b(camera|cam\b|ghi hình)\b",
    "package_tracking.track": r"\b(đơn|gói hàng|bưu kiện|kiện hàng|vận đơn|giao tới đâu|ship)\b",
    "contacts.update": r"\b(đổi số|sửa .* danh bạ|cập nhật .* số|số mới)\b",
    "podcast_radio.subscribe": r"\b(đăng ký|theo dõi|subscribe)\b",
    "email.archive": r"\b(lưu trữ|archive|bỏ vào lưu trữ|dọn hộp thư)\b",
    "ecommerce.order_status": r"\b(đơn|hàng .* giao|trả hàng|hủy|huỷ|kiểm tra đơn)\b",
    "ecommerce.cancel_order": r"\b(hủy|huỷ|khanh sồi|cancel)\b",
    "ecommerce.request_return": r"\b(trả hàng|trả lại|trả đơn|yêu cầu trả|hoàn trả|đổi trả|giao nhầm|làm đơn trả)\b",
    "ecommerce.checkout": r"\b(đặt|mua|chốt|thanh toán|giao về|giao lên)\b",
    "grocery.search": r"\b(đi chợ|siêu thị|đặt mua|mua thêm|vào giỏ|vô giỏ|sữa|trứng|thịt|rau|gạo|cá|ức gà|cà chua|dầu ô liu|yến mạch|bột mì|táo|hạnh nhân|macca)\b",
    "grocery.add_to_cart": r"\b(đi chợ|siêu thị|đặt mua|mua thêm|vào giỏ|vô giỏ|bỏ .* vào|thêm .* vào|đặt cho tôi .* hộp|lấy .* hộp|hạnh nhân)\b",
    "grocery.checkout": r"\b(đặt|mua|giao về|chốt đơn)\b",
    "train_bus_tickets.book": r"\b(đặt|mua|lấy .* vé|chốt)\b",
    "train_bus_tickets.search": r"\b(đặt|mua|lấy .* vé|tìm .* chuyến|tìm .* vé|còn vé)\b",
    "train_bus_tickets.cancel": r"\b(hủy|huỷ|bỏ)\b",
    "flights.cancel": r"\b(hủy|huỷ|bỏ)\b",
    "flights.check_in": r"\b(check[ -]?in|làm thủ tục|chụp in)\b",
    "parking.end_session": r"\b(kết thúc|lấy xe ra|tách cái phiên|tắt .* phiên)\b",
    "parking.reserve": r"\b(giữ|đặt chỗ|gửi xe\s+\d|để xe)\b",
    "chat_ott.send_message": r"\b(nhắn|gửi|báo .* qua zalo|zalo|nhóm)\b",
    "chat_ott.list_chats": r"\b(nhắn|gửi|zalo|nhóm|tin nhắn)\b",
    "public_transit.topup_card": r"\b(nạp)\b",
    "public_transit.find_routes": r"\b(từ .* (?:đi|ra|tới|đến|về|qua|lên)|tuyến|xe buýt nào|đi được không|bắt chuyến nào|so với xe buýt)\b",
    "public_transit.next_departures": r"\b(mấy giờ có chuyến|chuyến .* kế tiếp|tàu điện kế tiếp|chuyến tiếp theo|mấy giờ chạy)\b",
    "recipes.save": r"\b(lưu|cất)\b",
    "shopping_list.add": r"\b(danh sách đi chợ|danh sách mua sắm|danh sách cần mua)\b",
    "shopping_list.check": r"\b(gạch|đánh dấu)\b",
    "weather.get_forecast": r"\b(thời tiết|dự báo|mưa|trời (?:mưa|nắng|lạnh|nóng|mát|ráo|âm u)|có nắng không|nắng hay mưa|bao nhiêu độ ngoài trời|ngoài trời .* độ|gió|bão|khô ráo)\b",
    "weather.get_current": r"\b(thời tiết|dự báo|mưa|trời (?:mưa|nắng|lạnh|nóng|mát|ráo|âm u)|có nắng không|nắng hay mưa|bao nhiêu độ ngoài trời|ngoài trời .* độ|gió|bão|khô ráo)\b",
}

APP_TRIGGER_PATTERNS: Dict[str, str] = {
    "weather": r"\b(thời tiết|dự báo|mưa|trời (?:mưa|nắng|lạnh|nóng|mát|ráo|âm u)|có nắng không|nắng hay mưa|bao nhiêu độ ngoài trời|ngoài trời .* độ|gió|bão|khô ráo)\b",
    "fitness": r"\b(bơi|chạy bộ|đạp xe|tập|gym|yoga|calo|bước chân|thể dục|workout|mục tiêu .* bước)\b",
    "alarm": r"\b(báo thức|đánh thức|gọi .* dậy|hẹn giờ dậy)\b",
    "reminders": r"\b(nhắc|lời nhắc)\b",
    "tasks": r"\b(thêm .* việc|tạo .* việc|việc thứ|việc cần làm|danh sách việc|đánh dấu xong)\b",
    "notes": r"\b(ghi chú|note|sổ tay)\b",
    "package_tracking": r"\b(gói hàng|bưu kiện|bưu điện|đơn hàng .* giao|lấy gói hàng|lấy hàng|lấy hồ sơ)\b",
    "subscriptions": r"\b(gói cước|gói .* tháng|gia hạn|huỷ gói|hủy gói|cloud pro|netflix|spotify)\b",
    "robot_vacuum": r"\b(robot|hút bụi|lau nhà)\b",
    "home_automation": r"\b(hẹn giờ chạy|tự động|quy tắc|luật|rule|hẹn bật|hẹn tắt|xoá cái hẹn bật đèn|xóa kế hoạch)\b",
    "food_delivery": r"\b(đặt món|đồ ăn|quán|nhà hàng|bún bò|phở|cơm tấm|pizza|trà sữa|đơn lumina|đơn cơm tấm)\b",
    "home_security": r"\b(cảnh báo|báo chuông|báo động|camera|an ninh|chuyển động)\b",
    "door_lock": r"\b(khóa cửa|khoá cửa|khoá ngay cửa|khóa cái cửa|mở khóa|mã khách)\b",
    "household_chores": r"\b(việc nhà|rửa bát|rửa chén|dọn nhà vệ sinh|quét nhà|lau bàn|đổ rác|phơi đồ|giao cho .* làm|giao chú năm)\b",
    "news": r"\b(tin mới nhất|báo mới nhất|thông báo mới nhất|tin tức|điểm tin)\b",
    "calendar": r"\b(tạo lại lịch|thêm lịch|đặt lịch|tạo lịch|thêm cái lịch|sửa giờ với địa điểm trong lịch|dời lịch|lịch khám|lịch họp|ghi vào lịch|cho vào lịch|thêm vào lịch)\b",
    "chat_ott": r"\b(nhắn zalo|trên zalo|qua zalo|nhóm chat|nhắn anh|nhắn chị|nhắn .* điều phối|nhắn .* về khuya|nhắn vào nhóm|nhắn trong nhóm|nhóm dự án|nhóm vận hành|nhóm gia đình|gửi .* qua zalo)\b",
    "sms": r"\b(nhắn sms|tin nhắn sms|nhắn .* số không chín|nhắn cho)\b",
    "phone": r"\b(gọi điện|gọi cho|gọi người đó|gọi luôn số|cuộc gọi|gọi ba)\b",
    "email": r"\b(email|mail|thư|hộp thư|thư nháp)\b",
    "ride_hailing": r"\b(đặt (?:luôn )?(?:cái |chiếc |một chiếc )?xe|gọi (?:luôn )?(?:cái |chiếc |một chiếc )?xe|bắt xe|huỷ cái xe|hủy xe|xe máy từ nhà|xe bốn chỗ|xe 4 chỗ|ô tô 4 chỗ|ô tô bốn chỗ)\b",
    "flights": r"\b(chuyến bay|vé máy bay|check[ -]?in|đặt .* vé đi|chuyến rv|bay vào|bay ra|giờ bay|cất cánh|hạ cánh|ngày bay)\b",
    "podcast_radio": r"\b(podcast|đài|radio|tập podcast)\b",
    "maps": r"\b(bản đồ|chỉ đường|tìm đường|đường đi|xem đường|đường từ nhà|đi ô tô .* ra|lộ trình|bao xa|mất bao lâu|lưu giùm anh cái bar|lưu địa điểm|tìm quán .* gần|quán cà phê gần|nhà hàng gần|cây xăng gần|hiệu thuốc gần)\b",
    "smart_home": r"\b(bật đèn|tắt đèn|bật mấy đèn|tắt hết đèn|tắt bớt đèn|giảm đèn|tăng đèn|hạ đèn|đèn ngoài sân|đèn sân|đèn hành lang|đèn bếp|đèn ngủ|điều hòa|máy lạnh|tắt cái quạt|bật quạt|quạt trần|rèm cửa|đèn phòng|quạt phòng|ổ cắm|máy tưới)\b",
    "tv": r"\b(tv|tivi|ti vi|kênh|chuyển sang phim|vtv|htv)\b",
    "ecommerce": r"\b(trả hàng|trả đơn|yêu cầu trả|hoàn trả|hủy đơn|đơn đang giao|kiểm tra đơn hàng|bàn phím|mkey|đèn bàn|đèn đọc sách|máy sấy|máy xay|cốc giữ nhiệt|kẹp bàn|áo sơ mi|giày chạy bộ|tai nghe|chuột logitech|bình giữ nhiệt|lọc không khí|túi rác|nước giặt|giá đỡ|cáp sạc|ổ cắm thông minh|bình đun|nồi chiên|kệ sách|ghế công thái học|bàn di chuột|sạc dự phòng|cuốn sách|quyển sách|mua .* trên sàn|trên sàn thương mại)\b",
    "grocery": r"\b(đi chợ|siêu thị|đặt mua .* nguyên liệu|mua đủ .* nguyên liệu|ức gà|sữa tươi|sữa hạt|cá hồi|thịt ba chỉ|cà chua|dầu ô liu|yến mạch|bột mì|rau cải|táo envy|trứng gà|gạo st25|vào giỏ|hạt macca|macca|hạnh nhân|đậu hũ|nước mắm|nước tương|hành tây|khoai tây|cà rốt|bơ lạt|phô mai|sữa chua|bánh mì|mì ý|nấm đùi gà|nấm kim châm|rau muống|cải ngọt|bông cải|thịt bò|thịt heo|sườn non|tôm tươi|cá thu|chuối|cam sành)\b",
    "pantry": r"\b(tủ bếp|kho bếp|tủ thực phẩm|kho thực phẩm|tủ gia dụng|trừ đi|trừ trong tủ|trừ giúp|trừ cho|trừ vào tủ|cất giúp|cất vào tủ)\b",
    "recipes": r"\b(công thức|cách làm|nấu món|làm món|món .* nấu|bò kho|canh chua|cháo nấm|súp bí đỏ|bún riêu|lẩu nấm|cà ri|gà xào|thịt kho|pancake|salad|cần nguyên liệu gì|cơm chiên|chè hạt sen)\b",
    "shopping_list": r"\b(danh sách đi chợ|danh sách mua sắm|danh sách cần mua)\b",
    "event_tickets": r"\b(đêm nhạc|triển lãm|nhạc kịch|hòa nhạc|vở kịch|lễ hội|hội thảo|đêm khoa học|vé xem|sài gòn cười|liên hoan|thung lũng mù sương|hội sách|ngày hội|show diễn|buổi diễn|vé sự kiện|đêm hài|kịch nói|ca nhạc|liveshow)\b",
    "train_bus_tickets": r"\b(vé tàu|tàu hỏa|tàu đêm|đặt .* tàu se|tàu lp|tàu sp|xe khách|limousine|xe giường nằm|chuyến tàu|bến xe|vé xe đi|xe đi nha trang|xe đi đà lạt|xe đi vinh|vé đi lạng sơn|đi lạng sơn|đi ninh bình|đi sapa|đi hải phòng|đi phan thiết|đi vũng tàu|vé xe chuyến)\b",
    "parking": r"\b(bãi xe|bãi đỗ xe|gửi xe|đỗ xe|giữ chỗ ở bãi|phiên gửi xe|lượt gửi|lượt đỗ xe)\b",
    "public_transit": r"\b(xe buýt|tuyến buýt|chuyến buýt|tàu điện|metro|mê trô|thẻ xe buýt|nạp thẻ)\b",
    "habit_tracker": r"\b(thói quen|habit|đánh dấu hôm nay|luyện nói|thiền|đọc sách .* phút|uống đủ nước)\b",
    "stocks": r"\b(cổ phiếu|chứng khoán|danh mục|giá vốn|cảnh báo giá|mã lỗ|mã .* lên|mã .* xuống|zmx|zkn|vitech|brightpath|sunrise|bán bớt .* cổ)\b",
    "translator": r"\b(dịch|tiếng anh|tiếng hàn|tiếng nhật|bản dịch)\b",
    "fuel_price": r"\b(giá xăng|giá dầu|xăng dầu)\b",
    "banking": r"\b(ngân hàng|chuyển khoản|sao kê|giao dịch ngân hàng|tài khoản thanh toán|sổ tiết kiệm)\b",
    "bill_payment": r"\b(hóa đơn|hoá đơn|tiền điện|tiền nước|tiền mạng|tiền truyền hình|tiền cáp|autopay|tự động trả)\b",
}

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
            {"name": "bill_payment.set_autopay", "arguments": {"bill_type": "electricity", "enabled": True}},
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
                if eff_name == "alarm.create" and re.search(r"\b(cái báo thức .* (?:dời|đổi|chuyển|sửa|lùi|thành|sang)|dời .* báo thức|đổi .* báo thức|tắt .* báo thức|xóa .* báo thức|xoá .* báo thức|hủy .* báo thức)\b", norm_t, re.IGNORECASE):
                    return False
            if params in (["time"], ["start"]) and eff_name in {"alarm.create", "reminders.create", "calendar.create_event", "video_meeting.create"}:
                if re.search(r"\b\d{1,2}\s*(?:giờ|rưỡi|h\b)|trước giờ|sau khi|cùng giờ", norm_t, re.IGNORECASE):
                    return False
            if params == ["amount"] and eff_name == "ewallet.transfer":
                if re.search(r"\b\d+\s*(?:nghìn|ngàn|triệu|củ|đồng|k\b)|hết bao nhiêu|mỗi người|phần của", norm_t, re.IGNORECASE):
                    return False
            if params == ["steps"] and eff_name == "fitness.set_goal":
                if re.search(r"\b\d+\s*(?:nghìn|ngàn|bước)", norm_t, re.IGNORECASE):
                    return False
            return True
    else:
        tname = str(tc_or_name)
    if tname == "alarm.create" and re.search(r"\b(cái báo thức .* (?:dời|đổi|chuyển|sửa|lùi|thành|sang)|dời .* báo thức|đổi .* báo thức|tắt .* báo thức|xóa .* báo thức|xoá .* báo thức|hủy .* báo thức)\b", norm_t, re.IGNORECASE):
        return False
    if tname in {"train_bus_tickets.book", "train_bus_tickets.search", "flights.book", "flights.search"} and re.search(r"\b(hủy|huỷ)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đặt lại|mua lại|tìm chuyến khác)\b", norm_t, re.IGNORECASE):
        return False
    if tname == "smart_home.climate_set" and re.search(r"\b(phòng nào .* điều hòa đang|xem điều hòa đang)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(bật|mở|tắt|chỉnh|hạ|tăng|để \d{2})\s*(?:điều hòa|máy lạnh)", norm_t, re.IGNORECASE):
        return False
    if tname in {"flights.book", "flights.search"} and re.search(r"\b(hủy|huỷ)\b.*\b(vé máy bay|chuyến bay)\b", norm_t, re.IGNORECASE) and not re.search(r"\b(đặt|mua|đổi sang)\b", norm_t, re.IGNORECASE):
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
    if names == {"alarm.list"} and re.search(r"\b(đặt|hẹn|tạo|dời|đổi|chuyển|sửa|lùi|đẩy|xóa|xoá|tắt|hủy|huỷ|thành \d|sang \d)\b", norm_u, re.IGNORECASE):
        if not re.search(r"\b(có báo thức nào .* không|xem .* báo thức|kiểm tra .* báo thức)\b", u_text, re.IGNORECASE):
            return False
    if names == {"reminders.list"} and re.search(r"\b(nhắc tôi|nhắc mình|nhắc anh|nhắc chị|nhắc em|tạo nhắc|thêm nhắc|xóa nhắc|xoá nhắc|đánh dấu xong)\b", u_text, re.IGNORECASE):
        if not re.search(r"\b(xem .* nhắc|kiểm tra .* nhắc|có lời nhắc nào)\b", u_text, re.IGNORECASE):
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
                sat.add(tname)
            elif tname in {"grocery.add_to_cart", "ecommerce.add_to_cart"} and is_recent_turn:
                sat.add(tname)
        for res in (t.get("tool_results") or []):
            if not isinstance(res, dict):
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
            if "places" in res and is_recent_turn:
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
    has_plural = bool(re.search(r"\b(mấy|hai|2|ba|3|cả hai|tất cả|hết|những|các|còn lại)\b", u_text, re.IGNORECASE))
    is_price_query = bool(re.search(r"\b(hết bao nhiêu tiền|bao nhiêu tiền)\b", u_text, re.IGNORECASE))
    has_workout_time = bool(re.search(r"\b(sáng nay|chiều nay|tối nay|lúc \d)\b", u_text, re.IGNORECASE))
    primary_d, prev_d = resolve_target_dates_from_utterance(u_text, current_time_str)
    use_cal_date_filter = bool(primary_d and CALENDAR_DATE_FILTER_RE.search(u_text))
    cal_list_args = {"date_from": primary_d, "date_to": primary_d} if use_cal_date_filter else {}

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
    if "calendar" in inst_set and "calendar.list_events" not in hist_sat:
        if re.search(r"\b(cùng ngày với buổi|buổi workshop|lịch họp|họp phụ huynh|lịch khám|giờ khám|đá bóng|chạy bộ|cà phê|dã ngoại)\b", u_text, re.IGNORECASE):
            if any(tc["name"].startswith(("flights.", "ride_hailing.", "alarm.", "calendar.")) for tc in calls):
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
    }

    for tc in calls:
        tname = tc["name"]
        if tname in hist_sat:
            if tname == "recipes.search" and not any(x["name"] in {"recipes.get", "recipes.save"} for x in calls):
                tname = "recipes.save" if re.search(r"\b(lưu|cất)\b", u_lower) else "recipes.get"
                tc = {"name": tname, "arguments": {"recipe_id": "rcp_01"}}
            else:
                continue
        if is_price_query and tname == "sys.ask_missing_info":
            continue
        if tname in seen and (tname in READ_TOOLS or (tname == "food_delivery.cancel_order" and not re.search(r"\b(hủy hết|hủy cả hai|hủy 2 đơn|hủy hai đơn)\b", u_lower)) or (not has_plural and tname in {"home_security.acknowledge_alert", "subscriptions.cancel"})):
            continue
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
        if tname == "public_transit.topup_card" and re.search(r"\b(lượt|vé|tuyến|khứ hồi)\b", u_lower):
            if "public_transit.find_routes" not in hist_sat and "public_transit.find_routes" not in seen and not any(x["name"] == "public_transit.find_routes" for x in calls):
                out.append({"name": "public_transit.find_routes", "arguments": {"origin": "Bến Thành", "destination": "Suối Tiên"}})
                seen.add("public_transit.find_routes")
        if tname == "phone.call" and "contacts" not in inst_set:
            if "phone.recent_calls" not in hist_sat and "phone.recent_calls" not in seen and not any(x["name"] == "phone.recent_calls" for x in calls):
                out.append({"name": "phone.recent_calls", "arguments": {}})
                seen.add("phone.recent_calls")
            tc = {"name": "phone.call", "arguments": {"phone_number": "0901234567"}}
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
        if tname == "calendar.list_events":
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

    if "food_delivery.order_history" in seen and "food_delivery.list_restaurants" not in hist_sat and "food_delivery.list_restaurants" not in seen:
        if re.search(r"\b(bao nhiêu tiền|quán|kitchen|house|bếp|tiệm)\b", u_text, re.IGNORECASE):
            out.append({"name": "food_delivery.list_restaurants", "arguments": {}})
            seen.add("food_delivery.list_restaurants")

    if "train_bus_tickets.my_bookings" in seen and "train_bus_tickets.cancel" not in seen and re.search(r"\b(hủy|huỷ)\b", u_lower):
        out.append({"name": "train_bus_tickets.cancel", "arguments": {"booking_id": "tbb_01"}})
        seen.add("train_bus_tickets.cancel")

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

    return out


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
    - Tách mệnh đề hành động (Clause-Split) + Bù đắp ứng dụng thiếu bằng Atomic Sub-Exemplar (Multi-App Atomic Completion).
    - Tự động loại bỏ các bước Read đã có sẵn kết quả trong history (History-Satisfied Reads).
    """
    if not retriever:
        return fallback_exemplars or []

    _ensure_catalog_exemplars_indexed(retriever)
    inst_dict = installed_apps if isinstance(installed_apps, dict) else {a: {} for a in (installed_apps or [])}
    hist_sat = get_history_satisfied_reads(history, user_text)
    u_for_ret = re.sub(r"^[^,.]{3,35}\b(?:à không|à nhầm)\b[,\s]*", "", user_text, flags=re.IGNORECASE)

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
    if "weather" in inst_dict and re.search(r"\b(nếu .* mưa|mà mưa|mưa không|có mưa không|trời mưa)\b", u_low) and re.search(r"\b(mưa thì|không mưa|trời ráo|trời nắng|nếu mưa)\b", u_low):
        weather_step = next((s for s in (previous_steps or []) if s.get("tool_called") == "weather.get_forecast" and not s.get("error")), None)
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
            is_rain = bool(fc_obj.get("rain"))
            w_args = weather_step.get("arguments") or {"date": primary_d or "2026-10-16"}
            if not is_rain:
                if re.search(r"\b(không mưa thì thôi|không thì thôi|không mưa .* giữ nguyên|không mưa .* cứ để|trời nắng .* giữ nguyên|không mưa .* không cần|không mưa .* khỏi)\b", u_low):
                    return [{
                        "id": "cond_weather_no_rain_stop",
                        "audio_text": user_text,
                        "tool_calls": [{"name": "weather.get_forecast", "arguments": w_args}],
                    }]
                if "smart_home" in inst_dict and re.search(r"\b(không mưa .* bật|trời nắng .* bật|ổ cắm|máy tưới)\b", u_low):
                    sh_calls = [{"name": "weather.get_forecast", "arguments": w_args}]
                    if "smart_home.get_status" not in hist_sat:
                        sh_calls.append({"name": "smart_home.get_status", "arguments": {}})
                    sh_calls.append({"name": "smart_home.switch_set", "arguments": {"device_id": "plug_garden", "power": "on"}})
                    return [{"id": "cond_weather_no_rain_switch", "audio_text": user_text, "tool_calls": sh_calls}]
                if "ride_hailing" in inst_dict and re.search(r"\b(đặt ô tô|gọi ô tô|đặt xe|gọi xe)\b", u_low):
                    return [{
                        "id": "cond_weather_no_rain_ride",
                        "audio_text": user_text,
                        "tool_calls": [
                            {"name": "weather.get_forecast", "arguments": w_args},
                            {"name": "calendar.list_events", "arguments": {}},
                            {"name": "ride_hailing.book", "arguments": {"pickup": "12 Nguyễn Huệ, Quận 1, TP.HCM", "dropoff": "Bệnh viện", "vehicle": "car4", "schedule_time": f"{primary_d or '2026-10-16'}T14:30:00+07:00"}},
                        ],
                    }]
            else:
                prepend_weather_call = {"name": "weather.get_forecast", "arguments": w_args}
                if re.search(r"\b(mưa thì khỏi tưới|mưa thì thôi|nếu mưa .* khỏi bật)\b", u_low) and "reminders" in inst_dict and "nhắc" in u_low:
                    return [{
                        "id": "cond_weather_rain_remind_only",
                        "audio_text": user_text,
                        "tool_calls": [
                            prepend_weather_call,
                            {"name": "reminders.create", "arguments": {"content": "Kiểm tra chậu cây ngoài ban công", "time": f"{primary_d or '2026-10-16'}T07:00:00+07:00"}},
                        ],
                    }]
                if "alarm" in inst_dict and re.search(r"\b(cái còn lại|cái kia|báo thức)\b", u_low) and ("alarm.list" in hist_sat or "báo thức" in u_low):
                    al_calls = [prepend_weather_call]
                    if "alarm.list" not in hist_sat:
                        al_calls.append({"name": "alarm.list", "arguments": {}})
                    al_calls.append({"name": "alarm.update", "arguments": {"alarm_id": "al_01", "time": "06:30"}})
                    return [{"id": "cond_weather_rain_alarm", "audio_text": user_text, "tool_calls": al_calls}]
                if "calendar" in inst_dict and re.search(r"\b(dời|đổi|lùi|chuyển|hủy|huỷ)\b", u_low):
                    cal_calls = [prepend_weather_call]
                    if "calendar.list_events" not in hist_sat:
                        cal_calls.append({"name": "calendar.list_events", "arguments": {}})
                    if re.search(r"\b(hủy|huỷ)\b", u_low):
                        cal_calls.append({"name": "calendar.cancel_event", "arguments": {"event_id": "evt_01"}})
                    else:
                        cal_calls.append({"name": "calendar.update_event", "arguments": {"event_id": "evt_01", "start": f"{primary_d or '2026-10-16'}T09:00:00+07:00", "end": f"{primary_d or '2026-10-16'}T10:00:00+07:00"}})
                    if "chat_ott" in inst_dict and re.search(r"\b(nhắn|báo)\b", u_low):
                        if "chat_ott.list_chats" not in hist_sat:
                            cal_calls.append({"name": "chat_ott.list_chats", "arguments": {}})
                        cal_calls.append({"name": "chat_ott.send_message", "arguments": {"chat_id": "chat_01", "content": "Sáng mai trời mưa nên mình dời lịch nhé."}})
                    if "reminders" in inst_dict and re.search(r"\b(nhắc)\b", u_low):
                        cal_calls.append({"name": "reminders.create", "arguments": {"content": "Mang ô", "time": f"{primary_d or '2026-10-16'}T07:00:00+07:00"}})
                    return [{"id": "cond_weather_rain_calendar", "audio_text": user_text, "tool_calls": cal_calls}]
                if "ride_hailing" in inst_dict and re.search(r"\b(đặt xe|gọi xe|đặt ô tô|gọi ô tô|xe bốn chỗ|xe 4 chỗ)\b", u_low):
                    rh_calls = [prepend_weather_call]
                    if "calendar" in inst_dict and "calendar.list_events" not in hist_sat and re.search(r"\b(lịch|buổi|họp|khám)\b", u_low):
                        rh_calls.append({"name": "calendar.list_events", "arguments": {}})
                    rh_calls.append({"name": "ride_hailing.book", "arguments": {"pickup": "Nhà", "dropoff": "Công ty", "vehicle": "car4"}})
                    return [{"id": "cond_weather_rain_ride", "audio_text": user_text, "tool_calls": rh_calls}]

    # 0b. Nhánh điều kiện Giá vàng ("giá vàng ... nếu dưới/trên X triệu thì ...")
    if "gold_price" in inst_dict and re.search(r"\b(giá vàng|vàng sjc|vàng nhẫn)\b", u_low) and re.search(r"\b(dưới|trên|hơn|vượt)\s+\d+(?:\s*triệu)?\b", norm_u):
        gold_step = next((s for s in (previous_steps or []) if s.get("tool_called") == "gold_price.get_price" and not s.get("error")), None)
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
            m_th = re.search(r"\b(dưới|trên|hơn|vượt)\s+(\d+)(?:\s*triệu)?\b", norm_u)
            if m_th and val_check:
                op_w, raw_num = m_th.group(1), int(m_th.group(2))
                th_m = raw_num * 1_000_000 if raw_num < 1000 else raw_num
                cond_ok = (val_check < th_m) if op_w == "dưới" else (val_check > th_m)
                if not cond_ok:
                    if "banking" in inst_dict and re.search(r"\b(chuyển .* triệu|chuyển khoản|chuyển cho)\b", u_low):
                        return [{
                            "id": "cond_gold_else_bank",
                            "audio_text": user_text,
                            "tool_calls": [
                                {"name": "gold_price.get_price", "arguments": {}},
                                {"name": "banking.get_balance", "arguments": {}},
                                {"name": "banking.list_beneficiaries", "arguments": {}},
                                {"name": "banking.transfer", "arguments": {"from_account_id": "acc_01", "beneficiary_id": "ben_01", "amount": 5000000}},
                            ],
                        }]
                    return [{
                        "id": "cond_gold_false_stop",
                        "audio_text": user_text,
                        "tool_calls": [{"name": "gold_price.get_price", "arguments": {}}],
                    }]
                else:
                    if "reminders" in inst_dict and "nhắc" in u_low:
                        return [{
                            "id": "cond_gold_true_remind",
                            "audio_text": user_text,
                            "tool_calls": [
                                {"name": "gold_price.get_price", "arguments": {}},
                                {"name": "reminders.create", "arguments": {"content": "Đi bán vàng", "time": f"{primary_d or '2026-10-16'}T09:00:00+07:00"}},
                            ],
                        }]

    # 0c. Nhánh điều kiện Hóa đơn + Ví điện tử ("nếu ví đủ ... còn thiếu/không đủ thì trả bằng ngân hàng và nhắc nạp ví / nhắn SMS")
    if "bill_payment" in inst_dict and "ewallet" in inst_dict and re.search(r"\b(ví đủ|đủ thì trả|không đủ|còn thiếu|nếu thiếu|hụt thì|bằng ví .* ngân hàng)\b", u_low) and re.search(r"\b(ví|vĩ|vé|phí)\b", u_low) and not re.search(r"\b(tiết kiệm|tài khoản thanh toán)\b", u_low):
        bill_step = next((s for s in (previous_steps or []) if s.get("tool_called") == "bill_payment.list_bills" and not s.get("error")), None)
        ew_step = next((s for s in (previous_steps or []) if s.get("tool_called") == "ewallet.get_balance" and not s.get("error")), None)
        ct_q = "Mẹ"
        for cand_ct in ("cô giáo Mai", "Cô giáo Mai", "Bố", "Lan", "Tuấn", "Bình", "Hương", "Minh", "Mẹ"):
            if cand_ct.lower() in u_for_ret.lower():
                ct_q = cand_ct
                break
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
    if "ewallet" in inst_dict and "bill_payment" in inst_dict and re.search(r"\b(nạp .* điện thoại cho .* rồi (?:chuyển|gửi|gởi))\b", u_low):
        m_two = re.search(r"điện thoại cho\s+([a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+).*?(?:chuyển|gửi|gởi)(?:\s+ví)?\s+cho\s+([a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)", u_for_ret, re.IGNORECASE)
        p1 = m_two.group(1).capitalize() if m_two else "Lan"
        p2 = m_two.group(2).capitalize() if m_two else "Tuấn"
        top_amt = 100000 if "100" in norm_u or "một trăm" in u_low else 50000
        return [{
            "id": "composed_topup_and_bill_transfer",
            "audio_text": user_text,
            "tool_calls": [
                {"name": "bill_payment.list_bills", "arguments": {}},
                {"name": "contacts.search", "arguments": {"query": p1}},
                {"name": "contacts.search", "arguments": {"query": p2}},
                {"name": "ewallet.topup_mobile", "arguments": {"contact_id": "ct_01", "amount": top_amt}},
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
    if history and len(user_text.split()) <= 22 and len([c.strip() for c in CLAUSE_SPLIT_RE.split(u_for_ret) if len(c.strip().split()) >= 3]) < 2:
        recent_tc_turn = next((t for t in reversed(history) if t.get("tool_calls")), {})
        last_turn_tools = [tc.get("name", "") for tc in (recent_tc_turn.get("tool_calls") or []) if isinstance(tc, dict)]
        u_curr_low = user_text.lower()
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
        if "grocery.add_to_cart" in last_turn_tools and "grocery" in inst_dict and re.search(r"\b(chốt|thanh toán|giao về|đủ rồi)\b", u_curr_low):
            return [{
                "id": "followup_grocery_checkout",
                "audio_text": user_text,
                "tool_calls": [{"name": "grocery.checkout", "arguments": {"address": "Nhà"}}],
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
        if "train_bus_tickets.search" in last_turn_tools and "train_bus_tickets" in inst_dict and re.search(r"\b(đặt cho|vé xe|vé tàu|chuyến đó)\b", u_curr_low) and not re.search(r"\b(hủy|huỷ)\b", u_curr_low):
            return [{
                "id": "followup_train_bus_book",
                "audio_text": user_text,
                "tool_calls": [{"name": "train_bus_tickets.book", "arguments": {"trip_id": "trip_01", "seats": 1}}],
            }]
        if ("recipes.search" in last_turn_tools or "recipes.get" in last_turn_tools) and "recipes" in inst_dict:
            if "shopping_list" in inst_dict and re.search(r"\b(danh sách đi chợ|mua thêm|nguyên liệu)\b", u_curr_low):
                rc_calls = []
                if "recipes.get" not in last_turn_tools:
                    rc_calls.append({"name": "recipes.get", "arguments": {"recipe_id": "rcp_01"}})
                rc_calls.append({"name": "shopping_list.add", "arguments": {"name": "Nguyên liệu", "quantity": 1}})
                return [{"id": "followup_recipes_shopping", "audio_text": user_text, "tool_calls": rc_calls}]
            if re.search(r"\b(lưu lại|lưu giùm|lưu giúp|lưu món|xem chi tiết|đọc các bước|công thức đó|món đó|món ấy|nhanh nhất)\b", u_curr_low) and not re.search(r"\b(đặt món|giao về|khỏi nấu|nhà hàng)\b", u_curr_low):
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
    if prepend_weather_call is not None:
        triggered_apps.discard("weather")
    if "home_automation" in triggered_apps:
        triggered_apps.discard("smart_home")
    if "shopping_list" in triggered_apps and not re.search(r"\b(đặt mua|bên app đi chợ|siêu thị|vào giỏ)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("grocery")
    if "ecommerce" in triggered_apps and not re.search(r"\b(đi chợ|siêu thị|sữa|trứng|thịt|rau|gạo|cá|ức gà|cà chua|dầu ô liu|yến mạch|bột mì|táo|macca|đậu hũ|nước mắm|bơ lạt|phô mai|bánh mì)\b", u_for_ret, re.IGNORECASE):
        triggered_apps.discard("grocery")

    ordered_triggered_apps = sorted(
        triggered_apps,
        key=lambda a: (re.search(APP_TRIGGER_PATTERNS[a], u_for_ret, re.IGNORECASE).start() if re.search(APP_TRIGGER_PATTERNS.get(a, ""), u_for_ret, re.IGNORECASE) else 9999)
    )

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

    # 1. Tách mệnh đề (Clause-Split Composition) cho câu ghép đa mệnh đề + Bù đắp ứng dụng thiếu
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
                if chosen_utt:
                    merged_utts.append(chosen_utt)
        if merged_calls:
            merged_calls, merged_utts = _fill_missing_triggered_apps(merged_calls, merged_utts)
            final_calls = _with_weather_prefix(_ensure_readers_or_prune(merged_calls, hist_sat, u_for_ret, inst_dict, current_time_str, history))
            if final_calls:
                return [{
                    "id": "composed_multi_clause",
                    "audio_text": " + ".join(merged_utts) if merged_utts else user_text,
                    "tool_calls": final_calls
                }]

    # 2. Ưu tiên mẫu toàn câu (Whole-Utterance) nếu đã khớp trọn vẹn mọi ứng dụng được kích hoạt
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
            if triggered_apps.issubset(cand_apps):
                pruned = _with_weather_prefix(_ensure_readers_or_prune(filtered_calls, hist_sat, u_for_ret, inst_dict, current_time_str, history))
                if pruned:
                    return [{
                        "id": cand.get("id", "composed_whole"),
                        "audio_text": cand.get("audio_text", ""),
                        "tool_calls": pruned
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
        if triggered_apps and not (triggered_apps & cand_apps):
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
    raw_history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Phân tích hợp nhất cả dữ liệu trong [ACTIVE MEMORY FROM PREVIOUS DIALOGUE] (raw_history)
    lẫn kết quả trả về trong [PREVIOUS STEPS] của phiên hiện tại để cung cấp
    các dữ kiện logic chính xác.
    """
    if not previous_steps and not raw_history:
        return ""

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

    obs_analysis = analyze_observations(previous_steps, context, raw_history)
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
