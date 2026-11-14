import json
import re
import functools
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Tuple

WEEKDAY_VN = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]

VN_WORD_TO_NUM = [
    ("bốn nghìn một trăm", "4100"), ("bốn ngàn một trăm", "4100"),
    ("hai trăm", "200"), ("ba trăm", "300"), ("bốn trăm", "400"), ("năm trăm", "500"),
    ("sáu trăm", "600"), ("bảy trăm", "700"), ("tám trăm", "800"), ("chín trăm", "900"), ("một trăm", "100"),
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
    ("chín chục", "90"), ("tám chục", "80"), ("bảy chục", "70"), ("sáu chục", "60"),
    ("năm chục", "50"), ("bốn chục", "40"), ("ba chục", "30"), ("hai chục", "20"), ("một chục", "10"),
    ("mười", "10"), ("một", "1"), ("hai", "2"), ("ba", "3"), ("bốn", "4"),
    ("năm", "5"), ("sáu", "6"), ("bảy", "7"), ("tám", "8"), ("chín", "9"),
    ("không", "0"),
]

_COMPILED_VN_WORD_TO_NUM = [
    (
        re.compile(
            rf"(?<!bà\s)(?<!ông\s)(?<!chú\s)(?<!bác\s)(?<!cô\s)(?<!anh\s)(?<!chị\s)\b{w}\b"
            if " " not in w
            else rf"\b{w}\b"
        ),
        d,
    )
    for w, d in VN_WORD_TO_NUM
]
_DIGIT_TRAM_TWO_DIGIT_RE = re.compile(r"\b([1-9])\s*trăm\s+(\d{2})\b")
_DIGIT_TRAM_LINH_RE = re.compile(r"\b([1-9])\s*trăm\s+(?:linh|lẻ)\s*(\d)\b")
_DIGIT_TRAM_RE = re.compile(r"\b([1-9])\s*trăm\b")
_HUNDRED_TWO_DIGIT_RE = re.compile(r"\b([1-9])00\s+(\d{2})\b")
_HUNDRED_LINH_RE = re.compile(r"\b([1-9])00\s+(?:linh|lẻ)\s*(\d)\b")
_TRAM_TWO_DIGIT_RE = re.compile(r"\btrăm\s+(\d{2})\b")
_SPOKEN_PHONE_DIGITS_RE = re.compile(r"\b(?:0|1\s*[89]\s*0\s*0)(?:[\s,.-]*\d){4,10}\b")


@functools.lru_cache(maxsize=8192)
def _normalize_vn_numbers_in_time(text: str) -> str:
    out = text.lower()
    out = _DIGIT_TRAM_TWO_DIGIT_RE.sub(lambda m: str(int(m.group(1)) * 100 + int(m.group(2))), out)
    out = _DIGIT_TRAM_LINH_RE.sub(lambda m: str(int(m.group(1)) * 100 + int(m.group(2))), out)
    out = _DIGIT_TRAM_RE.sub(lambda m: f"{m.group(1)}00", out)
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
                    "điều phối", "bác sĩ gia đình", "thú y", "kế toán", "ké toán", "nhân sự",
                    "kỹ thuật", "trực kho", "pháp chế", "phát chế", "vận hành", "kinh doanh", "hành chính"
                )
                rel_matches = []
                for c in contacts:
                    c_rel = str(c.get("relation") or "").strip().lower()
                    if not c_rel:
                        continue
                    if c_rel in u_curr_only:
                        rel_matches.append(c)
                        continue
                    for role_kw in explicit_roles:
                        norm_role = "kế toán" if role_kw == "ké toán" else ("pháp chế" if role_kw == "phát chế" else role_kw.replace("bên ", ""))
                        if role_kw in u_curr_only and norm_role in c_rel:
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
                if chosen_unique is None and raw_history:
                    hist_parts_ct = []
                    for h_turn in raw_history:
                        if not isinstance(h_turn, dict):
                            continue
                        hist_parts_ct.append(str(h_turn.get("audio_text") or h_turn.get("text") or ""))
                        for tc_h in (h_turn.get("tool_calls") or []):
                            if isinstance(tc_h, dict) and tc_h.get("name") != "contacts.search":
                                hist_parts_ct.append(json.dumps(tc_h.get("arguments") or {}, ensure_ascii=False))
                        for tr_h in (h_turn.get("tool_results") or []):
                            if isinstance(tr_h, dict) and "contacts" not in tr_h:
                                hist_parts_ct.append(json.dumps(tr_h, ensure_ascii=False))
                    hist_blob_low = " ".join(hist_parts_ct).lower()
                    hist_ct_matches = []
                    for c in contacts:
                        c_id_l = str(c.get("id") or "").strip().lower()
                        c_nm_l = str(c.get("name") or "").strip().lower()
                        c_ph_l = str(c.get("phone") or "").strip().lower()
                        c_em_l = str(c.get("email") or "").strip().lower()
                        c_rl_l = str(c.get("relation") or "").strip().lower()
                        if (
                            (c_id_l and f'"{c_id_l}"' in hist_blob_low)
                            or (c_nm_l and len(c_nm_l.split()) >= 2 and c_nm_l in hist_blob_low)
                            or (c_ph_l and len(c_ph_l) >= 8 and c_ph_l in hist_blob_low)
                            or (c_em_l and "@" in c_em_l and c_em_l in hist_blob_low)
                            or (c_rl_l and len(c_rl_l) >= 4 and c_rl_l in hist_blob_low)
                        ):
                            hist_ct_matches.append(c)
                    if len(hist_ct_matches) == 1:
                        chosen_unique = hist_ct_matches[0]
                if chosen_unique is not None:
                    facts.append(
                        f"Liên hệ khớp duy nhất theo mô tả của người dùng: contact_id=\"{chosen_unique.get('id')}\", tên=\"{chosen_unique.get('name')}\", "
                        f"relation=\"{chosen_unique.get('relation', '')}\", phone=\"{chosen_unique.get('phone')}\", email=\"{chosen_unique.get('email', '')}\" "
                        f"-> BẮT BUỘC chọn `\"contact_id\": \"{chosen_unique.get('id')}\"` (KHÔNG gọi `sys.ask_missing_info`!)."
                    )
                else:
                    c_list = [f"{c.get('id')} ({c.get('name')}, quan_hệ={c.get('relation', '')}, SĐT={c.get('phone')})" for c in contacts]
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
            ac_on_rooms = {
                d.get("room") for d in devs
                if d.get("type") in ("ac", "climate", "air_conditioner")
                and (
                    d.get("power") == "on"
                    or (isinstance(d.get("state"), dict) and (d["state"].get("power") == "on" or d["state"].get("mode", "off") != "off"))
                )
            }
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


