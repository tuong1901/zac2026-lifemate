import json
import math
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Any, Optional

class CatalogFilter:
    """
    Bộ lọc công cụ: Chỉ giữ lại các tools thuộc danh sách installed_apps của từng mẫu,
    kèm theo sys tool (như sys.ask_missing_info).
    Giúp LLM không bị ngợp context và tránh gọi tool của app chưa cài.
    """
    def __init__(self, catalog_path: str):
        with open(catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)
            
        self.apps_by_id = {app["id"]: app for app in self.catalog.get("apps", [])}

    def get_allowed_tools(self, installed_apps: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        installed_apps có dạng: { "weather": {"tools": [...]}, "calendar": ... }
        """
        allowed = []
        installed_ids = set(installed_apps.keys())
        # sys tool luôn luôn được phép gọi
        installed_ids.add("sys")

        # A set has process-randomized iteration order. Tool order changes the prompt
        # and can change greedy LLM output, so keep it stable across runs.
        for app_id in sorted(installed_ids):
            if app_id in self.apps_by_id:
                app_info = self.apps_by_id[app_id]
                for tool in app_info.get("tools", []):
                    # Nếu installed_apps chỉ định danh sách tool cụ thể
                    if app_id in installed_apps and "tools" in installed_apps[app_id]:
                        specific_tools = installed_apps[app_id]["tools"]
                        if specific_tools and tool["name"] not in specific_tools:
                            continue
                    allowed.append(tool)
        return sorted(allowed, key=lambda tool: tool["name"])

    def format_tools_for_prompt(self, allowed_tools: List[Dict[str, Any]]) -> str:
        """
        Định dạng danh sách tool thành chuỗi ngắn gọn, tối ưu token context cho LLM.
        """
        lines = []
        for t in allowed_tools:
            name = t["name"]
            desc = t.get("description", "")
            params = t.get("parameters", {}).get("properties", {})
            required = t.get("parameters", {}).get("required", [])
            
            param_str = []
            for p_name, p_info in params.items():
                req_tag = "*" if p_name in required else ""
                p_type = p_info.get("type", "any")
                if "enum" in p_info:
                    p_type = f"enum={p_info['enum']}"
                param_str.append(f"{p_name}{req_tag}: {p_type}")
                
            lines.append(f"- `{name}` | {desc} | Tham số: [{', '.join(param_str)}]")
        return "\n".join(lines)

    def select_prompt_tools(
        self,
        allowed_tools: List[Dict[str, Any]],
        exemplars: List[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None,
        query_text: str = "",
        max_tools: int = 32,
    ) -> List[Dict[str, Any]]:
        """Build a compact prompt catalog without narrowing execution rights.

        Retrieved examples act as a cheap intent router. Every tool from the
        routed apps is retained, together with read tools referenced by schema
        ``x-ref`` fields. If retrieval has no usable evidence, the full catalog
        is the safer fallback.
        """
        if len(allowed_tools) <= max_tools:
            return list(allowed_tools)

        seed_names = set()
        exemplar_scores = []
        for exemplar in exemplars or []:
            if isinstance(exemplar.get("bm25_score"), (int, float)):
                exemplar_scores.append(float(exemplar["bm25_score"]))
            for call in exemplar.get("tool_calls", []) or []:
                name = call.get("name", "") if isinstance(call, dict) else ""
                if name:
                    seed_names.add(name)
                if name == "sys.ask_missing_info":
                    target = (call.get("arguments") or {}).get("tool_name", "")
                    if target:
                        seed_names.add(target)

        for turn in (history or [])[-2:]:
            for call in turn.get("tool_calls", []) or []:
                if isinstance(call, dict) and call.get("name"):
                    seed_names.add(call["name"])
                    if call["name"] == "sys.ask_missing_info":
                        target = (call.get("arguments") or {}).get("tool_name", "")
                        if target:
                            seed_names.add(target)

        exemplar_apps = {name.split(".", 1)[0] for name in seed_names if "." in name}
        lexical_apps = self._lexical_apps(query_text, allowed_tools)
        routed_apps = exemplar_apps | lexical_apps

        branch_count = len(re.findall(
            r"\b(?:và|rồi|xong|sau đó|đồng thời|nếu|không thì|còn nếu)\b",
            unicodedata.normalize("NFC", query_text.lower()),
        ))
        expected_apps = min(3, branch_count + 1)
        weak_exemplar = not exemplar_scores or max(exemplar_scores) < 2.5
        ambiguous_reference = bool(re.search(
            r"\b(?:cái đó|việc đó|chuyến đó|đơn đó|lần trước|lúc nãy|hồi nãy|anh ấy|chị ấy)\b",
            unicodedata.normalize("NFC", query_text.lower()),
        ))

        # Prefer recall when the router has weak evidence. A larger prompt is
        # cheaper than making the correct tool impossible for the planner to see.
        if (
            not routed_apps
            or (not lexical_apps and weak_exemplar)
            or (branch_count and len(routed_apps - {"sys"}) < expected_apps)
            or (ambiguous_reference and not history)
        ):
            return list(allowed_tools)

        selected = [
            tool for tool in allowed_tools
            if tool.get("name") == "sys.ask_missing_info"
            or tool.get("name") in seed_names
            or tool.get("name", "").split(".", 1)[0] in routed_apps
        ]

        referenced_apps = set()
        for tool in selected:
            properties = (tool.get("parameters") or {}).get("properties") or {}
            for info in properties.values():
                ref = info.get("x-ref", "") if isinstance(info, dict) else ""
                if isinstance(ref, str) and "." in ref:
                    referenced_apps.add(ref.split(".", 1)[0])
        if referenced_apps:
            selected_names = {tool.get("name") for tool in selected}
            selected.extend(
                tool for tool in allowed_tools
                if tool.get("name") not in selected_names
                and tool.get("kind") == "read"
                and tool.get("name", "").split(".", 1)[0] in referenced_apps
            )

        # Avoid a destructive hard cap: complete routed domains are safer than
        # silently hiding sibling operations from the planner.
        return sorted(selected, key=lambda tool: tool.get("name", ""))

    def _lexical_apps(
        self,
        query_text: str,
        allowed_tools: List[Dict[str, Any]],
    ) -> set:
        """Route domains from rare token overlap with catalog descriptions."""
        query_tokens = self._routing_tokens(query_text)
        if not query_tokens:
            return set()

        docs: Dict[str, set] = {}
        apps_by_id = getattr(self, "apps_by_id", {})
        for tool in allowed_tools:
            name = str(tool.get("name", ""))
            app_id = name.split(".", 1)[0]
            if app_id == "sys":
                continue
            app = apps_by_id.get(app_id, {})
            parts = [
                name.replace("_", " ").replace(".", " "),
                str(tool.get("description", "")),
                str(app.get("name_vi", "")),
                str(app.get("description", "")),
            ]
            properties = (tool.get("parameters") or {}).get("properties") or {}
            for param_name, info in properties.items():
                parts.append(param_name.replace("_", " "))
                if isinstance(info, dict):
                    parts.append(str(info.get("description", "")))
                    parts.extend(str(value) for value in info.get("enum", []) or [])
            docs.setdefault(app_id, set()).update(self._routing_tokens(" ".join(parts)))

        doc_frequency = Counter(token for tokens in docs.values() for token in tokens)
        app_count = max(len(docs), 1)
        scores = {
            app_id: sum(
                math.log((app_count + 1) / (doc_frequency[token] + 1)) + 1.0
                for token in query_tokens & tokens
            )
            for app_id, tokens in docs.items()
        }
        return {app_id for app_id, score in scores.items() if score >= 2.0}

    @staticmethod
    def _routing_tokens(text: str) -> set:
        normalized = unicodedata.normalize("NFC", str(text).lower())
        stopwords = {
            "anh", "chị", "em", "tôi", "mình", "giúp", "cho", "với", "nhé", "nha",
            "muốn", "cần", "được", "cái", "một", "những", "này", "đó", "thì", "lại",
            "xem", "kiểm", "tra", "thông", "tin", "danh", "sách", "hiện", "tại",
        }
        tokens = {
            token for token in re.findall(r"[^\W\d_]+", normalized, flags=re.UNICODE)
            if len(token) >= 2 and token not in stopwords
        }
        # Small domain-neutral vocabulary bridges common spoken forms to schema
        # terms. Keep this at the entity level (not whole workflows) so unseen
        # combinations still route compositionally.
        aliases = {
            "đô": {"usd", "exchange", "rate", "currency"},
            "đôla": {"usd", "exchange", "rate", "currency"},
        }
        return tokens | {
            alias
            for token in tokens
            for alias in aliases.get(token, set())
        }
