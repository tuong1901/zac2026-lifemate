"""
catalog_inspector.py — Pure JSON-Schema & Catalog Metadata Inspector (Zero Hardcode).
Extracts tool kinds, x-ref dependencies, prerequisite read tools, fuzzy-matched parameters,
and dynamic app metadata directly from any tools catalog JSON or list of tool specs.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Union

_READ_ACTION_FALLBACK_RE = re.compile(
    r"^(get_|list_|search|find|read|my_|order_status|order_history|expiring_soon|next_departures|headlines)"
)
_TOOL_REF_IN_DESC_RE = re.compile(r"\b([a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)\b")


def _extract_return_tokens(returns_spec: str) -> Set[str]:
    if not returns_spec:
        return set()
    raw_tokens = re.findall(r"[a-z0-9_]+", returns_spec.lower())
    expanded: Set[str] = set()
    for tok in raw_tokens:
        expanded.add(tok)
        if tok.endswith("ies") and len(tok) > 3:
            expanded.add(tok[:-3] + "y")
        elif tok.endswith("s") and len(tok) > 2:
            expanded.add(tok[:-1])
    return expanded


class CatalogInspector:
    """100% schema-driven inspector for tool catalogs (generalizes to unseen apps)."""

    def __init__(self, catalog_or_path: Union[str, Path, Dict[str, Any], List[Dict[str, Any]], None] = None):
        self.apps_by_id: Dict[str, Dict[str, Any]] = {}
        self.tools_by_name: Dict[str, Dict[str, Any]] = {}
        self._load(catalog_or_path)

    def _load(self, source: Any) -> None:
        if source is None:
            return
        data = source
        if isinstance(source, (str, Path)):
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)

        if isinstance(data, list):
            for tool in data:
                if isinstance(tool, dict) and "name" in tool:
                    self._register_tool(tool)
        elif isinstance(data, dict):
            if "apps" in data and isinstance(data["apps"], list):
                for app in data["apps"]:
                    if not isinstance(app, dict) or "id" not in app:
                        continue
                    app_id = app["id"]
                    self.apps_by_id[app_id] = app
                    for tool in app.get("tools", []):
                        if isinstance(tool, dict) and "name" in tool:
                            self._register_tool(tool, default_app=app_id)
            elif "tools" in data and isinstance(data["tools"], list):
                for tool in data["tools"]:
                    if isinstance(tool, dict) and "name" in tool:
                        self._register_tool(tool)
            else:
                for key, val in data.items():
                    if isinstance(val, dict) and ("parameters" in val or "kind" in val or "name" in val):
                        tool_copy = dict(val)
                        tool_copy.setdefault("name", key)
                        self._register_tool(tool_copy)

    def _register_tool(self, tool: Dict[str, Any], default_app: str = "") -> None:
        name = tool["name"]
        app_id = tool.get("app") or default_app or (name.split(".", 1)[0] if "." in name else "")
        if app_id and "app" not in tool:
            tool = dict(tool, app=app_id)
        self.tools_by_name[name] = tool
        if app_id and app_id not in self.apps_by_id:
            self.apps_by_id[app_id] = {"id": app_id, "name_vi": app_id, "description": "", "group": ""}

    def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        return self.tools_by_name.get(tool_name)

    def is_read_tool(self, tool_name: str) -> bool:
        tool = self.tools_by_name.get(tool_name)
        kind = tool.get("kind") if tool else None
        if kind:
            return kind == "read"
        action = tool_name.split(".", 1)[-1] if tool_name else ""
        return bool(_READ_ACTION_FALLBACK_RE.match(action))

    def is_meta_tool(self, tool_name: str) -> bool:
        tool = self.tools_by_name.get(tool_name)
        kind = tool.get("kind") if tool else None
        if kind:
            return kind == "meta"
        return bool(tool_name and tool_name.startswith("sys."))

    def is_write_tool(self, tool_name: str) -> bool:
        tool = self.tools_by_name.get(tool_name)
        kind = tool.get("kind") if tool else None
        if kind:
            return kind == "write"
        if not tool_name or "." not in tool_name:
            return False
        return not self.is_meta_tool(tool_name) and not self.is_read_tool(tool_name)

    def get_xref_params(self, tool_name: str) -> Dict[str, Dict[str, Any]]:
        tool = self.tools_by_name.get(tool_name)
        if not tool:
            return {}
        params_spec = tool.get("parameters") or {}
        properties = params_spec.get("properties") or {}
        required = set(params_spec.get("required") or [])

        result: Dict[str, Dict[str, Any]] = {}
        for param_name, prop in properties.items():
            if not isinstance(prop, dict):
                continue
            has_xref = "x-ref" in prop and bool(prop["x-ref"])
            is_req_id = param_name.endswith("_id") and (param_name in required)
            if not (has_xref or is_req_id):
                continue

            desc = prop.get("description") or ""
            raw_candidates = _TOOL_REF_IN_DESC_RE.findall(desc)
            if self.tools_by_name:
                valid_hints = [c for c in raw_candidates if c in self.tools_by_name]
                hints = valid_hints if valid_hints else [c for c in raw_candidates if c != prop.get("x-ref")]
            else:
                hints = [c for c in raw_candidates if c != prop.get("x-ref")]

            result[param_name] = {
                "xref": prop.get("x-ref", ""),
                "reader_hint": hints[0] if hints else None,
                "reader_hints": hints,
                "required": param_name in required,
            }
        return result

    def _prefer_root_readers(self, candidates: List[str], target_xref: str = "") -> List[str]:
        if len(candidates) <= 1:
            return candidates
        root_only = []
        for cand in candidates:
            cand_xrefs = self.get_xref_params(cand)
            req_xrefs = [m for m in cand_xrefs.values() if m.get("required")]
            if not req_xrefs:
                root_only.append(cand)
            elif target_xref and all(m.get("xref") != target_xref for m in req_xrefs):
                root_only.append(cand)
        return root_only if root_only else candidates

    def get_prerequisite_readers(
        self,
        tool_name: str,
        allowed_tool_names: Optional[Iterable[str]] = None,
    ) -> List[str]:
        if not self.is_write_tool(tool_name):
            return []
        xref_params = self.get_xref_params(tool_name)
        if not xref_params:
            return []

        allowed = set(self.tools_by_name.keys()) if allowed_tool_names is None else set(allowed_tool_names)
        allowed_reads = [t for t in sorted(allowed) if t != tool_name and self.is_read_tool(t)]
        if not allowed_reads:
            return []

        tool_obj = self.tools_by_name.get(tool_name) or {}
        tool_app = tool_obj.get("app") or (tool_name.split(".", 1)[0] if "." in tool_name else "")

        ordered_readers: List[str] = []
        for meta in xref_params.values():
            matched: List[str] = []
            hints = meta.get("reader_hints") or ([meta["reader_hint"]] if meta.get("reader_hint") else [])
            for hint in hints:
                if hint in allowed and hint != tool_name and self.is_read_tool(hint):
                    matched.append(hint)

            xref = meta.get("xref") or ""
            if not matched and xref:
                ref_app, ref_col = xref.split(".", 1) if "." in xref else ("", xref)
                col_norm = ref_col.lower().rstrip("s")
                if ref_app:
                    app_col_matches = [
                        r for r in allowed_reads
                        if ((self.tools_by_name.get(r) or {}).get("app") or r.split(".", 1)[0]) == ref_app
                        and (
                            ref_col.lower() in _extract_return_tokens((self.tools_by_name.get(r) or {}).get("returns", ""))
                            or col_norm in _extract_return_tokens((self.tools_by_name.get(r) or {}).get("returns", ""))
                        )
                    ]
                    matched = self._prefer_root_readers(app_col_matches, target_xref=xref)
                if not matched and ref_col:
                    any_col_matches = [
                        r for r in allowed_reads
                        if ref_col.lower() in _extract_return_tokens((self.tools_by_name.get(r) or {}).get("returns", ""))
                        or col_norm in _extract_return_tokens((self.tools_by_name.get(r) or {}).get("returns", ""))
                    ]
                    matched = self._prefer_root_readers(any_col_matches, target_xref=xref)
                if not matched and ref_app:
                    ref_app_reads = [
                        r for r in allowed_reads
                        if ((self.tools_by_name.get(r) or {}).get("app") or r.split(".", 1)[0]) == ref_app
                    ]
                    matched = self._prefer_root_readers(ref_app_reads, target_xref=xref)

            if not matched and tool_app:
                same_app_reads = [
                    r for r in allowed_reads
                    if ((self.tools_by_name.get(r) or {}).get("app") or r.split(".", 1)[0]) == tool_app
                ]
                matched = self._prefer_root_readers(same_app_reads, target_xref=xref)

            for r in matched:
                if r not in ordered_readers:
                    ordered_readers.append(r)

        return ordered_readers

    def get_fuzzy_params(self, tool_name: str) -> Set[str]:
        tool = self.tools_by_name.get(tool_name)
        if not tool:
            return set()
        properties = (tool.get("parameters") or {}).get("properties") or {}
        return {
            p_name
            for p_name, prop in properties.items()
            if isinstance(prop, dict) and prop.get("x-match") == "fuzzy"
        }

    def get_app_metadata(self) -> Dict[str, Dict[str, str]]:
        return {
            app_id: {
                "name_vi": str(app.get("name_vi", app_id)),
                "description": str(app.get("description", "")),
                "group": str(app.get("group", "")),
            }
            for app_id, app in self.apps_by_id.items()
        }
