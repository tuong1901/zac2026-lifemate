import json
from typing import Dict, List, Any

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

        for app_id in installed_ids:
            if app_id in self.apps_by_id:
                app_info = self.apps_by_id[app_id]
                for tool in app_info.get("tools", []):
                    # Nếu installed_apps chỉ định danh sách tool cụ thể
                    if app_id in installed_apps and "tools" in installed_apps[app_id]:
                        specific_tools = installed_apps[app_id]["tools"]
                        if specific_tools and tool["name"] not in specific_tools:
                            continue
                    allowed.append(tool)
        return allowed

    def format_tools_for_prompt(self, allowed_tools: List[Dict[str, Any]]) -> str:
        """
        Định dạng danh sách tool thành chuỗi ngắn gọn cho LLM
        """
        lines = []
        for t in allowed_tools:
            name = t["name"]
            desc = t.get("description", "")
            params = t.get("parameters", {}).get("properties", {})
            required = t.get("parameters", {}).get("required", [])
            
            param_str = []
            for p_name, p_info in params.items():
                req_tag = "(required)" if p_name in required else "(optional)"
                param_str.append(f"{p_name}: {p_info.get('type', 'any')} {req_tag}")
                
            lines.append(f"- Tool: `{name}` | Mô tả: {desc} | Tham số: [{', '.join(param_str)}]")
        return "\n".join(lines)
