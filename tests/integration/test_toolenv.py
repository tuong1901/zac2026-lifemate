from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
from lifemate.tools.toolenv_adapter import OfficialToolEnvAdapter, load_catalog


class OfficialToolEnvIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value=os.environ.get("ORIGINAL_DATA_DIR")
        if not value:
            cls.skip_reason="set ORIGINAL_DATA_DIR to the official package root"
            return
        cls.root=Path(value)
        if not (cls.root/"train/train.jsonl").is_file():
            cls.skip_reason="official train JSONL is absent"

    def setUp(self):
        if getattr(self,"skip_reason",None): self.skipTest(self.skip_reason)

    def test_one_official_sample_uses_catalog_and_submits_hash(self):
        import json
        with (self.root/"train/train.jsonl").open(encoding="utf-8") as f: sample=json.loads(f.readline())
        _, catalog=load_catalog(self.root,"train")
        allowed={t for app in sample["context"]["installed_apps"].values() for t in app.get("tools",[])}
        env=OfficialToolEnvAdapter(self.root,"train",sample["id"])
        tool=next(name for name in allowed if name in catalog and catalog[name].get("kind")=="read" and not catalog[name].get("parameters",{}).get("required"))
        schema=env.validate(tool,{})
        self.assertTrue(schema["ok"],schema)
        observation=env.call(tool,{})
        self.assertIsInstance(observation,dict)
        self.assertEqual(len(env.get_tool_calls()),1)
        self.assertRegex(env.get_state_hash(),r"^[0-9a-f]{64}$")


if __name__ == "__main__": unittest.main()
