from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

from lifemate.agent.controller import ToolController
from lifemate.agent.loop import run_episode
from lifemate.agent.parser import AgentOutputError, parse_agent_output
from lifemate.agent.state import WorkflowState
from lifemate.config import _expand, discover_original_data
from lifemate.data.formatter import encode_assistant_only
from lifemate.data.loader import iter_jsonl, write_jsonl
from lifemate.data.parser import parse_target
from lifemate.data.splitter import load_group_split
from lifemate.data.trajectory import group_trajectories
from lifemate.deployment.submission import validate_submission
from lifemate.models.loader import inspect_local_model
from lifemate.models.policy import PolicyContextOverflow
from lifemate.training.trainer import check_fsdp_hardware


class FakeEnv:
    def __init__(self): self.calls=[]
    def validate(self, name, args): return {"ok": name in {"sys.ask_missing_info", "notes.create"} and isinstance(args, dict), "error":"invalid"}
    def call(self, name, args):
        self.calls.append({"name":name,"arguments":args})
        return {"ok":True,"result":"observed"}
    def get_tool_calls(self): return list(self.calls)
    def get_state_hash(self): return "a"*64


class CoreTests(unittest.TestCase):
    def test_strict_target_json(self):
        target=parse_target('{"thought":"","actions":[],"is_finished":true}')
        self.assertTrue(target["is_finished"])
        with self.assertRaises(ValueError): parse_target('```json\n{}\n```')

    def test_target_rejects_extra_or_malformed_action_fields(self):
        with self.assertRaises(ValueError): parse_target('{"thought":"","actions":[{"name":"x"}],"is_finished":false}')
        with self.assertRaises(ValueError): parse_target('{"thought":"","actions":[],"is_finished":true,"extra":1}')

    def test_jsonl_utf8_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"data.jsonl"
            write_jsonl(path,[{"text":"Đà Nẵng"}])
            self.assertEqual(list(iter_jsonl(path)),[{"text":"Đà Nẵng"}])

    def test_trajectory_order_and_continuity(self):
        rows=[{"source_id":"s","id":"s#step1"},{"source_id":"s","id":"s#step0"}]
        grouped=group_trajectories(rows)
        self.assertEqual([r["id"] for r in grouped["s"]],["s#step0","s#step1"])
        with self.assertRaises(ValueError): group_trajectories([{"source_id":"s","id":"s#step0"},{"source_id":"s","id":"s#step2"}])

    def test_split_manifest_exact_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"split.json"; p.write_text(json.dumps({"a":"train","b":"validation"}),encoding="utf-8")
            self.assertEqual(load_group_split(p,{"a","b"})["a"],"train")
            with self.assertRaises(ValueError): load_group_split(p,{"a"})

    def test_assistant_only_loss_mask(self):
        class Tokenizer:
            chat_template="dummy"
            def apply_chat_template(self,messages,tokenize,add_generation_prompt):
                content="".join(f"<{m['role']}>{m['content']}" for m in messages)
                if add_generation_prompt: content += "<assistant>"
                if messages[-1]["role"] == "assistant": content += "<eot>"
                return content
            def __call__(self,text,add_special_tokens): return {"input_ids":[ord(c) for c in text]}
        messages=[{"role":"user","content":"U"},{"role":"assistant","content":"T"}]
        encoded=encode_assistant_only(messages,Tokenizer())
        prompt_len=len("<user>U<assistant>")
        self.assertEqual(encoded["labels"][:prompt_len],[-100]*prompt_len)
        self.assertTrue(all(x!=-100 for x in encoded["labels"][prompt_len:]))

    def test_qwen_output_schema_and_one_action_per_observation(self):
        good=parse_agent_output('{"thought":"","actions":[{"name":"notes.create","arguments":{"text":"x"}}],"is_finished":false}')
        self.assertEqual(len(good["actions"]),1)
        with self.assertRaises(AgentOutputError): parse_agent_output('{"thought":"","actions":[],"is_finished":false}')
        with self.assertRaises(AgentOutputError): parse_agent_output('{"thought":"","actions":[{"name":"a","arguments":{}},{"name":"b","arguments":{}}],"is_finished":false}')

    def test_controller_prevents_duplicate_successful_write(self):
        env=FakeEnv(); catalog={"notes.create":{"kind":"write","parameters":{}}}
        ctl=ToolController(env,{"notes.create"},catalog,WorkflowState("task"))
        output=ctl.parse('{"thought":"","actions":[{"name":"notes.create","arguments":{"text":"x"}}],"is_finished":false}')
        ctl.execute(output)
        with self.assertRaises(ValueError): ctl.execute(output)
        self.assertEqual(len(env.calls),1)

    def test_failed_action_does_not_mark_completed(self):
        state=WorkflowState("task"); action={"name":"notes.create","arguments":{"text":"x"}}
        state.record_result(action,{"ok":True,"done":False})
        self.assertEqual(state.completed,[])

    def test_loop_observes_tool_result_and_terminates(self):
        env=FakeEnv(); ctl=ToolController(env,{"notes.create"},{"notes.create":{"kind":"write"}})
        outputs=iter([
            '{"thought":"create","actions":[{"name":"notes.create","arguments":{"text":"x"}}],"is_finished":false}',
            '{"thought":"done","actions":[],"is_finished":true}'])
        seen=[]
        def policy(messages):
            seen.append(list(messages))
            return next(outputs)
        result=run_episode(sample_id="s",initial_messages=[{"role":"user","content":"do it"}],controller=ctl,policy=policy,max_steps=3)
        self.assertEqual(result["status"],"finished")
        self.assertIn("TOOL OBSERVATION",seen[1][-1]["content"])

    def test_loop_stops_on_context_overflow(self):
        ctl=ToolController(FakeEnv(),set(),{})
        result=run_episode(sample_id="s",initial_messages=[],controller=ctl,
                           policy=lambda _: (_ for _ in ()).throw(PolicyContextOverflow("too long")),max_steps=10)
        self.assertEqual(result["status"],"context_overflow")
        self.assertEqual(result["steps"],1)

    def test_multiturn_continuation_reuses_controller(self):
        env=FakeEnv(); ctl=ToolController(env,{"sys.ask_missing_info"},{"sys.ask_missing_info":{"kind":"meta"}})
        ask='{"thought":"need info","actions":[{"name":"sys.ask_missing_info","arguments":{"tool_name":"notes.create","params":["text"]}}],"is_finished":false}'
        first=run_episode(sample_id="s",initial_messages=[{"role":"user","content":"create"}],controller=ctl,policy=lambda _:ask)
        self.assertEqual(first["status"],"waiting_for_user")
        ctl.allowed.add("notes.create"); ctl.catalog["notes.create"]={"kind":"write"}
        answer='{"thought":"continue","actions":[{"name":"notes.create","arguments":{"text":"x"}}],"is_finished":false}'
        resumed=first["transcript"]+[{"role":"user","content":"text is x"}]
        second=run_episode(sample_id="s",initial_messages=resumed,controller=ctl,policy=lambda _:answer)
        self.assertEqual(second["tool_calls"],env.calls)
        self.assertFalse(ctl.state.waiting_for_user)

    def test_pending_work_blocks_finish_before_execution(self):
        env=FakeEnv(); state=WorkflowState("task",pending=["send"])
        ctl=ToolController(env,set(),{},state)
        with self.assertRaises(ValueError): ctl.execute({"thought":"","actions":[],"is_finished":True})

    def test_submission_exact_schema_counts_ids_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"submission.jsonl"
            p.write_text(json.dumps({"id":"x","tool_calls":[],"state_hash":"b"*64})+"\n",encoding="utf-8")
            self.assertEqual(validate_submission(p,["x"])["records"],1)
            with self.assertRaises(ValueError): validate_submission(p,["x","y"])

    def test_local_model_identity_is_checked_without_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/"config.json").write_text(json.dumps({"model_type":"qwen3","_name_or_path":"Qwen/Qwen3-4B-Instruct-2507","_commit_hash":"a"*40,"max_position_embeddings":262144}),encoding="utf-8")
            (p/"model.safetensors").touch()
            result=inspect_local_model(p)
            self.assertEqual(result["model_id_verified"],"Qwen/Qwen3-4B-Instruct-2507")
            (p/"config.json").write_text(json.dumps({"model_type":"qwen3","_name_or_path":"Qwen/Qwen3-8B","_commit_hash":"a"*40}),encoding="utf-8")
            with self.assertRaises(ValueError): inspect_local_model(p)

    def test_fp32_training_requires_sharded_multigpu_before_model_download(self):
        with patch("lifemate.training.trainer.platform.system", return_value="Linux"):
            hw={"cuda_available":True,"gpu":[{"index":0,"name":"Tesla T4","total_vram_bytes":16*1024**3,"free_vram_bytes":15*1024**3}]}
            with self.assertRaisesRegex(RuntimeError,"at least two GPUs"):
                check_fsdp_hardware(hw)

    def test_two_t4_fsdp_hardware_passes_conservative_gate(self):
        with patch("lifemate.training.trainer.platform.system", return_value="Linux"):
            hw={"cuda_available":True,"gpu":[
                {"index":0,"name":"Tesla T4","total_vram_bytes":16*1024**3,"free_vram_bytes":14*1024**3},
                {"index":1,"name":"Tesla T4","total_vram_bytes":16*1024**3,"free_vram_bytes":14*1024**3}]}
            self.assertEqual(len(check_fsdp_hardware(hw)),2)

    def test_env_config_expansion_and_original_data_discovery(self):
        self.assertEqual(_expand("${ABSENT_VAR:abc}"),"abc")
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for rel in ("train/train.jsonl","public_test/public_test.jsonl","train/zaic_tools"):
                path=root/rel
                if path.suffix:
                    path.parent.mkdir(parents=True,exist_ok=True); path.write_text("",encoding="utf-8")
                else:
                    path.mkdir(parents=True,exist_ok=True)
            self.assertEqual(discover_original_data([root]),root.resolve())


if __name__ == "__main__": unittest.main()
