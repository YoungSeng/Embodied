"""CPU scoring and observability contracts, including external old checkpoints."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"));sys.path.insert(0,str(ROOT))
from ui14_common import *
from eaglevl.train.ui5_excel_logger import build_eval_rows


class Neg11EvaluationTests(unittest.TestCase):
    def test_periodic_composite_eval_opens_cached_verification_session(self):
        import run_ui14_eval as evaluation
        from ui14_verification import current_checks
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/"negative_extension_manifest.json",{})
            with mock.patch.dict(os.environ,{"UI_EVAL_MANIFEST":str(root/"evaluation_manifest.json")}),\
                 mock.patch.object(evaluation,"_run",side_effect=lambda args: current_checks() is not None):
                self.assertTrue(evaluation.run(None))
            self.assertIsNone(current_checks())

    def test_external_checkpoint_uses_new_set_without_mutating_config(self):
        from tests.test_ui14_pipeline import UI14EvaluationTests
        UI14EvaluationTests._exercise_full_evaluation(self,1000,real_scorer=True,eval_set_id="neg11-fixture",external=True)

    def test_counts_reconstruct_ui5_metrics_and_micro_unknown_is_not_zero(self):
        from qwen3vl_merge_and_score_fixed_5tasks import write_all_tasks_summary
        from collect_ui5_metrics import parse_markdown_report
        metric={"image":{"tp":3,"fp":2,"fn":1,"tn":4,"precision":.6,"recall":.75,"f1":2*.6*.75/1.35,"accuracy":.7},
                "bbox":{"tp":3,"fp":2,"fn":1,"precision":.6,"recall":.75,"f1":2*.6*.75/1.35,"count_accuracy":.5}}
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            path=Path(tmp)/"report.txt"
            write_all_tasks_summary(task_summaries={t.task_key:metric for t in UI5_TASKS},output_path=str(path))
            parsed=parse_markdown_report(path)
        for task in UI5_TASKS:
            values=parsed["tasks"][task.task_key]
            self.assertEqual(values["image"]["tn"],4)
            self.assertNotIn("tn",values["bbox"])
            self.assertAlmostEqual(values["image"]["f1"],2*3/(2*3+2+1))
        for task in UI9_TASKS:parsed["tasks"][task.task_key]={k:dict(v) for k,v in metric.items()}
        rows=build_eval_rows(step=0,checkpoint="checkpoint",metrics=parsed,metadata={"eval_set_id":"fixture"})
        self.assertEqual(len(rows),36)
        micro=next(r for r in rows if r["task"]=="five_task_micro" and r["granularity"]=="image")
        self.assertEqual((micro["tp"],micro["fp"],micro["fn"],micro["tn"]),(15,10,5,20))
        parsed["tasks"]["cropping"]["image"]["tp"]=None
        rows=build_eval_rows(step=0,checkpoint="checkpoint",metrics=parsed)
        micro=next(r for r in rows if r["task"]=="five_task_micro" and r["granularity"]=="image")
        self.assertIsNone(micro["tp"]);self.assertIsNone(micro["f1"])

    def test_tile_feature_states_and_parse_error_audit_preserve_predictions(self):
        from collect_ui5_metrics import collect_gate_metrics,consistent_feature
        from ui14_neg11 import audit_errors
        from locany_ui5_common import aggregate_tiled_gate_diagnostics
        self.assertIsNone(consistent_feature([True,None]));self.assertIsNone(consistent_feature([True,False]))
        for feature in ("pbd_enabled","coordinate_bridge_enabled","slot_routing_enabled"):
            self.assertIs(aggregate_tiled_gate_diagnostics([{feature:True},{feature:True}],crop_mode="scan")[feature],True)
            self.assertIsNone(aggregate_tiled_gate_diagnostics([{feature:True},{}],crop_mode="scan")[feature])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); task="synth_loneword"; test=root/"test.jsonl"
            rows=[]
            for index,(text,status) in enumerate((("", "parse_error"),("<box>broken</box>","parse_error"),("prose", "parse_error"),("<box>none</box>","ok"))):
                image=str(root/f"{index}.png");rows.append({"source_image":image,"boxes_px":[]})
                gate={"image_path":image,"prediction_status":status,"final_boxes_pixel_xyxy":[],"p_defect":.1}
                write_json(root/"pred"/task/"gate"/f"{index}.json",gate)
                write_json(root/"pred"/task/"raw"/f"{index}.json",{"raw_answer":text,
                    "inference_crop":{"tiles":[{"answer":text,"status":status,"gate":{
                        "pbd_enabled":True,"coordinate_bridge_enabled":True,"slot_routing_enabled":True}}]}})
            write_jsonl(test,rows)
            write_json(root/"pred"/task/"errors/error.json",{"image_path":"runtime.png","error":"CUDA OOM"})
            before={p:p.read_bytes() for p in (root/"pred").rglob("*.json")}
            gates=collect_gate_metrics(root/"pred",None,task_files={task:test})
            self.assertIs(gates[task]["pbd_enabled"],True)
            self.assertIs(gates[task]["coordinate_bridge_enabled"],True)
            self.assertIs(gates[task]["slot_routing_enabled"],True)
            audit=audit_errors(root/"pred",root/"audit.json")
            self.assertEqual(audit["totals"],{"task_images":4,"parse_error":3,"empty_output":1,"bbox_parse":1,"illegal_format":1,"runtime_failure":1})
            self.assertEqual(before,{p:p.read_bytes() for p in before})

    def test_formal_profile_uses_isolated_paths_and_global_ratio_two(self):
        from submit_locany_ui5 import parse_args,render_job
        from ui14_checks import validate_formal_yaml
        from ui14_neg11_data import NEG_DATA,NEG_OUTPUT,NEG_PROJECT
        args=parse_args(["--profile","m32-cpt9000-ui14-neg11-v1","--machine","a800","--gpus","4","--resource-group","aiai_locate","--render-only"])
        rendered,runtime=render_job(args)
        validate_formal_yaml(rendered,runtime,config_path=args.config)
        self.assertEqual(runtime["PROJECT_ROOT"],NEG_PROJECT)
        self.assertEqual(runtime["UI14_DATA_ROOT"],NEG_DATA)
        self.assertEqual(runtime["OUTPUT_DIR"],NEG_OUTPUT)
        self.assertEqual(runtime["UI_NEGATIVE_TO_POSITIVE_RATIO"],2.)
        self.assertEqual(runtime["MAX_STEPS"],16000)


if __name__=="__main__":unittest.main()
