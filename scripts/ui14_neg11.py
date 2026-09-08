#!/usr/bin/env python3
"""Isolated neg11 coordinator. All parent paths are read-only inputs."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from collections import Counter, defaultdict
from ui14_common import *
from ui14_neg11_data import (NEG_DATA, NEG_OUTPUT, OLD_OUTPUT, assert_isolated, seed_evidence,
                            inventory, normalize, validate_extension)
from ui14_progress import ProgressSession
from ui14_verification import preparation_lock, verification_session

STAGES=("inventory","audit-raw","normalize","cache-prepare","cache","cache-finalize","finalize","check","submit","status","eval-existing","audit-errors")


def audit_errors(prediction, destination):
    """Read-only diagnosis. Preserve original parser status and scoring denominator."""
    result={}; examples=defaultdict(list)
    for task in UI_TASKS:
        folder=Path(prediction)/task.task_key; counts=Counter(); seen=set()
        for path in sorted((folder/"gate").glob("*.json")):
            gate=read_json(path); image=gate.get("source_image_id") or gate.get("image_path") or path.name
            if image in seen: raise ValueError("Multiple gate records for the same task-image")
            seen.add(image); counts["task_images"]+=1
            if gate.get("prediction_status")!="parse_error": continue
            counts["parse_error"]+=1
            raw_path=folder/"raw"/path.name
            if raw_path.is_file():
                raw=read_json(raw_path); answer=raw.get("raw_answer",raw.get("answer",""))
                tiles=raw.get("inference_crop",{}).get("tiles",[])
                failed=[str(t.get("answer","")) for t in tiles if t.get("status")=="parse_error"]
                texts=failed or [str(answer)]
                kind="empty_output" if all(not s.strip() for s in texts) else "bbox_parse" if any("<box" in s.lower() for s in texts) else "illegal_format"
            else: kind="missing_raw_evidence"; texts=[]
            counts[kind]+=1
            if len(examples[kind])<5: examples[kind].append({"task":task.task_key,"image":image,"gate":str(path),"raw":str(raw_path),"excerpt":"\n".join(texts)[:1500]})
        for path in (folder/"errors").glob("*.json"):
            error=read_json(path)
            image=error.get("source_image_id") or error.get("image_path")
            if image not in seen: counts["runtime_failure"]+=1
        result[task.task_key]=dict(counts)
    value={"prediction_root":str(prediction),"tasks":result,"examples":dict(examples),
           "totals":dict(sum((Counter(v) for v in result.values()),Counter())),
           "policy":"diagnosis only; parse_error remains illegal; no predictions or denominators modified"}
    write_json(destination,value)
    return value


def summarize_parent(args):
    """Only completed old step-1000 predictions; never its live workbook/state."""
    from uuid import uuid4
    from run_ui5_eval import build_score_command
    from run_ui14_eval import score_ui9
    from collect_ui5_metrics import parse_markdown_report, collect_gate_metrics
    parent=Path(args.parent_root);old=Path(args.old_output)
    prediction=old/"inference-checkpoint-1000-ui14"
    if not prediction.is_dir(): return {"status":"unavailable","prediction":str(prediction)}
    report=Path(args.data_root)/"parent_prediction_audit"
    audit_errors(prediction,report/"parse_errors.json")
    state=old/"evaluation/ui14-step-1000.json"
    if not state.is_file() or read_json(state).get("status")!="success":
        return {"status":"parse_audit_only","reason":"parent evaluation is not marked complete"}
    specs=read_json(parent/"evaluation_manifest.json")["tasks"]
    run_name="step-1000-"+uuid4().hex; destination=report/run_name
    options=SimpleNamespace(scorer_root=PROJECT_ROOT,input_dir=Path(specs[0]["test"]).parent)
    subprocess.run(build_score_command(options,prediction_dir=prediction,raw_evaluation_root=report,run_name=run_name),check=True)
    metrics=parse_markdown_report(destination/"all_tasks_evaluation.txt")
    gates=collect_gate_metrics(prediction,options.input_dir,PROJECT_ROOT)
    for spec in specs[5:]: metrics["tasks"][spec["task_key"]],_=score_ui9(spec,prediction,destination)
    gates.update(collect_gate_metrics(prediction,None,task_files={s["task_key"]:Path(s["test"]) for s in specs[5:]}))
    write_json(destination/"metrics.json",{"eval_set_id":file_digest(parent/"evaluation_manifest.json"),"dataset":"parent only",**metrics})
    write_json(destination/"gate_diagnostics.json",gates)
    write_json(report/"latest.json",{"status":"complete","directory":str(destination),"parent_predictions":str(prediction)})
    return {"status":"complete","directory":str(destination)}


def eval_existing(args):
    root=Path(args.data_root); manifest=root/"evaluation_manifest.json"
    if not args.checkpoint: raise ValueError("eval-existing requires --checkpoint PATH")
    checkpoint=Path(args.checkpoint).resolve(strict=True)
    from ui14_profile import validate_prepared_profile
    validate_prepared_profile({"UI14_DATA_ROOT":str(root),"INIT_CHECKPOINT":args.init_checkpoint})
    model_files=sorted(set(checkpoint.glob("*.safetensors"))|set(checkpoint.glob("pytorch_model*.bin")))
    if not model_files: raise ValueError("Checkpoint has no model weights")
    # Actual weight bytes, not a logical architecture signature, identify this
    # comparison. No old predictions are imported on a weak/path-only match.
    weights={p.name:file_digest(p) for p in model_files}
    weights["config.json"]=file_digest(checkpoint/"config.json")
    for name in ("processor_config.json","preprocessor_config.json","tokenizer_config.json","tokenizer.json",
                 "special_tokens_map.json","added_tokens.json","vocab.json","merges.txt",
                 "chat_template.json","chat_template.jinja"):
        if (checkpoint/name).is_file(): weights[name]=file_digest(checkpoint/name)
    model_id=digest(weights); eval_id=read_json(manifest)["eval_set_id"]
    output=root/"comparisons"/model_id/eval_id
    config_bytes=(checkpoint/"config.json").read_bytes()
    state=read_json(checkpoint/"trainer_state.json") if (checkpoint/"trainer_state.json").exists() else {}
    step=int(state.get("global_step",checkpoint.name.rsplit("-",1)[-1]))
    write_json(output/"comparison_binding.json",{"checkpoint":str(checkpoint),"weight_files":weights,
        "weight_id":model_id,"eval_set_id":eval_id,"prediction_import":"none; infer unless this exact comparison already has results"})
    from run_ui14_eval import run
    os.environ["UI_EVAL_MANIFEST"]=str(manifest)
    options=SimpleNamespace(output_dir=output,checkpoint=checkpoint,skip_patch=True,external_eval_set=True,
        base_model=Path(args.init_checkpoint),step=step,project_root=PROJECT_ROOT,eval_gpu_devices="0,1,2,3",
        eval_inference_workers_per_gpu=2,attn_implementation="sdpa",scorer_root=PROJECT_ROOT,
        input_dir=root,recipe_path=root/"training_recipe.json",tile_nms_iou=.5)
    try: return run(options)
    finally:
        if (checkpoint/"config.json").read_bytes()!=config_bytes: raise RuntimeError("Read-only checkpoint configuration changed")


def status(args):
    root=Path(args.data_root); value={"data_root":str(root),"output_dir":args.output_dir,"stages":{}}
    for stage in STAGES:
        path=root/"stage_summaries"/f"{stage}.json"
        value["stages"][stage]=read_json(path) if path.is_file() else {"status":"not_run"}
    for name in ("progress.json","inventory_summary.json","cache_reuse_summary.json","negative_image_counts.json","cpu_check_report.json"):
        path=root/name
        if path.is_file(): value[name]=read_json(path)
    print(json.dumps(value,ensure_ascii=False,indent=2))
    return value


def run(args):
    root,parent=Path(args.data_root),Path(args.parent_root)
    assert_isolated(root,parent,args.source_root,args.old_output,args.output_dir)
    if args.stage=="status": return status(args)
    if not parent.is_dir() or not Path(args.source_root).is_dir():
        raise FileNotFoundError("Read-only A800 inputs are unavailable here: "+str(parent)+"; "+str(args.source_root))
    root.mkdir(parents=True,exist_ok=True)
    with preparation_lock(root/"coordination"):
        started=time.time()
        write_json(root/"stage_summaries"/f"{args.stage}.json",{"status":"running","stage":args.stage,
            "started_at_unix":started,"pid":os.getpid(),"output_path":str(root)})
        seed_evidence(root,parent)
        # Detection reads handoff manifests only; no image evidence journal or
        # CPU image inspection is opened on the allocated GPU path.
        session=nullcontext() if args.stage=="cache" else verification_session(root,full=args.full_verify)
        with ProgressSession("neg11-"+args.stage,root,args.progress_interval_seconds) as progress,session:
            try:
                if args.stage=="inventory":
                    result=inventory(args)
                    pred=Path(args.old_output)/"inference-checkpoint-1000-ui14"
                    if pred.is_dir(): audit_errors(pred,root/"parent_parse_error_audit.json")
                elif args.stage=="normalize": result=normalize(args)
                elif args.stage=="audit-raw":
                    from ui14_neg11_raw_audit import audit_raw
                    result=audit_raw(args)
                elif args.stage in ("cache-prepare","cache","cache-finalize"):
                    from ui14_neg11_cache import import_parent_cache, cache_status
                    if args.stage=="cache-prepare": import_parent_cache(root,parent)
                    stages={"cache-prepare":"prepare","cache":"detect","cache-finalize":"crops"}
                    command=[sys.executable,str(PROJECT_ROOT/"scripts/prepare_ui14_detector_crops.py"),
                        "--stage",stages[args.stage],"--data-root",str(root),"--gpus","0,1,2,3",
                        "--prepare-workers",str(args.workers),"--ui5-cache",args.ui5_cache]
                    progress.delegated=True
                    try: subprocess.run(command,check=True)
                    finally: progress.delegated=False
                    if args.stage=="cache-prepare":
                        from ui14_neg11_cache import seed_cross_task_detections
                        seed_cross_task_detections(root,args.ui5_cache)
                    result=cache_status(root)
                elif args.stage in ("finalize","check"):
                    import ui14_neg11_finalize as final
                    result=getattr(final,args.stage)(args)
                elif args.stage=="submit":
                    command=[sys.executable,str(PROJECT_ROOT/"scripts/submit_locany_ui5.py"),
                        "--profile","m32-cpt9000-ui14-neg11-v1","--machine","a800","--resource-group",args.resource_group,
                        "--gpus","4","--ui14-data-root",str(root)]
                    if args.render_only: command.append("--render-only")
                    subprocess.run(command,check=True); result={"mlx_command":command,"render_only":args.render_only}
                elif args.stage=="eval-existing": result=eval_existing(args)
                elif args.stage=="audit-errors":
                    result=summarize_parent(args)
                write_json(root/"stage_summaries"/f"{args.stage}.json",{"status":"complete","stage":args.stage,
                    "elapsed_seconds":time.time()-started,"result":result,"output_path":str(root)})
                return result
            except BaseException as exc:
                write_json(root/"stage_summaries"/f"{args.stage}.json",{"status":"failed","stage":args.stage,
                    "elapsed_seconds":time.time()-started,"error":str(exc),"output_path":str(root)})
                raise


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=STAGES)
    p.add_argument("--data-root",default=os.environ.get("UI14_DATA_ROOT",NEG_DATA))
    p.add_argument("--parent-root",default=os.environ.get("UI14_PARENT_DATA_ROOT",DATA_ROOT))
    p.add_argument("--source-root",default=os.environ.get("UI9_DATA_ROOT",UI9_DATA_ROOT))
    p.add_argument("--output-dir",default=NEG_OUTPUT)
    p.add_argument("--old-output",default=OLD_OUTPUT)
    p.add_argument("--init-checkpoint",default=INIT_CHECKPOINT)
    p.add_argument("--ui5-cache",default=WORKSPACE+"/code/Eagle_LocateUI5_v4/Embodied-ui5-det-crop/work_dirs/ui5_eval_detector_cache_horizontal_v5")
    p.add_argument("--workers",type=int,default=int(os.environ.get("UI14_PREPARE_WORKERS","16")))
    p.add_argument("--checkpoint")
    p.add_argument("--resource-group",default=os.environ.get("UI14_RESOURCE_GROUP","aiai_locate"))
    p.add_argument("--render-only",action="store_true")
    p.add_argument("--full-verify",action="store_true")
    p.add_argument("--progress-interval-seconds",type=float,default=float(os.environ.get("UI14_PROGRESS_INTERVAL_SECONDS","10")))
    return p.parse_args(argv)


if __name__=="__main__": run(parse_args())
