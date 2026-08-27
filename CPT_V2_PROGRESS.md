# UI CPT v2 implementation checkpoint

## Latest archive — 2026-08-27

User requested a pause due to quota. The working tree remains intentionally
uncommitted; no Merlin job was submitted.

Latest verified state after resuming:

- 49/49 CPT unit tests pass, including split/leakage, observability, evaluator,
  checkpoint selection, overfitting, Excel failure isolation, smoke-output
  validation, resume counters, and Merlin static contracts.
- `bash -n` passes for `shell/run_locany_cpt.sh` and
  `shell/run_locany_cpt_merlin.sh`.
- 146 Python files compile through the built-in `compile()` check without
  imports or bytecode writes.
- All five synthetic example JSON/JSONL files parse successfully.
- `git diff --check` passes; its only output is the existing Windows
  LF-to-CRLF warning.
- Generated `__pycache__` directories have been removed from the workspace.
- The independent XLSX check is now isolated: the workbook has exactly
  `Overview`, `TrainMetrics`, and `EvalMetrics`, and all three data tables are
  present, but the exported example lacks frozen header rows and filters. The
  optional training-side writer still sets both properties and its test passes.
  Regenerate the standalone example with the approved spreadsheet artifact
  runtime before calling the XLSX deliverable final.

Additional completed work since the earlier archive:

- CPT disables processor truncation while SFT keeps historical truncation.
- Runtime oversize record/group hashes and static record/source/line JSONL are
  available; manifest contains matching compact hashes.
- Inference errors remain visible and count as primary=0 instead of dropping
  from held-out denominators; micro primary uses sample/box weights.
- Evaluator validates selected rows against the split manifest, locks and
  rescales Base cache metrics offline, and isolates checkpoint comparisons by
  manifest plus evaluation protocol.
- Smoke jobs now run checkpoint-10 -> automatic resume -> checkpoint-20 in one
  job, emit metrics at steps 5/10/15/20, and H20 smoke is allowed.
- `scripts/validate_locany_cpt_smoke.py` validates ten-task accounting,
  monotonic resume state, per-rank dataloader files, eval queue, workbook, and
  A800/H20 schema equality.
- Multi-image rows now use shared-image connected components, so records that
  overlap on any image cannot receive different group IDs.
- Best-checkpoint selection now rejects incomplete held-out and train-pool
  candidates inside the policy itself, not only at its caller.
- README, implementation report, and schema-only examples were added; the
  standalone workbook remains pending the artifact repair noted above.

Resume/continue with:

1. Regenerate the standalone XLSX with frozen row 1 and enabled filters once
   the workspace dependency loader is available, then inspect values/formulas,
   scan errors, render all three sheets, and re-run the independent check.
2. Review the final `git status`/diff after that artifact update.
3. With user authorization/resources, submit A800 and H20x2 smoke jobs, run
   `validate_locany_cpt_smoke.py`, then perform held-out eval. Do not claim GPU
   or held-out results before those jobs finish.

Saved: 2026-08-27 (Asia/Shanghai)

Scope: CPT only. SFT default behavior must remain unchanged; runtime additions are gated by `LOCANY_CPT_MODE=1`.

## Completed in the current working tree

- Fixed the DeepSpeed/Accelerate `DummyOptim.param_groups` compatibility failure.
- Disabled UI5-only optimizer-update assertions when CPT runs with UI relation disabled, fixing the step-20 `image_gate/relation/slot_gate` false failure.
- Added deterministic image-content-grouped 98/2 train/held-out splitting, manifest, split summary, train/val/val_fast recipes, label-aware minimum validation coverage, and leakage validation.
- Added exact CPT attempted/accepted/oversize/trained/packed accounting, main/MTP token metadata, task-token IDs, per-task fused-loss CE aggregation, distributed union/reduce, checkpoint state, resume sampling checks, coverage/repeat/effective-epoch fields, warnings, throughput, and peak-memory fields.
- Added fixed sampling modes: `sample_equal` (default), `sqrt_size`, `token_balanced`, and `hybrid`, plus an offline exposure simulator.
- Added static processor/post-MTP length analysis and recipe mean-token updates.
- Rebuilt the evaluator around deterministic subsets, explicit held-out/train-pool labels, Base caching, point/VQA/class-aware defect/one-to-one box/OCR/task-macro metrics, raw prediction artifacts, offline rescoring, teacher-forced main-token CE, and held-out checkpoint selection policy.
- Added optional JSON/JSONL-to-Excel export with exactly `Overview`, `TrainMetrics`, and `EvalMetrics`; Excel failure is warning-only.
- Added TorchElastic `@record` and 20-step A800/H20x2 smoke job definitions.

## Verification completed

- All 49 CPT tests pass after the latest evaluator/accounting and split-policy refinements.
- Bash syntax, Python compile, example JSON parsing, and `git diff --check` pass.
- The training-side Excel writer passes tests for exactly three sheets, frozen
  headers, enabled filters, and warning-only failure behavior.
- The standalone example XLSX still needs regeneration for frozen headers and
  filters as described in the latest archive above.
- No GPU smoke was run locally because this Windows workspace has no CUDA/PyTorch training runtime or NAS dataset.

## Resume here

1. Regenerate and verify the standalone XLSX artifact.
2. Review the fused per-token CE path and dataset checkpoint counters in the real CUDA environment.
3. Run 20-step A800 and H20x2 smoke jobs, including resume, schema comparison, distributed aggregation, original TorchElastic error capture, and step-time overhead measurement.
4. Run held-out eval smoke (10 examples x 10 tasks), then generate the final evidence-based report. Do not claim smoke/eval results until those jobs finish.

## Important commands for continuation

```bash
python -m unittest tests.test_cpt_split tests.test_cpt_observability tests.test_cpt_eval_metrics tests.test_cpt_checkpoint_selection tests.test_cpt_excel tests.test_locany_cpt -v

mlx job submitv2 --path locany_cpt_v4_a100x4_smoke_merlin.yaml
mlx job submitv2 --path locany_cpt_v4_h20x2_smoke_merlin.yaml
```

The working tree is intentionally left uncommitted so the next session can inspect and amend the implementation before creating a final commit.
