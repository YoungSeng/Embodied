"""
Eagle3-VL MTP Finetuning with Stream (Online) Packing
Combines Multi-Token Prediction (MTP) with efficient stream packing

Key Features:
1. [MTP] Multi-token prediction with block-based attention pattern
2. [Stream Packing] Online sample packing for efficient GPU utilization
3. [Attention] Proper attention masking for both packing and MTP blocks
4. [Resume] Perfect stateful resume support

"""

import os
import os.path as osp
import copy
import hashlib
import logging
import math
import random
import sys
import warnings
from contextlib import nullcontext
import numpy as np
from typing import Any, Dict, List, Mapping, Optional, Tuple
from dataclasses import dataclass, field

import json
import shutil
import time
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load
import torch.distributed as dist
import transformers
import traceback
import socket
from collections import defaultdict
from pathlib import Path
from eaglevl.dist_utils import init_dist

import packaging.version as version
from eaglevl.model.moon_vit.modeling_vit import MoonVitPretrainedModel
from eaglevl.patch import (
    replace_liger_fused_ops,
    replace_train_dataloader,
    replace_train_sampler
)
from eaglevl.model.locany.modeling_locateanything import LocateAnythingForConditionalGeneration
from eaglevl.model.locany.configuration_locateanything import LocateAnythingConfig
from eaglevl.model.locany.ui_relation_setup import (
    configure_ui5_model_config,
    initialize_or_validate_ui_relation,
)
from eaglevl.utils.locany.processing_locateanything import LocateAnythingProcessor
from eaglevl.utils.locany.image_processing_locateanything import LocateAnythingImageProcessor
from eaglevl.sp_utils import set_pg_manager, get_pg_manager
from eaglevl.train.constants import (
    special_tokens_list, IMG_CONTEXT_TOKEN, TEXT_MASK_TOKEN,
    NULL_TOKEN, BOX_START_TOKEN, BOX_END_TOKEN,
    REF_START_TOKEN, REF_END_TOKEN, number_tokens_list
)
from eaglevl.train.arguments import ModelArguments, DataTrainingArguments
from eaglevl.train.trainer_monkey_patch import replace_create_optimizer_with_various_lr
from eaglevl.train.dataset_sampling import (
    resolve_dataset_sampling_weight,
    resolve_recipe_entry_paths,
)
from eaglevl.train.optimizer_utils import optimizer_parameters
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset, IterableDataset, DataLoader
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments,
                          set_seed, AutoProcessor)
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)
from transformers import TrainerCallback
from eaglevl.train.tools import (SaveCheckpointCallback, MemoryLoggerCallback, 
                                  MilestoneCheckpointCallback, get_last_checkpoint_guard, 
                                  load_config, process_multimodal_sample)
from eaglevl.train.augmentation import apply_resize_augmentation
from eaglevl.train.ui_defect_data import (
    build_balanced_ui_indices,
    build_task_balanced_all_records_indices,
    build_task_source_balanced_rotating_plan,
    extract_ui_defect_targets,
    identify_ui_defect_task,
    is_positive_ui_defect,
    materialize_task_source_balanced_rotating_indices,
)
from eaglevl.train.ui5_excel_logger import UI5ExcelLogger, TRAIN_TASKS
from eaglevl.train.ui5_checkpoint_utils import atomic_save_with_fsync, validate_checkpoint
from eaglevl.train.ui5_curriculum import (
    CURRICULUM_POOLS,
    CurriculumGroupCycle,
    DeferredSampleLocations,
    UI5CurriculumSchedule,
    canonical_curriculum_pool,
    curriculum_artifact_identity,
    curriculum_pool_draw_counts,
    prepare_worker_states_for_resume,
    should_export_model_at_training_end,
    should_write_training_done_marker,
    training_continuity_config,
)
from eaglevl.train.ui5_sampling_coverage import (
    is_monotonic_coverage,
    write_sampling_coverage_atomic,
)
from dotenv import load_dotenv
load_dotenv()
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


if version.parse(torch.__version__) >= version.parse("2.4.0"):
    torch.serialization.add_safe_globals(
        [np.core.multiarray._reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))])


# ============ Patch ============
replace_liger_fused_ops()

# Patch HF PreTrainedModel to accept "magi" as attn_implementation.
# HF validates attn_implementation in __init__ and only allows
# eager/sdpa/flash_attention_2/flash_attention_3. We keep "magi" as-is so
# the custom Qwen2 stack can dispatch to the MagiAttention path explicitly.
import transformers.modeling_utils as _hf_modeling_utils
_orig_check_and_adjust = _hf_modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation

def _patched_check_and_adjust(self, attn_implementation, is_init_check=False):
    if attn_implementation == "magi":
        return "magi"
    return _orig_check_and_adjust(self, attn_implementation, is_init_check=is_init_check)

_hf_modeling_utils.PreTrainedModel._check_and_adjust_attn_implementation = _patched_check_and_adjust
# ========================

# ============ for loading large images ============
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
# ==================================================

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def training_args_serialization_preflight(training_args):
    """Exercise the exact small ``torch.save`` that previously ended in SIGBUS.

    Only rank 0 writes because Hugging Face also writes ``training_args.bin``
    on the saving rank.  Failure is broadcast so the other ranks do not wait
    indefinitely for a process that already exited with a Python exception.
    """

    error = None
    if get_rank() == 0:
        diagnostics_dir = Path(training_args.output_dir) / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        target = diagnostics_dir / f".training_args_preflight.{os.getpid()}.bin"
        logger.info(
            "[Checkpoint preflight] training_args.bin save START path=%s", target
        )
        try:
            torch.save(training_args, target)
            with open(target, "rb+") as handle:
                os.fsync(handle.fileno())
            size = target.stat().st_size
            if size <= 0:
                raise RuntimeError("torch.save produced an empty file")
            restored = torch.load(target, map_location="cpu", weights_only=False)
            if type(restored) is not type(training_args):
                raise TypeError(
                    "training arguments round-trip changed type: "
                    f"saved={type(training_args)!r}, restored={type(restored)!r}"
                )
            logger.info(
                "[Checkpoint preflight] training_args.bin save DONE "
                "path=%s size_bytes=%s",
                target,
                size,
            )
        except BaseException as exc:
            error = (
                "TrainingArguments save/reload preflight failed on the checkpoint "
                f"filesystem: path={target}; {type(exc).__name__}: {exc}"
            )
            logger.exception("[Checkpoint preflight] FAILED: %s", error)
        finally:
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "[Checkpoint preflight] could not remove temporary file %s: %s",
                    target,
                    exc,
                )
    if dist.is_available() and dist.is_initialized():
        message = [error]
        dist.broadcast_object_list(message, src=0)
        error = message[0]
        dist.barrier()
    if error is not None:
        raise RuntimeError(error)


class LazyJsonlLoader:
    """Lazy loader for JSONL files with fast index-based access."""
    
    def __init__(self, paths: List[str]):
        if isinstance(paths, str):
            paths = [paths]
        self.paths = paths
        self.offsets = []
        self._file_handles = {}
        self._build_index()

    def _build_index(self):
        for file_idx, path in enumerate(self.paths):
            if not os.path.exists(path):
                logger.warning(f"File not found: {path}")
                continue
            with open(path, 'rb') as f:
                offset = 0
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        self.offsets.append((file_idx, offset))
                    offset = f.tell()
        logger.info(f"Indexed {len(self.offsets)} lines from {len(self.paths)} files.")

    def __len__(self):
        return len(self.offsets)

    def _get_file_handle(self, file_idx: int):
        import threading
        thread_id = threading.get_ident()
        key = (thread_id, file_idx)
        if key not in self._file_handles:
            self._file_handles[key] = open(self.paths[file_idx], 'r', encoding='utf-8')
        return self._file_handles[key]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError("Index out of range")
        file_idx, offset = self.offsets[idx]
        f = self._get_file_handle(file_idx)
        f.seek(offset)
        line = f.readline()
        return json.loads(line)

    def __del__(self):
        for f in self._file_handles.values():
            try:
                f.close()
            except:
                pass
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['_file_handles'] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file_handles = {}


class LazySupervisedDatasetMTP(Dataset):
    """Lazy-loading dataset with MTP (Multi-Token Prediction) support."""
    
    def __init__(self,
                 ds_name: str,
                 meta: dict,
                 processor, 
                 block_size: int = 6,
                 repeat_time: float = 1,
                 max_frames: int = 16,
                 target_fps: int = 2,
                 video_total_pixels: int = 32000 * 28 * 28 * 0.9,
                 balance_ui_defects: bool = False,
                 ui_records_per_class: int = 17604,
                 ui_negative_to_positive_ratio: float = 2.0,
                 ui_sampling_mode: str = "fixed_ratio",
                 curriculum_group_sampling: bool = False):
        super().__init__()
        self.ds_name = ds_name
        self.processor = processor
        self.max_length = self.processor.tokenizer.model_max_length
        self.repeat_time = repeat_time
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.video_total_pixels = video_total_pixels
        self.block_size = block_size
        self.data_augment = meta.get("data_augment", False)
        self.visual_prompt = bool(meta.get("visual_prompt", False))
        self.ui5_crop_recipe = bool(meta.get("ui5_crop_recipe", False))
        self.curriculum_group_sampling = bool(curriculum_group_sampling)
        self.curriculum_pool = (
            canonical_curriculum_pool(meta.get("curriculum_pool"))
            if self.curriculum_group_sampling
            else None
        )
        self.balance_ui_defects = bool(meta.get("balance_ui_defects", balance_ui_defects))
        # An explicitly exported runtime mode must override the recipe's
        # historical default.  This lets a new sampler reuse an already audited
        # immutable crop recipe without rewriting its digest-bound marker.
        runtime_sampling_mode = os.environ.get("UI5_UI_SAMPLING_MODE")
        self.ui_sampling_mode = str(
            runtime_sampling_mode
            or (
                ui_sampling_mode
                if ui_sampling_mode != "fixed_ratio"
                else meta.get("ui_sampling_mode", ui_sampling_mode)
            )
        )
        if self.ui_sampling_mode not in {
            "fixed_ratio",
            "task_balanced_all_records",
            "task_source_balanced_rotating",
        }:
            raise ValueError(
                f"[{self.ds_name}] unsupported ui_sampling_mode={self.ui_sampling_mode!r}"
            )

        ann_paths = meta["annotation"]
        if not isinstance(ann_paths, (list, tuple)):
            ann_paths = [ann_paths]
        self.root = meta.get("root", "")
        
        logger.info(f"[Dataset] {self.ds_name} Indexing JSONL files...")
        start_time = time.time()
        self.lazy_loader = LazyJsonlLoader(ann_paths)
        logger.info(f"[Dataset] {self.ds_name} Indexing done in {time.time() - start_time:.2f}s.")
        
        original_num_rows = len(self.lazy_loader)
        self._raw_recipe_rows = original_num_rows
        self._raw_manual_repair_indices = {
            index
            for index in range(original_num_rows)
            if self.lazy_loader[index].get("_ui5_crop_source") == "manual_gt_repair"
        }
        logger.info(
            f"[Dataset] {self.ds_name} Found {original_num_rows} samples. "
            f"visual_prompt={self.visual_prompt}"
        )
        self.active_indices = list(range(original_num_rows))
        self._balanced_logical_buckets = None
        self._all_records_task_buckets = None
        self._source_balanced_plan = None

        exclusion_path = str(meta.get("excluded_samples", ""))
        excluded_pairs = set()
        excluded_ids = []
        if exclusion_path:
            exclusion_file = Path(exclusion_path).expanduser().resolve()
            if not exclusion_file.is_file():
                raise FileNotFoundError(
                    f"[{self.ds_name}] excluded sample manifest is missing: {exclusion_file}"
                )
            with exclusion_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    exclusion = json.loads(line)
                    sample_id = str(exclusion["sample_id"])
                    task = str(exclusion["task"])
                    excluded_pairs.add((sample_id, task))
                    excluded_ids.append(sample_id)
            before_exclusion = len(self.active_indices)
            self.active_indices = [
                index
                for index in self.active_indices
                if (
                    str(self.lazy_loader[index].get("_ui5_sample_id", "")),
                    str(self.lazy_loader[index].get("_ui5_task", "")),
                )
                not in excluded_pairs
            ]
            logger.info(
                "[Dataset] %s exclusion manifest=%s before=%s after=%s "
                "excluded_sample_ids=%s",
                self.ds_name,
                exclusion_file,
                before_exclusion,
                len(self.active_indices),
                sorted(set(excluded_ids)),
            )
        self._excluded_record_count = original_num_rows - len(self.active_indices)
        missing_manual_after_exclusion = self._raw_manual_repair_indices - set(
            self.active_indices
        )
        if missing_manual_after_exclusion:
            raise RuntimeError(
                f"[{self.ds_name}] exclusion filtering removed manual_gt_repair records: "
                f"{sorted(missing_manual_after_exclusion)[:20]}"
            )
        self._legal_crop_record_count = sum(
            self.lazy_loader[index].get("_ui5_record_kind") == "crop"
            for index in self.active_indices
        )

        crop_mode_enabled = _env_flag("UI5_USE_DETECTION_CROPS", False)
        if crop_mode_enabled and not self.ui5_crop_recipe:
            raise RuntimeError(
                f"[{self.ds_name}] UI5_USE_DETECTION_CROPS=1 but selected dataset "
                "is not an audited crop recipe"
            )
        if self.ui5_crop_recipe:
            source_counts = defaultdict(int)
            record_kind_counts = defaultdict(int)
            examples = defaultdict(list)
            for index in self.active_indices:
                record = self.lazy_loader[index]
                kind = str(record.get("_ui5_record_kind", "unknown"))
                source = str(record.get("_ui5_crop_source") or "full_image")
                record_kind_counts[kind] += 1
                source_counts[source] += 1
                if kind == "crop" and len(examples[source]) < 3:
                    examples[source].append(str(record.get("image", "")))
            crop_count = int(record_kind_counts.get("crop", 0))
            if crop_count <= 0:
                raise RuntimeError(
                    f"[{self.ds_name}] audited crop recipe contains zero crop records"
                )
            logger.info(
                "[Dataset] %s AUDITED CROP RECIPE loaded: record_kinds=%s "
                "crop_sources=%s examples=%s",
                self.ds_name,
                dict(record_kind_counts),
                dict(source_counts),
                dict(examples),
            )

        if self.ui_sampling_mode == "task_source_balanced_rotating":
            candidate_indices = list(self.active_indices)
            index_records = [self.lazy_loader[index] for index in candidate_indices]
            plan = build_task_source_balanced_rotating_plan(
                index_records,
                negative_to_positive_ratio=ui_negative_to_positive_ratio,
            )
            first_epoch = materialize_task_source_balanced_rotating_indices(
                plan, seed=202603, epoch_index=0
            )
            required_manual = {
                index
                for index, record in enumerate(index_records)
                if record.get("_ui5_crop_source") == "manual_gt_repair"
            }
            planned_indices = {
                index
                for polarities in plan["buckets"].values()
                for source_groups in polarities.values()
                for values in source_groups.values()
                for index in values
            }
            missing_manual = required_manual - planned_indices
            if missing_manual:
                raise RuntimeError(
                    f"[{self.ds_name}] source-balanced sampler dropped manual repairs: "
                    f"{sorted(missing_manual)[:20]}"
                )
            if planned_indices != set(range(len(index_records))):
                missing = sorted(set(range(len(index_records))) - planned_indices)
                raise RuntimeError(
                    f"[{self.ds_name}] source-balanced active plan dropped legal records: "
                    f"count={len(missing)}, first={missing[:20]}"
                )
            if len(first_epoch) != int(plan["epoch_length"]):
                raise RuntimeError(
                    f"[{self.ds_name}] source-balanced first epoch is incomplete"
                )
            self._source_balanced_plan = plan
            self._active_pool_length = len(self.active_indices)
            self._length = int(plan["epoch_length"])
            logger.info(
                "[Dataset] %s task_source_balanced_rotating: raw_recipe=%s "
                "active_pool=%s effective_epoch=%s positive_slots_per_task=%s "
                "negative_slots_per_task=%s effective_negative:positive=%.6f "
                "source_groups=%s records=%s never_active=%s "
                "manual_repair_retention=%s/%s",
                self.ds_name,
                original_num_rows,
                self._active_pool_length,
                self._length,
                plan["positive_slots_per_task"],
                plan["negative_slots_per_task"],
                plan["negative_to_positive_ratio"],
                plan["source_groups_by_task"],
                plan["records_by_task"],
                max(
                    0,
                    original_num_rows
                    - self._excluded_record_count
                    - self._active_pool_length,
                ),
                len(required_manual),
                len(required_manual),
            )
        elif self.ui_sampling_mode == "task_balanced_all_records":
            candidate_indices = list(self.active_indices)
            index_records = [self.lazy_loader[index] for index in candidate_indices]
            # Validation and the first deterministic epoch are intentionally
            # performed before training starts so an incomplete recipe fails closed.
            first_epoch = build_task_balanced_all_records_indices(index_records)
            buckets = {defect_type: [] for defect_type in range(5)}
            for logical_index, record in enumerate(index_records):
                task = identify_ui_defect_task(record)
                if task is None:
                    raise RuntimeError(
                        f"[{self.ds_name}] task_balanced_all_records found a non-UI record"
                    )
                buckets[task[1]].append(logical_index)
            required_manual = {
                index
                for index, record in enumerate(index_records)
                if record.get("_ui5_crop_source") == "manual_gt_repair"
            }
            missing_manual = required_manual - set(first_epoch)
            if missing_manual:
                raise RuntimeError(
                    f"[{self.ds_name}] all-record sampler dropped manual repairs: "
                    f"{sorted(missing_manual)[:20]}"
                )
            self._all_records_task_buckets = buckets
            self._active_pool_length = len(self.active_indices)
            self._length = max(len(values) for values in buckets.values()) * len(buckets)
            logger.info(
                "[Dataset] %s task_balanced_all_records: raw_recipe=%s "
                "active_pool=%s effective_epoch=%s task_records=%s "
                "never_active=%s manual_repair_retention=%s/%s",
                self.ds_name,
                original_num_rows,
                self._active_pool_length,
                self._length,
                {defect_type: len(values) for defect_type, values in buckets.items()},
                max(
                    0,
                    original_num_rows
                    - self._excluded_record_count
                    - self._active_pool_length,
                ),
                len(required_manual),
                len(required_manual),
            )
        elif self.balance_ui_defects:
            logger.info(
                f"[Dataset] {self.ds_name} building balanced UI index: "
                f"records_per_class={ui_records_per_class}, "
                f"negative:positive={ui_negative_to_positive_ratio}:1"
            )
            candidate_indices = list(self.active_indices)
            index_records = [self.lazy_loader[index] for index in candidate_indices]
            selected_local_indices = build_balanced_ui_indices(
                index_records,
                records_per_class=ui_records_per_class,
                negative_to_positive_ratio=ui_negative_to_positive_ratio,
            )
            required_manual_local_indices = {
                index
                for index, record in enumerate(index_records)
                if record.get("_ui5_crop_source") == "manual_gt_repair"
            }
            required_manual_gt_keys = set()
            for index in required_manual_local_indices:
                record = index_records[index]
                repair_gt_indices = record.get("_ui5_manual_repair_gt_indices")
                if not isinstance(repair_gt_indices, list) or not repair_gt_indices:
                    raise RuntimeError(
                        f"[Dataset] {self.ds_name} manual_gt_repair record lacks "
                        f"repair GT mapping: sample_id={record.get('_ui5_sample_id')}"
                    )
                required_manual_gt_keys.update(
                    (str(record.get("_ui5_sample_id")), int(gt_index))
                    for gt_index in repair_gt_indices
                )
            missing_manual_local_indices = required_manual_local_indices - set(
                selected_local_indices
            )
            if missing_manual_local_indices:
                missing_records = [
                    {
                        "sample_id": index_records[index].get("_ui5_sample_id"),
                        "task": index_records[index].get("_ui5_task"),
                        "repair_gt_indices": index_records[index].get(
                            "_ui5_manual_repair_gt_indices", []
                        ),
                    }
                    for index in sorted(missing_manual_local_indices)[:20]
                ]
                raise RuntimeError(
                    f"[Dataset] {self.ds_name} balanced UI index dropped required "
                    f"manual_gt_repair records: {missing_records}"
                )
            selected_manual_gt_keys = {
                (str(index_records[index].get("_ui5_sample_id")), int(gt_index))
                for index in selected_local_indices
                if index_records[index].get("_ui5_crop_source") == "manual_gt_repair"
                for gt_index in index_records[index]["_ui5_manual_repair_gt_indices"]
            }
            missing_manual_gt_keys = required_manual_gt_keys - selected_manual_gt_keys
            if missing_manual_gt_keys:
                raise RuntimeError(
                    f"[Dataset] {self.ds_name} balanced UI index dropped required repair GT "
                    f"mappings: {sorted(missing_manual_gt_keys)[:20]}"
                )
            self.active_indices = [
                candidate_indices[index] for index in selected_local_indices
            ]
            logger.info(
                f"[Dataset] {self.ds_name} balanced to {len(self.active_indices)} records "
                f"({len(self.active_indices) // 5} per class)."
            )
            logger.info(
                "[Dataset] %s manual_gt_repair retention after balancing: "
                "records=%s/%s repair_gt_keys=%s/%s",
                self.ds_name,
                len(required_manual_local_indices),
                len(required_manual_local_indices),
                len(selected_manual_gt_keys),
                len(required_manual_gt_keys),
            )
            self._balanced_logical_buckets = {
                defect_type: {"positive": [], "negative": []}
                for defect_type in range(5)
            }
            for logical_index, raw_index in enumerate(self.active_indices):
                # ``raw_index`` addresses LazyJsonlLoader, not the compact
                # ``index_records`` list (which may already be exclusion-filtered).
                record = self.lazy_loader[raw_index]
                defect_type = identify_ui_defect_task(record)[1]
                label = "positive" if is_positive_ui_defect(record) else "negative"
                self._balanced_logical_buckets[defect_type][label].append(logical_index)
        elif repeat_time < 1:
            if self.active_indices:
                partial_len = int(len(self.active_indices) * repeat_time)
                if partial_len > 0:
                    rnd = random.Random(10086)
                    sampled_indices = set(rnd.sample(self.active_indices, partial_len))
                    self.active_indices = [
                        index for index in self.active_indices if index in sampled_indices
                    ]
                    logger.info(f"[Dataset] {self.ds_name} Downsampled to {len(self.active_indices)} samples.")
                else:
                    self.active_indices = []
        
        if self.ui_sampling_mode not in {
            "task_balanced_all_records",
            "task_source_balanced_rotating",
        }:
            self._active_pool_length = len(self.active_indices)
            self._length = self._active_pool_length

        self._curriculum_group_cycle = None
        self._curriculum_group_view_indices = {}
        if self.curriculum_group_sampling:
            self._initialize_curriculum_group_cycle()

    def _initialize_curriculum_group_cycle(self) -> None:
        """Build one deterministic cyclic view stream per UI5 sample group."""

        grouped = defaultdict(list)
        for logical_index, raw_index in enumerate(self.active_indices):
            record = self.lazy_loader[raw_index]
            group_id = str(record.get("_ui5_sample_id") or "").strip()
            if not group_id:
                raise RuntimeError(
                    f"[{self.ds_name}] curriculum record lacks _ui5_sample_id: "
                    f"raw_index={raw_index}"
                )
            record_pool = canonical_curriculum_pool(
                record.get("_ui5_curriculum_pool")
            )
            if record_pool != self.curriculum_pool:
                raise RuntimeError(
                    f"[{self.ds_name}] curriculum record pool mismatch: "
                    f"record={record_pool}, dataset={self.curriculum_pool}, "
                    f"group={group_id}"
                )
            kind = str(record.get("_ui5_record_kind") or "").strip()
            if kind == "crop":
                crop_id = str(record.get("_ui5_crop_id") or "").strip()
                raw_crop_index = record.get("_ui5_crop_index")
                if not crop_id or isinstance(raw_crop_index, bool):
                    raise RuntimeError(
                        f"[{self.ds_name}] invalid crop identity for group {group_id}"
                    )
                try:
                    crop_index = int(raw_crop_index)
                    if crop_index < 0 or float(raw_crop_index) != crop_index:
                        raise ValueError
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"[{self.ds_name}] invalid crop index for group {group_id}"
                    ) from exc
                view_id = f"crop:{crop_index}:{crop_id}"
                sort_key = (crop_index, view_id)
            else:
                view_id = f"{kind or 'full_image'}:global"
                sort_key = (0, view_id)
            grouped[group_id].append(
                {
                    "logical_index": logical_index,
                    "raw_index": raw_index,
                    "kind": kind,
                    "view_id": view_id,
                    "sort_key": sort_key,
                    "base_tile_count": record.get("_ui5_base_tile_count"),
                }
            )

        group_views = {}
        view_lookup = {}
        for group_id, entries in sorted(grouped.items()):
            kinds = {entry["kind"] for entry in entries}
            if "crop" in kinds:
                if kinds != {"crop"}:
                    raise RuntimeError(
                        f"[{self.ds_name}] group {group_id} mixes crop and global views"
                    )
                declared_counts = {entry["base_tile_count"] for entry in entries}
                if declared_counts != {len(entries)}:
                    raise RuntimeError(
                        f"[{self.ds_name}] group {group_id} base-tile count mismatch: "
                        f"declared={declared_counts}, observed={len(entries)}"
                    )
                crop_indices = sorted(entry["sort_key"][0] for entry in entries)
                if crop_indices != list(range(len(entries))):
                    raise RuntimeError(
                        f"[{self.ds_name}] group {group_id} crop indices are not contiguous"
                    )
            elif len(entries) != 1:
                raise RuntimeError(
                    f"[{self.ds_name}] non-crop group {group_id} must have one global view; "
                    f"observed={len(entries)}"
                )
            ordered = sorted(entries, key=lambda item: item["sort_key"])
            views = []
            for entry in ordered:
                key = (group_id, entry["view_id"])
                if key in view_lookup:
                    raise RuntimeError(
                        f"[{self.ds_name}] duplicate curriculum view identity: {key}"
                    )
                view_lookup[key] = int(entry["logical_index"])
                views.append(str(entry["view_id"]))
            group_views[group_id] = views

        self._curriculum_group_cycle = CurriculumGroupCycle(group_views)
        self._curriculum_group_view_indices = view_lookup
        view_counts = [len(views) for views in group_views.values()]
        logger.warning(
            "[UI5 curriculum] dataset=%s sampling_unit=sample_group groups=%s "
            "records=%s views_per_group=min:%s,max:%s fingerprint=%s",
            self.ds_name,
            self._curriculum_group_cycle.group_count,
            len(self.active_indices),
            min(view_counts),
            max(view_counts),
            self._curriculum_group_cycle.fingerprint,
        )

    @property
    def curriculum_draw_length(self) -> int:
        if self._curriculum_group_cycle is not None:
            return self._curriculum_group_cycle.group_count
        return self._length

    def curriculum_group_identity(self) -> Optional[dict]:
        if self._curriculum_group_cycle is None:
            return None
        return self._curriculum_group_cycle.identity

    def curriculum_group_iterator_state(self, *, seed: int, global_idx: int) -> Optional[dict]:
        if self._curriculum_group_cycle is None:
            return None
        return self._curriculum_group_cycle.iterator_state(
            seed=int(seed), global_idx=int(global_idx)
        )

    def validate_curriculum_group_iterator_state(
        self, state: Mapping[str, Any], *, seed: int, global_idx: int
    ) -> None:
        if self._curriculum_group_cycle is None:
            if state is not None:
                raise RuntimeError(
                    f"[{self.ds_name}] unexpected curriculum group iterator state"
                )
            return
        if not isinstance(state, dict):
            raise RuntimeError(
                f"[{self.ds_name}] missing curriculum group iterator state"
            )
        self._curriculum_group_cycle.validate_iterator_state(
            state, seed=int(seed), global_idx=int(global_idx)
        )

    def __len__(self):
        return self._length

    @staticmethod
    def _evenly_interleave(positive: List[int], negative: List[int]) -> List[int]:
        """Spread minority examples over a class stream instead of clustering them."""
        output = []
        positive_index = 0
        negative_index = 0
        total = len(positive) + len(negative)
        for position in range(total):
            desired_positive = ((position + 1) * len(positive)) // max(total, 1)
            if desired_positive > positive_index:
                output.append(positive[positive_index])
                positive_index += 1
            else:
                output.append(negative[negative_index])
                negative_index += 1
        return output

    def get_epoch_indices(self, shuffle_seed: int, epoch_index: int = 0) -> List[int]:
        rng = random.Random(shuffle_seed)
        if self._source_balanced_plan is not None:
            base_seed = int(shuffle_seed) - int(epoch_index) * 999983
            return materialize_task_source_balanced_rotating_indices(
                self._source_balanced_plan,
                seed=base_seed,
                epoch_index=epoch_index,
            )
        if self._all_records_task_buckets is not None:
            streams = {}
            for defect_type, values in self._all_records_task_buckets.items():
                stream = list(values)
                rng.shuffle(stream)
                streams[defect_type] = stream
            task_order = sorted(streams)
            rng.shuffle(task_order)
            longest = max(len(stream) for stream in streams.values())
            indices = []
            for position in range(longest):
                rotated = (
                    task_order[position % len(task_order):]
                    + task_order[:position % len(task_order)]
                )
                indices.extend(
                    streams[defect_type][position % len(streams[defect_type])]
                    for defect_type in rotated
                )
            if len(indices) != self._length:
                raise RuntimeError(
                    f"[{self.ds_name}] all-record epoch length mismatch: "
                    f"{len(indices)} != {self._length}"
                )
            return indices
        if self._balanced_logical_buckets is None:
            indices = list(range(self._length))
            rng.shuffle(indices)
            return indices

        class_streams = {}
        for defect_type, buckets in self._balanced_logical_buckets.items():
            positive = list(buckets["positive"])
            negative = list(buckets["negative"])
            rng.shuffle(positive)
            rng.shuffle(negative)
            class_streams[defect_type] = self._evenly_interleave(positive, negative)

        class_order = list(sorted(class_streams))
        rng.shuffle(class_order)
        per_class = len(class_streams[class_order[0]])
        indices = []
        for position in range(per_class):
            # Rotate the starting class so adjacent packed batches do not always
            # see the same family first.
            rotated = class_order[position % len(class_order):] + class_order[:position % len(class_order)]
            indices.extend(class_streams[defect_type][position] for defect_type in rotated)
        return indices

    def seen_raw_indices(self, seed: int, global_idx: int) -> set:
        """Reconstruct records consumed by one deterministic iterator."""
        if self._curriculum_group_cycle is not None:
            seen = set()
            for draw_index in range(max(0, int(global_idx))):
                draw = self._curriculum_group_cycle.draw_at(
                    draw_index, seed=int(seed)
                )
                logical_index = self._curriculum_group_view_indices[
                    (draw["group_id"], draw["view_id"])
                ]
                seen.add(self.active_indices[logical_index])
            return seen
        seen = set()
        remaining = max(0, int(global_idx))
        epoch = 0
        while remaining:
            indices = self.get_epoch_indices(
                seed + epoch * 999983, epoch_index=epoch
            )
            take = min(remaining, len(indices))
            seen.update(self.active_indices[index] for index in indices[:take])
            remaining -= take
            epoch += 1
        return seen

    def sampling_inventory(self, seen_raw_indices: Optional[set] = None) -> dict:
        active_raw = set(self.active_indices)
        seen_raw = set() if seen_raw_indices is None else set(seen_raw_indices)
        seen_raw &= active_raw
        record_kinds = defaultdict(int)
        crop_sources = defaultdict(int)
        task_counts = defaultdict(lambda: {"positive": 0, "negative": 0})
        source_crop_counts = defaultdict(int)
        active_source_ids = set()
        seen_source_ids = set()
        active_crop_raw = set()
        seen_crop_raw = set()
        manual_active = set()
        manual_seen = set()
        for raw_index in active_raw:
            record = self.lazy_loader[raw_index]
            kind = str(record.get("_ui5_record_kind", "unknown"))
            source = str(record.get("_ui5_crop_source") or "full_image")
            task = identify_ui_defect_task(record)
            task_name = task[0] if task else "unknown"
            polarity = "positive" if is_positive_ui_defect(record) else "negative"
            source_id = str(
                record.get("_ui5_image_id")
                or record.get("_ui5_source_image")
                or record.get("image", "")
            )
            record_kinds[kind] += 1
            crop_sources[source] += 1
            task_counts[task_name][polarity] += 1
            active_source_ids.add(source_id)
            if kind == "crop":
                active_crop_raw.add(raw_index)
                source_crop_counts[source_id] += 1
            if source == "manual_gt_repair":
                manual_active.add(raw_index)
            if raw_index in seen_raw:
                seen_source_ids.add(source_id)
                if kind == "crop":
                    seen_crop_raw.add(raw_index)
                if source == "manual_gt_repair":
                    manual_seen.add(raw_index)
        crop_count_values = sorted(source_crop_counts.values())
        task_inventory = {
            task: {
                **dict(values),
                "negative_to_positive_ratio": (
                    float(values["negative"]) / float(values["positive"])
                    if values["positive"]
                    else None
                ),
            }
            for task, values in sorted(task_counts.items())
        }
        never_active = max(
            0,
            self._raw_recipe_rows
            - self._excluded_record_count
            - len(active_raw),
        )
        return {
            "dataset": self.ds_name,
            "sampling_mode": self.ui_sampling_mode,
            "sampling_unit": (
                "sample_group"
                if self._curriculum_group_cycle is not None
                else "record"
            ),
            "sample_groups": (
                self._curriculum_group_cycle.group_count
                if self._curriculum_group_cycle is not None
                else len(active_raw)
            ),
            "group_cycle_fingerprint": (
                self._curriculum_group_cycle.fingerprint
                if self._curriculum_group_cycle is not None
                else None
            ),
            "raw_recipe_records": self._raw_recipe_rows,
            "excluded_records": self._excluded_record_count,
            "active_pool_records": len(active_raw),
            "effective_epoch_records": self._length,
            "never_entered_active_pool_legal_records": never_active,
            "never_active_legal_records": never_active,
            "record_kinds": dict(sorted(record_kinds.items())),
            "crop_sources": dict(sorted(crop_sources.items())),
            "tasks": task_inventory,
            "unique_source_images": len(active_source_ids),
            "source_crop_count": {
                "min": min(crop_count_values, default=0),
                "max": max(crop_count_values, default=0),
                "mean": (
                    sum(crop_count_values) / len(crop_count_values)
                    if crop_count_values else 0.0
                ),
            },
            "seen_unique_records": len(seen_raw),
            "seen_record_coverage": len(seen_raw) / max(1, len(active_raw)),
            "seen_unique_crops": len(seen_crop_raw),
            "seen_crop_coverage": len(seen_crop_raw) / max(1, len(active_crop_raw)),
            "active_crop_retention": (
                len(active_crop_raw) / self._legal_crop_record_count
                if self._legal_crop_record_count
                else 1.0
            ),
            "seen_unique_source_images": len(seen_source_ids),
            "seen_source_image_coverage": len(seen_source_ids) / max(1, len(active_source_ids)),
            "manual_repair_active": len(manual_active),
            "manual_repair_seen": len(manual_seen),
            "manual_repair_required": len(self._raw_manual_repair_indices),
            "manual_repair_retention": (
                len(manual_active) / len(self._raw_manual_repair_indices)
                if self._raw_manual_repair_indices
                else 1.0
            ),
            "source_balanced_rotation": (
                {
                    "positive_slots_per_task": self._source_balanced_plan[
                        "positive_slots_per_task"
                    ],
                    "negative_slots_per_task": self._source_balanced_plan[
                        "negative_slots_per_task"
                    ],
                    "effective_negative_to_positive_ratio": self._source_balanced_plan[
                        "negative_to_positive_ratio"
                    ],
                    "source_groups_by_task": self._source_balanced_plan[
                        "source_groups_by_task"
                    ],
                    "records_by_task": self._source_balanced_plan[
                        "records_by_task"
                    ],
                    "source_group_repeat_draws_per_epoch_by_task": self._source_balanced_plan[
                        "source_group_repeat_draws_per_epoch_by_task"
                    ],
                }
                if self._source_balanced_plan is not None
                else None
            ),
        }

    def get_targets_flag_with_mtp(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Create MTP (Multi-Token Prediction) blocks with proper labels."""
        tokenizer = self.processor.tokenizer
        targets_flag = torch.zeros_like(input_ids)
        
        box_end_id = tokenizer.convert_tokens_to_ids("</box>")
        ref_end_id = tokenizer.convert_tokens_to_ids("</ref>")
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        null_id = tokenizer.convert_tokens_to_ids("<null>")
        mask_id = tokenizer.convert_tokens_to_ids("<text_mask>")
        
        start_header_idxs = torch.where(
            input_ids == tokenizer.convert_tokens_to_ids("<|im_start|>")
        )[0]
        assistant_idxs = torch.where(
            input_ids == tokenizer.convert_tokens_to_ids("assistant")
        )[0]
        eot_idxs = torch.where(input_ids == eos_id)[0]
        
        # Identify assistant response positions for mask block generation
        # 同时记录每个 assistant 回复内 label 区间 [label_start, label_end]（含）
        resp_start_position_ids = []
        resp_end_position_ids = []
        resp_label_ranges = []
        
        for assistant_idx in assistant_idxs:
            sets = list(set(start_header_idxs + 1))
            sets = [each.item() for each in sets]
            if assistant_idx.item() in sets:
                st = assistant_idx + 1
                for eot_idx in eot_idxs:
                    if eot_idx > st:
                        # st+1 到 eot_idx（含）是需要监督的 token 区间
                        targets_flag[st+1: eot_idx + 1] = 1
                        resp_start_position_ids.append((st).item())
                        resp_end_position_ids.append((eot_idx + 1).item())
                        label_start = (st + 1).item()
                        label_end = eot_idx.item()
                        resp_label_ranges.append((label_start, label_end))
                        break
        
        targets = input_ids.clone()
        assert targets_flag.sum() > 0, f"No valid labels for training, skip sample in {self.ds_name}"
        targets[targets_flag == 0] = IGNORE_TOKEN_ID
        
        input_ids_np = input_ids.squeeze(0).cpu().numpy()
        targets_np = targets.squeeze(0).cpu().numpy()
        len_input_ids = len(input_ids_np)
        
        # ========= 分支 1：无检测序列标记（</box>、</ref>），采用随机 block 切分 =========
        has_box = (input_ids == box_end_id).any().item()
        has_ref = (input_ids == ref_end_id).any().item()
        if not (has_box or has_ref):
            all_mask_input_ids = []
            all_mask_targets = []
            all_mask_positions = []

            # 仅在单个 assistant 回复内部随机切分
            for label_start, label_end in resp_label_ranges:
                # 该回复内部的有效 supervision 位置（去掉 IGNORE_TOKEN_ID）
                resp_valid_positions = [
                    i for i in range(label_start, label_end + 1)
                    if targets_np[i] != IGNORE_TOKEN_ID
                ]
                num_valid = len(resp_valid_positions)
                if num_valid == 0:
                    continue

                num_blocks = num_valid // self.block_size
                if num_blocks == 0:
                    continue

                # 为了保证可复现性，随机数种子在外部 __getitem__ 中已设置
                used_tokens = num_blocks * self.block_size
                remaining = num_valid - used_tokens  # 0 <= remaining < self.block_size
                # 在 [0, remaining] 中随机选择一个 offset，使得切分起点是随机的
                offset = np.random.randint(0, remaining + 1) if remaining > 0 else 0

                for block_idx in range(num_blocks):
                    start_pos = offset + block_idx * self.block_size
                    end_pos = start_pos + self.block_size
                    block_valid = resp_valid_positions[start_pos:end_pos]
                    if len(block_valid) == 0:
                        continue

                    first_token_idx = int(block_valid[0])
                    anchor_idx = max(first_token_idx - 1, 0)

                    # 构建 target block：长度为 block_size，默认全 IGNORE_TOKEN_ID
                    target_block = np.full(self.block_size, IGNORE_TOKEN_ID, dtype=targets_np.dtype)
                    candidate_tokens = input_ids_np[block_valid]

                    for i, tok in enumerate(candidate_tokens):
                        if i >= self.block_size:
                            break
                        target_block[i] = tok
                        if tok == eos_id:
                            # EOS 之后保持 IGNORE_TOKEN_ID，不再计算 loss
                            break

                    # 构建输入 block：第一个位置是 anchor，其余为 <text_mask>
                    mask_input_block = np.full(self.block_size, mask_id, dtype=input_ids_np.dtype)
                    mask_input_block[0] = input_ids_np[anchor_idx]

                    # 位置 id 与原序列对齐，从 anchor 开始连续增长
                    pos_block = np.arange(anchor_idx, anchor_idx + self.block_size, dtype=np.int32)

                    all_mask_input_ids.append(mask_input_block)
                    all_mask_targets.append(target_block)
                    all_mask_positions.append(pos_block)

            if len(all_mask_input_ids) > 0:
                final_mask_ids = np.concatenate(all_mask_input_ids)
                final_mask_targets = np.concatenate(all_mask_targets)
                final_mask_positions = np.concatenate(all_mask_positions)

                pad_id = tokenizer.pad_token_id
                bridge_ignore = np.array([IGNORE_TOKEN_ID], dtype=targets_np.dtype)
                pad_token = np.array([pad_id], dtype=input_ids_np.dtype)

                input_ids_out = np.concatenate([input_ids_np, final_mask_ids, pad_token])
                targets_out = np.concatenate([targets_np, bridge_ignore, final_mask_targets])

                orig_pos = np.arange(len_input_ids, dtype=np.int32)
                pad_pos = np.array([final_mask_positions[-1] + 1], dtype=np.int32)
                position_ids_out = np.concatenate([orig_pos, final_mask_positions, pad_pos])
            else:
                input_ids_out = input_ids_np
                targets_out = targets_np
                position_ids_out = np.arange(len_input_ids, dtype=np.int32)

            input_ids = torch.tensor(input_ids_out, dtype=torch.long)
            targets = torch.tensor(targets_out, dtype=torch.long)
            position_ids = torch.tensor(position_ids_out, dtype=torch.long)

            return dict(
                input_ids=input_ids,
                labels=targets,
                attention_mask=input_ids.ne(tokenizer.pad_token_id),
                position_ids=position_ids,
            )

        # ========= 分支 2：检测序列标记存在，使用原有的 box/ref-aware 逻辑 =========
        all_mask_input_ids = []
        all_mask_targets = []
        all_mask_positions = []
        
        for start, end in zip(resp_start_position_ids, resp_end_position_ids):
            curr = start
            
            while curr < end:
                anchor_token = input_ids_np[curr]
                if anchor_token == eos_id:
                    break
                
                pred_start = curr + 1
                if pred_start > end:
                    break
                
                candidates = input_ids_np[pred_start : min(pred_start + self.block_size, end + 1)]
                if len(candidates) == 0:
                    break

                valid_len = len(candidates)
                
                eos_indices = np.where(candidates == eos_id)[0]
                if len(eos_indices) > 0:
                    first_eos_idx = eos_indices[0]
                    if first_eos_idx == 0:
                        valid_len = 1
                    else:
                        valid_len = first_eos_idx
                
                if valid_len > 1 or (len(eos_indices) > 0 and eos_indices[0] != 0):
                    ref_indices = np.where(candidates[:valid_len] == ref_end_id)[0]
                    if len(ref_indices) > 0:
                        valid_len = min(valid_len, ref_indices[0] + 1)
                    
                    box_indices = np.where(candidates[:valid_len] == box_end_id)[0]
                    if len(box_indices) > 0:
                        valid_len = min(valid_len, box_indices[0] + 1)

                target_block = np.full(self.block_size, null_id, dtype=input_ids_np.dtype)
                target_block[:valid_len] = candidates[:valid_len]
                
                mask_input_block = np.full(self.block_size, mask_id, dtype=input_ids_np.dtype)
                mask_input_block[0] = anchor_token 
                
                pos_block = np.arange(curr, curr + self.block_size, dtype=np.int32)
                
                all_mask_input_ids.append(mask_input_block)
                all_mask_targets.append(target_block)
                all_mask_positions.append(pos_block)
                
                curr += valid_len
                
        if len(all_mask_input_ids) > 0:
            final_mask_ids = np.concatenate(all_mask_input_ids)
            final_mask_targets = np.concatenate(all_mask_targets)
            final_mask_positions = np.concatenate(all_mask_positions)
            
            pad_id = tokenizer.pad_token_id
            bridge_ignore = np.array([IGNORE_TOKEN_ID], dtype=targets_np.dtype)
            pad_token = np.array([pad_id], dtype=input_ids_np.dtype)
            
            input_ids_out = np.concatenate([input_ids_np, final_mask_ids, pad_token])
            targets_out = np.concatenate([targets_np, bridge_ignore, final_mask_targets])
            
            orig_pos = np.arange(len_input_ids, dtype=np.int32)
            pad_pos = np.array([final_mask_positions[-1] + 1], dtype=np.int32)
            position_ids_out = np.concatenate([orig_pos, final_mask_positions, pad_pos])
        else:
            input_ids_out = input_ids_np
            targets_out = targets_np
            position_ids_out = np.arange(len_input_ids, dtype=np.int32)
        
        input_ids = torch.tensor(input_ids_out, dtype=torch.long)
        targets = torch.tensor(targets_out, dtype=torch.long)
        position_ids = torch.tensor(position_ids_out, dtype=torch.long)
        
        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            position_ids=position_ids,
        )

    def _validate_image_token_alignment(self, input_ids: torch.Tensor, pixel_values, image_grid_hws) -> None:
        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            image_token = getattr(self.processor, "image_token", IMG_CONTEXT_TOKEN)
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)

        if isinstance(image_grid_hws, torch.Tensor):
            grid_array = image_grid_hws.detach().cpu().numpy()
        else:
            grid_array = np.asarray(image_grid_hws)

        merge_kernel = getattr(self.processor.image_processor, "merge_kernel_size", [2, 2])
        expected_context_tokens = int(
            sum(int(h) * int(w) // (int(merge_kernel[0]) * int(merge_kernel[1])) for h, w in grid_array)
        )
        actual_context_tokens = int((input_ids == image_token_id).sum().item())
        if actual_context_tokens != expected_context_tokens:
            raise ValueError(
                f"[{self.ds_name}] image token mismatch: actual={actual_context_tokens}, "
                f"expected={expected_context_tokens}, num_images={len(grid_array)}, "
                f"grid_hws={grid_array.tolist()}"
            )

        expected_patches = int(sum(int(h) * int(w) for h, w in grid_array))
        actual_patches = int(pixel_values.shape[0])
        if actual_patches != expected_patches:
            raise ValueError(
                f"[{self.ds_name}] pixel patch mismatch: actual={actual_patches}, "
                f"expected={expected_patches}, num_images={len(grid_array)}, "
                f"grid_hws={grid_array.tolist()}"
            )

    def multi_modal_get_item(
        self, messages: list, ui_targets: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        message_text = self.processor.py_apply_chat_template(messages, tokenize=False)
        image_inputs, video_inputs = self.processor.process_vision_info(messages)
        
        if image_inputs is not None:
            image_inputs = [
                apply_resize_augmentation(
                    img, data_augment=self.data_augment,
                    min_long_edge=640, max_long_edge=2560, augment_prob=0.5
                ) for img in image_inputs
            ]
        
        inputs = self.processor(
            text=message_text, images=image_inputs, videos=video_inputs,
            return_tensors="pt", padding=False, truncation=True
        )
        input_ids = inputs["input_ids"][0]

        if "pixel_values" not in inputs:
            pixel_values = torch.zeros((4, 3, 14, 14), dtype=torch.float32)
            image_flags = torch.tensor([0], dtype=torch.long)
            image_grid_hws = np.array([[2, 2]])
        else:
            pixel_values = inputs["pixel_values"]
            image_grid_hws = inputs["image_grid_hws"]
            image_flags = torch.tensor([len(inputs["image_grid_hws"])], dtype=torch.long)
            self._validate_image_token_alignment(input_ids, pixel_values, image_grid_hws)

        labels_dict = self.get_targets_flag_with_mtp(input_ids)
        
        result = dict(
            input_ids=labels_dict["input_ids"],
            labels=labels_dict["labels"],
            position_ids=labels_dict["position_ids"],
            attention_mask=labels_dict["attention_mask"],
            image_flags=image_flags,
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
        )
        if ui_targets is not None:
            result.update(ui_targets)
        return result

    def _materialize_logical_index(self, logical_index: int) -> Dict[str, torch.Tensor]:
        real_idx = self.active_indices[int(logical_index)]
        data_item = self.lazy_loader[real_idx]
        ui_targets = extract_ui_defect_targets(data_item, max_boxes=8)
        data_item = process_multimodal_sample(
            data_item, self.root, self.max_frames,
            self.target_fps, self.video_total_pixels,
            visual_prompt=self.visual_prompt,
        )
        return self.multi_modal_get_item(data_item, ui_targets=ui_targets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        retry_count = 0
        current_idx = idx
        
        seed = int(idx + 10086)
        random.seed(seed)
        np.random.seed(seed)
        
        while retry_count <= 10:
            logical_index = (current_idx + retry_count) % self._active_pool_length
            real_idx = self.active_indices[logical_index]
            try:
                return self._materialize_logical_index(logical_index)
            except Exception as e:
                tb = traceback.format_exc()
                logger.warning(f"[{self.ds_name}] idx {real_idx} failed: {e}\n{tb}")
                retry_count += 1
        
        raise RuntimeError(f"[{self.ds_name}] Failed after 10 retries")
    
    def get_sample_at_global_idx(self, global_idx: int, seed: int) -> Dict[str, torch.Tensor]:
        """Get a sample by global index (used for resume)."""
        if self._curriculum_group_cycle is not None:
            draw = self._curriculum_group_cycle.draw_at(
                int(global_idx), seed=int(seed)
            )
            logical_index = self._curriculum_group_view_indices[
                (draw["group_id"], draw["view_id"])
            ]
            deterministic_seed = int(logical_index + 10086)
            random.seed(deterministic_seed)
            np.random.seed(deterministic_seed)
            try:
                return self._materialize_logical_index(logical_index)
            except Exception as exc:
                raise RuntimeError(
                    f"[{self.ds_name}] exact curriculum group/view failed: "
                    f"group={draw['group_id']} view={draw['view_id']} "
                    f"global_idx={global_idx}"
                ) from exc
        ds_len = self._length
        if ds_len == 0:
            raise ValueError("Dataset is empty")
        
        epoch = global_idx // ds_len
        pos = global_idx % ds_len
        
        shuffle_seed = seed + epoch * 999983
        indices = self.get_epoch_indices(shuffle_seed, epoch_index=epoch)
        
        real_idx = indices[pos]
        return self[real_idx]


@dataclass
class IteratorState:
    """Iterator state."""
    seed: int
    global_idx: int
    
    def to_dict(self) -> dict:
        return {'seed': self.seed, 'global_idx': self.global_idx}
    
    @classmethod
    def from_dict(cls, d: dict) -> 'IteratorState':
        return cls(seed=d['seed'], global_idx=d['global_idx'])


class DeterministicIterator:
    """Deterministic dataset iterator."""
    
    def __init__(self, dataset: LazySupervisedDatasetMTP, seed: int, start_global_idx: int = 0):
        self.dataset = dataset
        self.seed = seed
        self.ds_len = int(dataset.curriculum_draw_length)
        self.ds_name = getattr(dataset, 'ds_name', 'unknown')
        self.global_idx = start_global_idx
        
        self._cached_epoch = -1
        self._cached_indices = None
    
    def _get_epoch_indices(self, epoch: int) -> list:
        if self._cached_epoch == epoch and self._cached_indices is not None:
            return self._cached_indices
        
        shuffle_seed = self.seed + epoch * 999983
        indices = self.dataset.get_epoch_indices(shuffle_seed, epoch_index=epoch)
        
        self._cached_epoch = epoch
        self._cached_indices = indices
        return indices
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[dict, int]:
        if self.ds_len == 0:
            raise StopIteration
        
        current_global_idx = self.global_idx
        if self.dataset.curriculum_group_sampling:
            sample = self.dataset.get_sample_at_global_idx(
                current_global_idx, self.seed
            )
            self.global_idx += 1
            return sample, current_global_idx
        epoch = current_global_idx // self.ds_len
        pos = current_global_idx % self.ds_len
        indices = self._get_epoch_indices(epoch)
        
        real_idx = indices[pos]
        sample = self.dataset[real_idx]
        self.global_idx += 1
        
        return sample, current_global_idx
    
    def peek_global_idx(self) -> int:
        return self.global_idx
    
    def state_dict(self) -> dict:
        state = IteratorState(seed=self.seed, global_idx=self.global_idx).to_dict()
        group_state = self.dataset.curriculum_group_iterator_state(
            seed=self.seed, global_idx=self.global_idx
        )
        if group_state is not None:
            state["curriculum_group_cycle"] = group_state
        return state
    
    @classmethod
    def from_state_dict(cls, dataset: LazySupervisedDatasetMTP, state: dict) -> 'DeterministicIterator':
        iterator = cls(
            dataset=dataset,
            seed=state['seed'],
            start_global_idx=state['global_idx'],
        )
        dataset.validate_curriculum_group_iterator_state(
            state.get("curriculum_group_cycle"),
            seed=iterator.seed,
            global_idx=iterator.global_idx,
        )
        return iterator


@dataclass
class WorkerState:
    """Complete state of a worker, including buffer state."""
    iterator_states: List[dict]
    sample_rng_state: tuple
    samples_produced: int
    batches_produced: int
    dataset_sampler_draws: List[int]
    current_batch_locations: List[Tuple[int, int]]
    buffer_locations: List[Tuple[int, int]]
    deferred_locations: List[Tuple[int, int]]
    
    def to_dict(self) -> dict:
        return {
            'iterator_states': self.iterator_states,
            'sample_rng_state': self.sample_rng_state,
            'samples_produced': self.samples_produced,
            'batches_produced': self.batches_produced,
            'dataset_sampler_draws': self.dataset_sampler_draws,
            'current_batch_locations': self.current_batch_locations,
            'buffer_locations': self.buffer_locations,
            'deferred_locations': self.deferred_locations,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'WorkerState':
        raw_rng_state = d['sample_rng_state']
        if isinstance(raw_rng_state, (list, tuple)) and len(raw_rng_state) == 3:
            version, internal_state, gauss_next = raw_rng_state
            if isinstance(internal_state, list):
                internal_state = tuple(internal_state)
            sample_rng_state = (version, internal_state, gauss_next)
        else:
            sample_rng_state = tuple(raw_rng_state) if isinstance(raw_rng_state, list) else raw_rng_state
        iterator_states = d['iterator_states']
        dataset_sampler_draws = d.get('dataset_sampler_draws')
        if dataset_sampler_draws is None:
            # Version-7 checkpoints predate deferred replay, so every sampler
            # selection advanced exactly one dataset iterator.
            dataset_sampler_draws = [
                int(state.get('global_idx', 0)) for state in iterator_states
            ]
        
        return cls(
            iterator_states=iterator_states,
            sample_rng_state=sample_rng_state,
            samples_produced=d.get('samples_produced', 0),
            batches_produced=d.get('batches_produced', 0),
            dataset_sampler_draws=list(dataset_sampler_draws),
            current_batch_locations=d.get('current_batch_locations', []),
            buffer_locations=d.get('buffer_locations', []),
            deferred_locations=d.get('deferred_locations', []),
        )


class StreamPackedDatasetMTP(IterableDataset):
    """Online packing IterableDataset with MTP support and perfect stateful resume."""
    
    def __init__(
        self,
        tokenizer,
        data_rank: int,
        data_world_size: int,
        datasets: List[LazySupervisedDatasetMTP],
        dataset_weight: List[float] = None,
        max_num_tokens_per_sample: int = 16384,
        max_num_tokens: int = 36864,
        log_freq: int = 10000,
        base_seed: int = 42,
        buffer_size: int = 32,
        curriculum_schedule: Optional[UI5CurriculumSchedule] = None,
        dataset_pools: Optional[List[str]] = None,
        curriculum_identity: Optional[dict] = None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.datasets = datasets
        self.max_num_tokens_per_sample = max_num_tokens_per_sample
        self.max_num_tokens = max_num_tokens
        self.log_freq = log_freq
        self.base_seed = base_seed
        self.buffer_size = buffer_size

        if dataset_weight is None:
            dataset_weight = [1] * len(datasets)
        total_weight = sum(dataset_weight)
        if total_weight <= 0:
            raise ValueError("dataset weights must sum to a positive value")
        self.base_dataset_weight = [float(w) / total_weight for w in dataset_weight]
        self.curriculum_schedule = curriculum_schedule
        self.curriculum_identity = copy.deepcopy(curriculum_identity)
        self.dataset_pools = (
            [canonical_curriculum_pool(pool) for pool in dataset_pools]
            if dataset_pools is not None
            else None
        )
        if self.curriculum_schedule is not None:
            if self.dataset_pools is None or len(self.dataset_pools) != len(self.datasets):
                raise ValueError(
                    "Scheduled curriculum requires one curriculum_pool per dataset"
                )
            if any(not dataset.curriculum_group_sampling for dataset in self.datasets):
                raise ValueError(
                    "Scheduled UI5 curriculum requires sample-group sampling for every pool"
                )
            if len(self.dataset_pools) != len(CURRICULUM_POOLS) or set(
                self.dataset_pools
            ) != set(CURRICULUM_POOLS):
                raise ValueError(
                    "Scheduled UI5 curriculum requires exactly one dataset per pool: "
                    f"observed={self.dataset_pools}"
                )
            self._curriculum_completed_global_step = 0
            self._curriculum_stage_index = 0
            self.dataset_weight = self.curriculum_schedule.effective_dataset_weights(
                dataset_pools=self.dataset_pools,
                base_weights=self.base_dataset_weight,
                completed_global_step=0,
            )
        else:
            self._curriculum_completed_global_step = 0
            self._curriculum_stage_index = None
            self.dataset_weight = list(self.base_dataset_weight)
        self._configured_num_workers: Optional[int] = None
        self._saved_num_workers: Optional[int] = None
        self._strict_resume = self.curriculum_schedule is not None
        self._resume_loaded = False
        
        self._worker_states: Dict[str, dict] = {}
        self._resume_states: Dict[str, dict] = {}
        
        if get_rank() == 0:
            ds_info = '\n'.join([f'  {ds.ds_name}: weight={w*100:.2f}%, len={len(ds)}' 
                                 for ds, w in zip(self.datasets, self.dataset_weight)])
            logger.info(f'StreamPackedDatasetMTP initialized:\n'
                       f'  max_num_tokens_per_sample={max_num_tokens_per_sample}\n'
                       f'  max_num_tokens={max_num_tokens}\n'
                       f'  buffer_size={buffer_size}\n'
                       f'  base_seed={base_seed}\n'
                       f'  data_rank={data_rank}, data_world_size={data_world_size}\n'
                       f'Datasets:\n{ds_info}')

    def configure_num_workers(self, num_workers: int) -> None:
        configured = int(num_workers)
        if configured < 0:
            raise ValueError("dataloader_num_workers must be non-negative")
        # IterableDataset uses the main process as one logical worker when
        # DataLoader workers are disabled.
        self._configured_num_workers = max(configured, 1)

    def _stream_resume_config(self) -> dict:
        return {
            "base_seed": int(self.base_seed),
            "data_world_size": int(self.data_world_size),
            "max_num_tokens_per_sample": int(self.max_num_tokens_per_sample),
            "max_num_tokens": int(self.max_num_tokens),
            "buffer_size": int(self.buffer_size),
            "datasets": [
                {
                    "name": dataset.ds_name,
                    "rows": len(dataset),
                    "sampling_unit": (
                        "sample_group"
                        if dataset.curriculum_group_sampling
                        else "record"
                    ),
                    "curriculum_group_identity": dataset.curriculum_group_identity(),
                    "base_probability": float(probability),
                    "curriculum_pool": (
                        self.dataset_pools[index]
                        if self.dataset_pools is not None
                        else None
                    ),
                }
                for index, (dataset, probability) in enumerate(
                    zip(self.datasets, self.base_dataset_weight)
                )
            ],
            "curriculum_schedule": (
                self.curriculum_schedule.to_dict()
                if self.curriculum_schedule is not None
                else None
            ),
            "curriculum_artifact_identity": copy.deepcopy(
                self.curriculum_identity
            ),
        }

    def state_dict(self, *, completed_global_step: Optional[int] = None) -> dict:
        if completed_global_step is not None:
            self._curriculum_completed_global_step = int(completed_global_step)
        state = {
            'worker_states': copy.deepcopy(self._worker_states),
            'base_seed': self.base_seed,
            'stream_resume_config': self._stream_resume_config(),
            'num_workers': self._configured_num_workers,
            'version': 8,
        }
        if self.curriculum_schedule is not None:
            state["curriculum_sampler"] = self.curriculum_schedule.sampler_state(
                completed_global_step=self._curriculum_completed_global_step,
                sampling_stage_index=int(self._curriculum_stage_index),
            )
        return state

    def load_state_dict(self, state: dict, *, expected_global_step: Optional[int] = None):
        version = state.get('version', 1)
        if self._strict_resume and version < 8:
            raise RuntimeError(
                "Scheduled UI5 curriculum requires dataloader state version >= 8; "
                f"checkpoint has version={version}"
            )
        if version < 3:
            logger.warning(f"Loading old state version {version}, perfect resume not available.")

        saved_config = state.get("stream_resume_config")
        current_config = self._stream_resume_config()
        if self._strict_resume and saved_config != current_config:
            raise RuntimeError(
                "UI5 stream configuration changed across resume: "
                f"saved={saved_config}, current={current_config}"
            )
        saved_num_workers = state.get("num_workers")
        self._saved_num_workers = (
            int(saved_num_workers) if saved_num_workers is not None else None
        )
        if (
            self._strict_resume
            and self._configured_num_workers is not None
            and self._saved_num_workers != self._configured_num_workers
        ):
            raise RuntimeError(
                "dataloader worker count changed across resume: "
                f"saved={self._saved_num_workers}, current={self._configured_num_workers}"
            )

        worker_states = state.get('worker_states')
        if not isinstance(worker_states, dict) or not worker_states:
            if self._strict_resume:
                raise RuntimeError("Scheduled UI5 checkpoint has no worker_states")
            return
        if (
            self._strict_resume
            and self._configured_num_workers is not None
            and len(worker_states) != self._configured_num_workers
        ):
            raise RuntimeError(
                "worker state count does not match dataloader_num_workers: "
                f"saved={len(worker_states)}, current={self._configured_num_workers}"
            )
        for worker_key, worker_state in worker_states.items():
            if not isinstance(worker_state, dict):
                raise RuntimeError(f"Invalid state for {worker_key}: expected a mapping")
            iterator_states = worker_state.get("iterator_states")
            if not isinstance(iterator_states, list) or len(iterator_states) != len(
                self.datasets
            ):
                raise RuntimeError(
                    f"Invalid iterator state count for {worker_key}: "
                    f"saved={len(iterator_states) if isinstance(iterator_states, list) else None}, "
                    f"datasets={len(self.datasets)}"
                )
            if "sample_rng_state" not in worker_state:
                raise RuntimeError(f"Missing sample_rng_state for {worker_key}")
            if version >= 8:
                sampler_draws = worker_state.get("dataset_sampler_draws")
                if not isinstance(sampler_draws, list) or len(sampler_draws) != len(
                    self.datasets
                ) or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in sampler_draws
                ):
                    raise RuntimeError(
                        f"Invalid dataset_sampler_draws for {worker_key}"
                    )
                DeferredSampleLocations(
                    list(worker_state.get("deferred_locations", []))
                    + list(worker_state.get("current_batch_locations", []))
                    + list(worker_state.get("buffer_locations", [])),
                    dataset_count=len(self.datasets),
                    iterator_states=iterator_states,
                )
                for dataset, iterator_state in zip(self.datasets, iterator_states):
                    dataset.validate_curriculum_group_iterator_state(
                        iterator_state.get("curriculum_group_cycle"),
                        seed=int(iterator_state["seed"]),
                        global_idx=int(iterator_state["global_idx"]),
                    )

        if self.curriculum_schedule is not None:
            if expected_global_step is None:
                raise RuntimeError(
                    "expected_global_step is required for scheduled curriculum resume"
                )
            curriculum_state = state.get("curriculum_sampler")
            if not isinstance(curriculum_state, dict):
                raise RuntimeError("Scheduled UI5 checkpoint lacks curriculum_sampler state")
            self.curriculum_schedule.validate_sampler_state(
                curriculum_state, expected_global_step=int(expected_global_step)
            )
            saved_stage = int(curriculum_state["sampling_stage_index"])
            resume_stage = self.curriculum_schedule.stage_after_completed_step(
                int(expected_global_step)
            ).index
            worker_states, transition = prepare_worker_states_for_resume(
                worker_states,
                saved_sampling_stage=saved_stage,
                resume_sampling_stage=resume_stage,
            )
            if saved_stage != resume_stage:
                logger.warning(
                    "[UI5 curriculum] stage transition %s -> %s at global_step=%s; "
                    "deferred pending current_batch=%s buffer=%s existing=%s total=%s "
                    "for new-stage pool selection while preserving iterator and RNG state",
                    saved_stage,
                    resume_stage,
                    expected_global_step,
                    transition["current_batch_samples"],
                    transition["buffer_samples"],
                    transition["already_deferred_samples"],
                    transition["deferred_samples"],
                )
            self._curriculum_completed_global_step = int(expected_global_step)
            self._curriculum_stage_index = resume_stage
            self.dataset_weight = self.curriculum_schedule.effective_dataset_weights(
                dataset_pools=self.dataset_pools,
                base_weights=self.base_dataset_weight,
                completed_global_step=int(expected_global_step),
            )

        self._resume_states = copy.deepcopy(worker_states)
        self._worker_states = copy.deepcopy(worker_states)
        self._resume_loaded = True
        if get_rank() == 0:
            logger.info(f"Loaded resume states for {len(self._resume_states)} workers")

    def _get_sample_length(self, sample: Optional[dict]) -> int:
        if sample is None:
            return 0
        return sample['input_ids'].size(0)

    def _merge_samples(self, batch: Optional[dict], sample: dict) -> dict:
        """Merge a sample into the batch, tracking sample lengths."""
        sample_len = sample['input_ids'].size(0)
        
        if batch is None:
            result = copy.copy(sample)
            result['_sample_lengths'] = [sample_len]
            return result
        
        result = {}
        for k in batch:
            if k == '_sample_lengths':
                result[k] = batch[k] + [sample_len]
            elif k == 'image_grid_hws':
                if isinstance(batch[k], np.ndarray) and isinstance(sample[k], np.ndarray):
                    result[k] = np.concatenate([batch[k], sample[k]], axis=0)
                else:
                    result[k] = torch.cat([batch[k], sample[k]])
            elif k == 'pixel_values':
                result[k] = torch.cat([batch[k], sample[k]])
            elif isinstance(batch[k], torch.Tensor):
                result[k] = torch.cat([batch[k], sample[k]])
            else:
                result[k] = batch[k]
        
        return result

    def _finalize_batch(self, batch: dict) -> dict:
        """Finalize batch by computing sub_sample_lengths."""
        sample_lengths = batch.pop('_sample_lengths', [batch['input_ids'].size(0)])
        sub_sample_lengths = torch.tensor(sample_lengths, dtype=torch.long)
        
        batch['sub_sample_lengths'] = sub_sample_lengths
        # attention_mask is not needed here; model will use sub_sample_lengths to generate data_index
        
        return batch

    def __iter__(self):
        from torch.utils.data import get_worker_info
        
        worker_info = get_worker_info()
        local_worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        
        global_worker_id = num_workers * self.data_rank + local_worker_id
        worker_key = f'worker_{global_worker_id}'
        is_main_log = (global_worker_id == 0)
        
        if is_main_log:
            logger.info(f'[{worker_key}] Starting iteration with MTP Buffer Strategy...')
        
        # Initialize state
        worker_seed = self.base_seed + global_worker_id
        sample_rng = random.Random(worker_seed)
        
        iterators: List[DeterministicIterator] = []
        iterator_seeds: List[int] = []
        for ds_idx, ds in enumerate(self.datasets):
            iter_seed = self.base_seed + global_worker_id * 10000 + ds_idx
            iterator_seeds.append(iter_seed)
            iterators.append(DeterministicIterator(ds, seed=iter_seed, start_global_idx=0))
        
        samples_produced = 0
        batches_produced = 0
        skipped_count = 0
        dataset_sampler_draws = [0 for _ in self.datasets]
        
        current_batch = None
        current_batch_locations: List[Tuple[int, int]] = []
        buffer: List[Tuple[dict, int, int]] = []
        deferred_locations = DeferredSampleLocations(
            [], dataset_count=len(self.datasets)
        )
        
        # Resume handling
        if self._strict_resume and self._resume_loaded and worker_key not in self._resume_states:
            raise RuntimeError(
                f"No restored state for {worker_key}; refusing non-continuous resume"
            )
        if worker_key in self._resume_states:
            saved = self._resume_states.pop(worker_key)
            try:
                ws = WorkerState.from_dict(saved)
                
                if is_main_log:
                    logger.info(f'[{worker_key}] Resuming...')
                
                for ds_idx, iter_state in enumerate(ws.iterator_states):
                    iterators[ds_idx] = DeterministicIterator.from_state_dict(
                        self.datasets[ds_idx], iter_state
                    )
                
                sample_rng.setstate(ws.sample_rng_state)
                samples_produced = ws.samples_produced
                batches_produced = ws.batches_produced
                dataset_sampler_draws = list(ws.dataset_sampler_draws)
                deferred_locations = DeferredSampleLocations(
                    ws.deferred_locations,
                    dataset_count=len(self.datasets),
                    iterator_states=ws.iterator_states,
                )
                
                if ws.current_batch_locations:
                    if is_main_log:
                        logger.info(f'[{worker_key}] Rebuilding current_batch ({len(ws.current_batch_locations)} samples)...')
                    for loc in ws.current_batch_locations:
                        ds_idx, global_idx = loc
                        try:
                            sample = self.datasets[ds_idx].get_sample_at_global_idx(global_idx, iterator_seeds[ds_idx])
                            if self._get_sample_length(sample) <= self.max_num_tokens_per_sample:
                                current_batch = self._merge_samples(current_batch, sample)
                                current_batch_locations.append(loc)
                        except Exception as e:
                            if self._strict_resume:
                                raise RuntimeError(
                                    f"[{worker_key}] failed to restore batch sample {loc}"
                                ) from e
                            logger.warning(f'[{worker_key}] Failed to restore batch sample {loc}: {e}')

                if ws.buffer_locations:
                    if is_main_log:
                        logger.info(f'[{worker_key}] Rebuilding buffer ({len(ws.buffer_locations)} samples)...')
                    for loc in ws.buffer_locations:
                        ds_idx, global_idx = loc
                        try:
                            sample = self.datasets[ds_idx].get_sample_at_global_idx(global_idx, iterator_seeds[ds_idx])
                            if self._get_sample_length(sample) <= self.max_num_tokens_per_sample:
                                buffer.append((sample, ds_idx, global_idx))
                        except Exception as e:
                            if self._strict_resume:
                                raise RuntimeError(
                                    f"[{worker_key}] failed to restore buffer sample {loc}"
                                ) from e
                            logger.warning(f'[{worker_key}] Failed to restore buffer sample {loc}: {e}')
                
                if is_main_log:
                    logger.info(
                        f'[{worker_key}] Resume complete. Buffer size: {len(buffer)}, '
                        f'deferred samples: {len(deferred_locations)}'
                    )
                    
            except Exception as e:
                logger.error(f'[{worker_key}] Failed to resume: {e}')
                traceback.print_exc()
                if self._strict_resume:
                    raise RuntimeError(
                        f"[{worker_key}] strict dataloader resume failed"
                    ) from e
                current_batch = None
                current_batch_locations = []
                buffer = []

        # Helper functions
        def build_state_snapshot() -> dict:
            return WorkerState(
                iterator_states=[it.state_dict() for it in iterators],
                sample_rng_state=sample_rng.getstate(),
                samples_produced=samples_produced,
                batches_produced=batches_produced,
                dataset_sampler_draws=list(dataset_sampler_draws),
                current_batch_locations=list(current_batch_locations),
                buffer_locations=[(b[1], b[2]) for b in buffer],
                deferred_locations=deferred_locations.to_list(),
            ).to_dict()

        def fetch_next_sample() -> Tuple[dict, int, int]:
            nonlocal samples_produced, skipped_count
            while True:
                ds_idx = sample_rng.choices(range(len(self.datasets)), weights=self.dataset_weight)[0]
                dataset_sampler_draws[ds_idx] += 1
                try:
                    deferred_location = deferred_locations.pop_for_dataset(ds_idx)
                    if deferred_location is not None:
                        _, global_idx = deferred_location
                        sample = self.datasets[ds_idx].get_sample_at_global_idx(
                            global_idx, iterator_seeds[ds_idx]
                        )
                        if self._get_sample_length(sample) > self.max_num_tokens_per_sample:
                            raise RuntimeError(
                                f"[{worker_key}] restored deferred sample became too long: "
                                f"location={deferred_location}"
                            )
                        return sample, ds_idx, global_idx
                    sample, global_idx = next(iterators[ds_idx])
                    samples_produced += 1
                    if self._get_sample_length(sample) > self.max_num_tokens_per_sample:
                        skipped_count += 1
                        continue
                    return sample, ds_idx, global_idx
                except StopIteration:
                    continue

        # Main loop
        while True:
            # Try to fill current_batch from buffer (Best-Fit Strategy)
            current_len = self._get_sample_length(current_batch)
            remaining_space = self.max_num_tokens - current_len
            
            best_fit_idx = -1
            max_fit_len = -1
            
            for i, (buf_sample, _, _) in enumerate(buffer):
                s_len = self._get_sample_length(buf_sample)
                if s_len <= remaining_space:
                    if s_len > max_fit_len:
                        max_fit_len = s_len
                        best_fit_idx = i
            
            if best_fit_idx != -1:
                sample, ds_idx, global_idx = buffer.pop(best_fit_idx)
                current_batch = self._merge_samples(current_batch, sample)
                current_batch_locations.append((ds_idx, global_idx))
                continue

            if len(buffer) < self.buffer_size:
                new_sample, ds_idx, global_idx = fetch_next_sample()
                new_len = self._get_sample_length(new_sample)
                
                if new_len <= remaining_space:
                    current_batch = self._merge_samples(current_batch, new_sample)
                    current_batch_locations.append((ds_idx, global_idx))
                    continue
                else:
                    buffer.append((new_sample, ds_idx, global_idx))
            
            # Yield Logic
            if current_batch is not None:
                batches_produced += 1
                output_batch = self._finalize_batch(current_batch)
                
                current_batch = None
                current_batch_locations = []
                
                # Start new batch with largest sample from buffer (Big Rocks First)
                if len(buffer) > 0:
                    buffer.sort(key=lambda x: self._get_sample_length(x[0]), reverse=True)
                    sample, ds_idx, global_idx = buffer.pop(0)
                    current_batch = self._merge_samples(None, sample)
                    current_batch_locations = [(ds_idx, global_idx)]
                
                state_snapshot = build_state_snapshot()
                
                output_batch['_worker_key'] = worker_key
                output_batch['_batch_idx'] = batches_produced
                output_batch['_state_snapshot'] = state_snapshot
                
                yield output_batch
                
                if is_main_log and batches_produced % self.log_freq == 0:
                    packing_efficiency = current_len / self.max_num_tokens * 100
                    logger.info(f'batches={batches_produced}, samples={samples_produced}, '
                               f'buffer_len={len(buffer)}, packing_eff={packing_efficiency:.1f}%')
            else:
                if len(buffer) == 0:
                    continue
                else:
                    buffer.sort(key=lambda x: self._get_sample_length(x[0]), reverse=True)
                    sample, ds_idx, global_idx = buffer.pop(0)
                    current_batch = self._merge_samples(None, sample)
                    current_batch_locations = [(ds_idx, global_idx)]


def packed_collate_fn_mtp(features: List[dict], dataset: Optional[StreamPackedDatasetMTP] = None) -> dict:
    """Collator for MTP packing: processes batch and preserves state metadata."""
    assert len(features) == 1, f"Expected batch_size=1 for packing, got {len(features)}"
    
    feat = features[0]
    input_len = int(feat['input_ids'].shape[0])
    label_len = int(feat['labels'].shape[0])
    pos_len = int(feat['position_ids'].shape[-1])
    sub_sample_lengths = feat['sub_sample_lengths']
    packed_len = int(sub_sample_lengths.sum().item()) if isinstance(sub_sample_lengths, torch.Tensor) else int(sum(sub_sample_lengths))
    non_ignore_labels = int(feat['labels'][1:].ne(IGNORE_TOKEN_ID).sum().item()) if feat['labels'].numel() > 1 else 0

    if not (input_len == label_len == pos_len == packed_len):
        raise ValueError(
            f"Packed feature length mismatch: input_ids={input_len}, labels={label_len}, "
            f"position_ids={pos_len}, sub_sample_lengths_sum={packed_len}"
        )
    if non_ignore_labels == 0:
        raise ValueError(
            f"Packed feature has no valid shifted labels: input_ids={input_len}, "
            f"labels_non_ignore_after_shift={non_ignore_labels}, "
            f"sub_sample_lengths={sub_sample_lengths.tolist() if isinstance(sub_sample_lengths, torch.Tensor) else sub_sample_lengths}"
        )
    
    worker_key = feat.get('_worker_key', None)
    state_snapshot = feat.get('_state_snapshot', None)
    
    image_flags = feat['image_flags']
    if not isinstance(image_flags, torch.Tensor):
        image_flags = torch.tensor(image_flags)

    pos = feat['position_ids'].unsqueeze(0)  # [L] -> [1, L]

    result = dict(
        input_ids=feat['input_ids'].unsqueeze(0),
        labels=feat['labels'].unsqueeze(0),
        attention_mask=None,
        position_ids=pos,
        pixel_values=feat['pixel_values'],
        image_flags=image_flags,
        sub_sample_lengths=[sub_sample_lengths],
    )

    for key in ("relation_family", "defect_type", "target_boxes", "target_box_mask"):
        if key in feat:
            result[key] = feat[key]

    if 'image_grid_hws' in feat:
        grid = feat['image_grid_hws']
        if isinstance(grid, np.ndarray):
            grid = torch.from_numpy(grid)
        result['image_grid_hws'] = grid
    
    if worker_key is not None:
        result['_worker_key'] = worker_key
    if state_snapshot is not None:
        result['_state_snapshot'] = state_snapshot
        
    return result


class PackedCollatorMTP:
    """Pickle-able collator class for MTP packing."""
    
    def __init__(self, pad_id: int = 0, dataset: StreamPackedDatasetMTP = None):
        self.pad_id = pad_id
        self.dataset = dataset
    
    def __call__(self, features):
        return packed_collate_fn_mtp(features, dataset=self.dataset)


class StateAwareDataLoader:
    """Wrapper around DataLoader to capture state snapshots from worker processes."""
    def __init__(self, dataloader, dataset: StreamPackedDatasetMTP):
        self.dataloader = dataloader
        self.dataset = dataset

    def __iter__(self):
        for batch in self.dataloader:
            if '_worker_key' in batch and '_state_snapshot' in batch:
                worker_key = batch.pop('_worker_key')
                state_snapshot = batch.pop('_state_snapshot')
                
                if self.dataset is not None:
                    # Worker-side DataLoader prefetch may already have built
                    # later batches.  Advance the durable main-process state
                    # only when this wrapper actually hands a batch to Trainer;
                    # unseen prefetched batches must be regenerated on resume.
                    self.dataset._worker_states[worker_key] = state_snapshot

            batch.pop('_batch_idx', None)
            
            yield batch

    def __len__(self):
        return len(self.dataloader)


class DataloaderStateCallback(TrainerCallback):
    """Callback to save dataloader state."""
    
    def __init__(self, train_dataset: StreamPackedDatasetMTP):
        self.train_dataset = train_dataset
    
    def on_save(self, args, state, control, **kwargs):
        if not hasattr(self.train_dataset, 'state_dict'):
            return control
        self.train_dataset._dataloader_checkpoint_error = None
        
        checkpoint_folder = f"checkpoint-{state.global_step}"
        output_dir = os.path.join(args.output_dir, checkpoint_folder)
        rank = get_rank()
        state_path = os.path.join(output_dir, f"dataloader_state_rank{rank}.pt")

        try:
            ds_state = self.train_dataset.state_dict(
                completed_global_step=int(state.global_step)
            )
            
            if rank == 0:
                total_batches = sum(
                    ws.get('batches_produced', 0) 
                    for ws in ds_state.get('worker_states', {}).values()
                )
                total_samples = sum(
                    ws.get('samples_produced', 0) 
                    for ws in ds_state.get('worker_states', {}).values()
                )
                logger.info(f"Saving dataloader state: total_batches={total_batches}")
            
            atomic_save_with_fsync(torch.save, ds_state, state_path)
            
            if rank == 0:
                logger.info(f"Saved dataloader state to {state_path}")
                
        except Exception as e:
            logger.error(f"Rank {rank}: Failed to save dataloader state: {e}")
            traceback.print_exc()
            # Do not raise before the completion callback's distributed
            # collective: another rank could already be waiting there forever.
            # Completion gathers these local errors and raises synchronously on
            # every rank before it can publish checkpoint_complete.json.
            self.train_dataset._dataloader_checkpoint_error = (
                f"rank {rank}: {type(e).__name__}: {e}"
            )
        return control


class SamplingCoverageCallback(TrainerCallback):
    """Persist auditable all-record sampling coverage every 1,000 steps."""

    def __init__(self, train_dataset: StreamPackedDatasetMTP, interval: int = 1000):
        self.train_dataset = train_dataset
        self.interval = int(interval)
        self._written_steps = set()

    def _local_seen(self) -> list:
        seen = [set() for _ in self.train_dataset.datasets]
        samples_drawn = [0 for _ in self.train_dataset.datasets]
        for worker in self.train_dataset._worker_states.values():
            for ds_index, iterator_state in enumerate(worker.get("iterator_states", [])):
                if ds_index >= len(seen):
                    continue
                global_idx = int(iterator_state.get("global_idx", 0))
                seed = int(iterator_state.get("seed", 0))
                samples_drawn[ds_index] += global_idx
                seen[ds_index].update(
                    self.train_dataset.datasets[ds_index].seen_raw_indices(seed, global_idx)
                )
        return [
            {"seen": sorted(values), "samples_drawn": samples_drawn[index]}
            for index, values in enumerate(seen)
        ]

    @staticmethod
    def _is_monotonic(previous: dict, current: dict) -> bool:
        return is_monotonic_coverage(previous, current)

    def _write(self, args, state, *, resume_start: bool = False) -> None:
        step = int(state.global_step)
        write_key = (step, "resume_start" if resume_start else "periodic")
        if write_key in self._written_steps:
            return
        local = self._local_seen()
        gathered = None
        if dist.is_available() and dist.is_initialized():
            if get_rank() == 0:
                gathered = [None] * dist.get_world_size()
            dist.gather_object(local, gathered, dst=0)
        else:
            gathered = [local]
        if get_rank() != 0:
            self._written_steps.add(write_key)
            return
        datasets = []
        for ds_index, dataset in enumerate(self.train_dataset.datasets):
            union_seen = set()
            samples_drawn = 0
            for rank_payload in gathered:
                item = rank_payload[ds_index]
                union_seen.update(item["seen"])
                samples_drawn += int(item["samples_drawn"])
            inventory = dataset.sampling_inventory(union_seen)
            inventory["samples_drawn_with_repetition"] = samples_drawn
            inventory["repeated_draws"] = max(
                0, samples_drawn - inventory["seen_unique_records"]
            )
            datasets.append(inventory)
        payload = {
            "schema_version": 1,
            "global_step": step,
            "event": "resume_start" if resume_start else "periodic_coverage",
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "datasets": datasets,
        }
        output_dir = Path(args.output_dir) / "diagnostics"
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / (
            f"sampling_coverage_resume_start_step_{step}.json"
            if resume_start
            else f"sampling_coverage_step_{step}.json"
        )
        if not write_sampling_coverage_atomic(destination, payload):
            logger.warning(
                "[UI5 sampling] refusing non-monotonic overwrite: %s",
                destination,
            )
            self._written_steps.add(write_key)
            return
        logger.info("[UI5 sampling] wrote %s: %s", destination, datasets)
        self._written_steps.add(write_key)

    def on_train_begin(self, args, state, control, **kwargs):
        self._write(args, state, resume_start=int(state.global_step) > 0)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step and state.global_step % self.interval == 0:
            self._write(args, state)
        return control


class CheckpointCompletionCallback(TrainerCallback):
    """Validate every rank's resume state before declaring a save complete."""

    def __init__(
        self,
        train_dataset: StreamPackedDatasetMTP,
        *,
        model_args=None,
        data_args=None,
    ):
        self.train_dataset = train_dataset
        self.model_args = model_args
        self.data_args = data_args
        self._segment_source_global_step: Optional[int] = None
        self._segment_target_global_step: Optional[int] = None

    def set_segment_bounds(self, *, source_global_step: int, target_global_step: int):
        source = int(source_global_step)
        target = int(target_global_step)
        if source < 0 or target <= source:
            raise ValueError(
                "Invalid checkpoint segment bounds: "
                f"source={source}, target={target}"
            )
        self._segment_source_global_step = source
        self._segment_target_global_step = target

    def _write_continuity_manifest(self, checkpoint_dir, args, state) -> dict:
        schedule = getattr(self.train_dataset, "curriculum_schedule", None)
        stream_config = self.train_dataset._stream_resume_config()
        stream_digest = hashlib.sha256(
            json.dumps(
                stream_config,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        fp16 = bool(getattr(args, "fp16", False))
        bf16 = bool(getattr(args, "bf16", False))
        deepspeed_enabled = bool(getattr(args, "deepspeed", None))
        segment_source = (
            self._segment_source_global_step
            if self._segment_source_global_step is not None
            else int(os.environ.get("CURRICULUM_START_STEP", "0"))
        )
        segment_target = (
            self._segment_target_global_step
            if self._segment_target_global_step is not None
            else int(
                os.environ.get(
                    "LOCANY_STOP_AFTER_STEP", getattr(args, "max_steps", 0)
                )
            )
        )
        optimizer_config = (
            training_continuity_config(
                args,
                schedule,
                model_args=self.model_args,
                data_args=self.data_args,
            )
            if schedule is not None
            else None
        )
        optimizer_config_digest = (
            hashlib.sha256(
                json.dumps(
                    optimizer_config,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if optimizer_config is not None
            else None
        )
        payload = {
            "schema_version": 1,
            "global_step": int(state.global_step),
            "source_global_step": int(segment_source),
            "segment_target_global_step": segment_target,
            "target_total_steps": (
                int(schedule.total_steps)
                if schedule is not None
                else int(getattr(args, "max_steps", 0))
            ),
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "precision": "fp16" if fp16 else ("bf16" if bf16 else "fp32"),
            "gradient_scaler": {
                "applicable": fp16,
                "storage": (
                    "deepspeed_optimizer_state"
                    if fp16 and deepspeed_enabled
                    else ("scaler.pt" if fp16 else "not_applicable")
                ),
            },
            "optimizer_state": (
                "deepspeed_optimizer_state" if deepspeed_enabled else "optimizer.pt"
            ),
            "scheduler_state": (
                "deepspeed_model_state" if deepspeed_enabled else "scheduler.pt"
            ),
            "rng_state_pattern": "rng_state*.pth",
            "cuda_rng_required": bool(torch.cuda.is_available()),
            "dataloader_state_pattern": "dataloader_state_rank*.pt",
            "dataloader_state_version": 8,
            "curriculum_mode": "scheduled" if schedule is not None else "none",
            "curriculum_schedule_fingerprint": (
                schedule.fingerprint if schedule is not None else None
            ),
            "stream_resume_config_digest": stream_digest,
            "training_continuity_config": optimizer_config,
            "training_continuity_config_digest": optimizer_config_digest,
        }
        manifest = osp.join(checkpoint_dir, "continuity_state.json")
        temporary = f"{manifest}.tmp-{os.getpid()}"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, manifest)
        finally:
            if osp.exists(temporary):
                os.remove(temporary)
        return payload

    def on_save(self, args, state, control, **kwargs):
        rank = get_rank()
        checkpoint_dir = osp.join(args.output_dir, f"checkpoint-{state.global_step}")
        validation_error = None
        local_state_error = getattr(
            self.train_dataset, "_dataloader_checkpoint_error", None
        )
        if dist.is_available() and dist.is_initialized():
            state_errors = [None] * dist.get_world_size()
            dist.all_gather_object(state_errors, local_state_error)
        else:
            state_errors = [local_state_error]
        if rank == 0:
            failed_rank_states = [error for error in state_errors if error is not None]
            if failed_rank_states:
                validation_error = (
                    "Dataloader checkpoint state failed on one or more ranks: "
                    + "; ".join(failed_rank_states)
                )
            else:
                temporary = None
                try:
                    continuity = self._write_continuity_manifest(
                        checkpoint_dir, args, state
                    )
                    strict = continuity["curriculum_mode"] == "scheduled"
                    report = validate_checkpoint(
                        Path(checkpoint_dir),
                        mode="resume",
                        expected_ranks=(
                            dist.get_world_size() if dist.is_initialized() else 1
                        ),
                        strict=strict,
                        scaler_required=bool(getattr(args, "fp16", False)),
                        expected_curriculum_fingerprint=continuity[
                            "curriculum_schedule_fingerprint"
                        ],
                        require_completion_marker=False,
                    )
                    if not report["valid"]:
                        validation_error = (
                            "Checkpoint save returned but resume validation failed: "
                            f"checkpoint={checkpoint_dir}; "
                            f"errors={'; '.join(report['errors'])}"
                        )
                    else:
                        marker = osp.join(
                            checkpoint_dir, "checkpoint_complete.json"
                        )
                        temporary = f"{marker}.tmp-{os.getpid()}"
                        payload = {
                            "schema_version": 1,
                            "global_step": int(state.global_step),
                            "completed_at_unix": time.time(),
                            "hostname": socket.gethostname(),
                            "world_size": (
                                dist.get_world_size()
                                if dist.is_initialized()
                                else 1
                            ),
                            "validation": report,
                        }
                        with open(temporary, "w", encoding="utf-8") as handle:
                            json.dump(
                                payload,
                                handle,
                                ensure_ascii=False,
                                indent=2,
                                sort_keys=True,
                            )
                            handle.write("\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(temporary, marker)
                        logger.info(
                            "[Checkpoint] COMPLETE step=%s path=%s details=%s "
                            "warnings=%s",
                            state.global_step,
                            checkpoint_dir,
                            report["details"],
                            report["warnings"],
                        )
                except BaseException as exc:
                    validation_error = (
                        "Checkpoint completion validation/marker failed: "
                        f"checkpoint={checkpoint_dir}; {type(exc).__name__}: {exc}"
                    )
                    logger.exception("[Checkpoint] completion FAILED")
                finally:
                    if temporary and osp.exists(temporary):
                        try:
                            os.remove(temporary)
                        except OSError:
                            logger.exception(
                                "[Checkpoint] could not remove marker temporary file %s",
                                temporary,
                            )
        if dist.is_available() and dist.is_initialized():
            message = [validation_error]
            dist.broadcast_object_list(message, src=0)
            validation_error = message[0]
        if validation_error is not None:
            raise RuntimeError(validation_error)
        return control


class SegmentStopCallback(TrainerCallback):
    """Stop at an absolute global step and force a fully resumable checkpoint."""

    def __init__(self, stop_after_step: int):
        super().__init__()
        if stop_after_step <= 0:
            raise ValueError("stop_after_step must be positive")
        self.stop_after_step = stop_after_step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_step:
            control.should_save = True
            control.should_training_stop = True
            if get_rank() == 0:
                logger.info(
                    "Segment boundary reached at global_step=%s; forcing a resumable checkpoint",
                    state.global_step,
                )
        return control


class StreamPackingMTPTrainer(Trainer):
    """Trainer with StreamPackedDatasetMTP support."""

    _UI5_SCALARS = (
        "loss_total",
        "loss_lm",
        "loss_gate",
        "loss_image_gate",
        "loss_slot_gate",
        "loss_attention",
        "weighted_gate_loss",
        "weighted_slot_gate_loss",
        "weighted_attention_loss",
        "loss_lm_contribution",
        "loss_image_gate_contribution",
        "loss_slot_gate_contribution",
        "loss_attention_contribution",
        "loss_reconstructed",
        "loss_reconstruction_error",
        "attention_active_batch_rate",
        "grad_norm",
        "detail_layer5_norm",
        "detail_layer15_norm",
        "detail_layer26_norm",
        "detail_layer5_abs_max",
        "detail_layer15_abs_max",
        "detail_layer26_abs_max",
        "detail_layer5_saturation_fraction",
        "detail_layer15_saturation_fraction",
        "detail_layer26_saturation_fraction",
        "detail_norm_ratio",
        "detail_fused_norm",
        "relation_context_norm",
        "relation_gate_prob_mean",
        "pbd_delta_norm",
        "pbd_active_positions",
        "relation_grad_norm",
        "image_gate_grad_norm",
        "slot_gate_grad_norm",
        "pbd_grad_norm",
        "relation_grad_seen_steps",
        "image_gate_grad_seen_steps",
        "slot_gate_grad_seen_steps",
        "pbd_grad_seen_steps",
        "relation_absolute_update_norm",
        "relation_relative_update_norm",
        "relation_changed_element_count",
        "image_gate_absolute_update_norm",
        "image_gate_relative_update_norm",
        "image_gate_changed_element_count",
        "slot_gate_absolute_update_norm",
        "slot_gate_relative_update_norm",
        "slot_gate_changed_element_count",
        "pbd_absolute_update_norm",
        "pbd_relative_update_norm",
        "pbd_changed_element_count",
    )
    _DEFECT_TO_DIAGNOSTIC_TASK = {
        0: "text_overflow",
        1: "element_cropping",
        2: "element_overlap",
        3: "text_ellipsis",
        4: "content_missing",
    }

    def __init__(
        self,
        *args,
        sample_log_interval: int = 100,
        max_num_tokens: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # CPT explicitly disables the optional UI Relation/Gate/PBD path.  In
        # that mode the modules do not exist and no UI update is expected, so
        # UI5-only audits must not turn a healthy CPT run into a failure.
        self._ui5_enabled = bool(
            getattr(self.model, "enable_ui_relation", False)
        )
        self._total_samples = 0
        self._sample_log_interval = sample_log_interval
        self._max_num_tokens = int(max_num_tokens)
        self._start_step = None  # 记录开始的step，用于resume时正确计算平均值
        self._ui5_excel = UI5ExcelLogger(
            osp.join(
                self.args.output_dir,
                "diagnostics",
                "ui5_training_evaluation.xlsx",
            )
        )
        self._ui5_global_epoch_offset = self._ui5_excel.latest_train_global_epoch()
        self._ui5_segment_start_epoch = None
        self._ui5_last_flushed_step = 0
        self._reset_ui5_window()
        self._ui5_window_path = osp.join(
            self.args.output_dir,
            "diagnostics",
            f".ui5_train_window_rank{get_rank()}.json",
        )
        if self._ui5_enabled:
            self._load_ui5_window_state()
        self._ui5_hook_squares = {}
        self._ui5_hook_seen = set()
        self._ui5_hook_handles = []
        self._ui5_last_grad_seen_global = {}
        self._ui5_parameter_baseline = {}
        self._ui5_real_data_audit_logged = False
        if self._ui5_enabled:
            self._register_ui5_gradient_hooks()
            self._snapshot_ui5_parameters()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Create one absolute-step LR schedule shared by all eval segments."""

        schedule = getattr(self.train_dataset, "curriculum_schedule", None)
        if schedule is None:
            return super().create_scheduler(
                num_training_steps=num_training_steps,
                optimizer=optimizer,
            )
        if self.lr_scheduler is not None:
            return self.lr_scheduler
        reference_lr = float(schedule.stages[0].llm_lr)
        configured_lr = float(self.args.learning_rate)
        if not math.isclose(
            configured_lr, reference_lr, rel_tol=1.0e-12, abs_tol=0.0
        ):
            raise ValueError(
                "Scheduled curriculum requires --learning_rate to equal the first "
                f"LLM_LRS value: learning_rate={configured_lr}, expected={reference_lr}"
            )
        target_optimizer = optimizer if optimizer is not None else self.optimizer
        if target_optimizer is None:
            raise RuntimeError("optimizer must exist before the curriculum scheduler")

        def lr_lambda(completed_optimizer_steps: int) -> float:
            return schedule.lr_multiplier_for_completed_steps(
                int(completed_optimizer_steps), reference_lr=reference_lr
            )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            target_optimizer,
            lr_lambda=lr_lambda,
        )
        if self.is_world_process_zero():
            logger.warning(
                "[UI5 curriculum LR] full_horizon=%s segment_target=%s "
                "boundaries=%s",
                schedule.total_steps,
                num_training_steps,
                [
                    {
                        "steps": [stage.first_optimizer_step, stage.last_optimizer_step],
                        "llm_lr": stage.llm_lr,
                    }
                    for stage in schedule.stages
                ],
            )
        return self.lr_scheduler

    def _checkpoint_trace(self, output_dir, stage, event, **details):
        """Fsync a compact breadcrumb before/after every checkpoint phase."""

        if get_rank() != 0:
            return
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        record = {
            "time_unix": time.time(),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "rank": get_rank(),
            "global_step": int(getattr(self.state, "global_step", 0)),
            "stage": stage,
            "event": event,
            **details,
        }
        trace_path = osp.join(output_dir, "checkpoint_save_trace.jsonl")
        with open(trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        logger.info(
            "[Checkpoint] %s %s step=%s path=%s details=%s",
            stage,
            event,
            record["global_step"],
            output_dir,
            details,
        )

    def _save(self, output_dir=None, state_dict=None):
        """Trace model/processor and the exact ``training_args.bin`` write."""

        output_dir = output_dir or self.args.output_dir
        self._checkpoint_trace(output_dir, "model_and_training_args", "START")
        original_torch_save = torch.save

        def traced_torch_save(obj, destination, *args, **kwargs):
            target = getattr(destination, "name", destination)
            destination_is_path = isinstance(destination, (str, os.PathLike))
            try:
                target_text = os.fspath(target)
            except TypeError:
                target_text = repr(target)
            stage = (
                "training_args.bin"
                if osp.basename(target_text) == "training_args.bin"
                else "torch.save"
            )
            self._checkpoint_trace(
                output_dir,
                stage,
                "START",
                target=target_text,
                object_type=f"{type(obj).__module__}.{type(obj).__name__}",
            )
            temporary_target = None
            try:
                if stage == "training_args.bin" and destination_is_path:
                    result = atomic_save_with_fsync(
                        original_torch_save, obj, target_text, *args, **kwargs
                    )
                else:
                    result = original_torch_save(obj, destination, *args, **kwargs)
            except BaseException as exc:
                self._checkpoint_trace(
                    output_dir,
                    stage,
                    "FAILED",
                    target=target_text,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            finally:
                if temporary_target and osp.exists(temporary_target):
                    try:
                        os.remove(temporary_target)
                    except OSError:
                        logger.exception(
                            "[Checkpoint] could not remove temporary training args %s",
                            temporary_target,
                        )
            size = None
            try:
                size = osp.getsize(target_text)
            except (OSError, TypeError):
                pass
            self._checkpoint_trace(
                output_dir,
                stage,
                "DONE",
                target=target_text,
                size_bytes=size,
            )
            return result

        torch.save = traced_torch_save
        try:
            result = super()._save(output_dir=output_dir, state_dict=state_dict)
        except BaseException as exc:
            self._checkpoint_trace(
                output_dir,
                "model_and_training_args",
                "FAILED",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            torch.save = original_torch_save
        self._checkpoint_trace(output_dir, "model_and_training_args", "DONE")
        return result

    def _save_optimizer_and_scheduler(self, output_dir):
        self._checkpoint_trace(output_dir, "optimizer_and_scheduler", "START")
        try:
            result = super()._save_optimizer_and_scheduler(output_dir)
        except BaseException as exc:
            self._checkpoint_trace(
                output_dir,
                "optimizer_and_scheduler",
                "FAILED",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._checkpoint_trace(output_dir, "optimizer_and_scheduler", "DONE")
        return result

    def _save_rng_state(self, output_dir):
        self._checkpoint_trace(output_dir, "rng_state", "START")
        try:
            result = super()._save_rng_state(output_dir)
        except BaseException as exc:
            self._checkpoint_trace(
                output_dir,
                "rng_state",
                "FAILED",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self._checkpoint_trace(output_dir, "rng_state", "DONE")
        return result

    def _save_checkpoint(self, model, trial, *args, **kwargs):
        run_dir = self._get_output_dir(trial=trial)
        checkpoint_dir = osp.join(run_dir, f"checkpoint-{self.state.global_step}")
        self._checkpoint_trace(checkpoint_dir, "trainer_checkpoint", "START")
        state_object = self.state
        original_state_save = state_object.save_to_json

        def traced_state_save(target):
            self._checkpoint_trace(
                checkpoint_dir,
                "trainer_state.json",
                "START",
                target=os.fspath(target),
            )
            try:
                result = original_state_save(target)
            except BaseException as exc:
                self._checkpoint_trace(
                    checkpoint_dir,
                    "trainer_state.json",
                    "FAILED",
                    target=os.fspath(target),
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            self._checkpoint_trace(
                checkpoint_dir,
                "trainer_state.json",
                "DONE",
                target=os.fspath(target),
                size_bytes=(osp.getsize(target) if osp.isfile(target) else None),
            )
            return result

        state_object.save_to_json = traced_state_save
        try:
            result = super()._save_checkpoint(model, trial, *args, **kwargs)
        except BaseException as exc:
            self._checkpoint_trace(
                checkpoint_dir,
                "trainer_checkpoint",
                "FAILED",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            state_object.save_to_json = original_state_save
        self._checkpoint_trace(checkpoint_dir, "trainer_checkpoint", "DONE")
        return result

    def _reset_ui5_window(self):
        self._ui5_scalar = {
            name: {"sum": 0.0, "count": 0.0, "min": float("inf"), "max": float("-inf")}
            for name in self._UI5_SCALARS
        }
        self._ui5_tasks = {
            task: defaultdict(float) for task in TRAIN_TASKS
        }
        for values in self._ui5_tasks.values():
            # Keep the raw image-Gate observations for this optimizer-step
            # window.  PR-AUC/AP must be computed once over the complete
            # 100-step, all-rank population; averaging per-micro-batch AP is
            # not the same metric and is especially misleading for rare UI
            # defects.
            values["pr_scores"] = []
            values["pr_labels"] = []
        self._ui5_peak_gpu_memory_mb = 0.0

    def _load_ui5_window_state(self):
        if not osp.isfile(self._ui5_window_path):
            return
        try:
            with open(self._ui5_window_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            for name in self._UI5_SCALARS:
                if name in state.get("scalars", {}):
                    self._ui5_scalar[name].update(state["scalars"][name])
            for task in TRAIN_TASKS:
                self._ui5_tasks[task].update(state.get("tasks", {}).get(task, {}))
            self._ui5_peak_gpu_memory_mb = float(state.get("peak_gpu_memory_mb", 0.0))
            self._ui5_last_flushed_step = int(state.get("last_flushed_step", 0))
            logger.info(
                "[UI5Excel] restored partial rank window from %s",
                self._ui5_window_path,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to restore UI5 diagnostic window: {self._ui5_window_path}"
            ) from exc

    def _save_ui5_window_state(self):
        os.makedirs(osp.dirname(self._ui5_window_path), exist_ok=True)
        temporary = f"{self._ui5_window_path}.tmp-{os.getpid()}"
        state = {
            "schema_version": 1,
            "last_seen_step": int(self.state.global_step),
            "last_flushed_step": int(self._ui5_last_flushed_step),
            "scalars": self._ui5_scalar,
            "tasks": {
                task: dict(values) for task, values in self._ui5_tasks.items()
            },
            "peak_gpu_memory_mb": self._ui5_peak_gpu_memory_mb,
        }
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._ui5_window_path)
        finally:
            if osp.exists(temporary):
                os.remove(temporary)

    @staticmethod
    def _tensor_float(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            return float(value.detach().float().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _add_ui5_scalar(self, name, value):
        value = self._tensor_float(value)
        if value is None or not np.isfinite(value):
            return
        state = self._ui5_scalar[name]
        state["sum"] += value
        state["count"] += 1.0
        state["min"] = min(state["min"], value)
        state["max"] = max(state["max"], value)

    def _set_ui5_scalar(self, name, value):
        value = self._tensor_float(value)
        if value is None or not np.isfinite(value):
            return
        self._ui5_scalar[name] = {
            "sum": value,
            "count": 1.0,
            "min": value,
            "max": value,
        }

    @staticmethod
    def _ui5_parameter_group(name):
        if "relation_pbd" in name:
            return "pbd"
        if "relation_pyramid.image_gate_heads" in name:
            return "image_gate"
        if "relation_pyramid.gate_heads" in name:
            return "slot_gate"
        if "relation_pyramid" in name:
            return "relation"
        return None

    def _register_ui5_gradient_hooks(self):
        for name, parameter in self.model.named_parameters():
            group = self._ui5_parameter_group(name)
            if group is None or not parameter.requires_grad:
                continue

            def capture(gradient, group=group):
                square = gradient.detach().float().square().sum()
                previous = self._ui5_hook_squares.get(group)
                self._ui5_hook_squares[group] = square if previous is None else previous + square
                self._ui5_hook_seen.add(group)
                return gradient

            self._ui5_hook_handles.append(parameter.register_hook(capture))

    def _snapshot_ui5_parameters(self):
        tracked = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if self._ui5_parameter_group(name) is not None and parameter.requires_grad
        ]
        gather_context = nullcontext()
        if tracked and any(hasattr(parameter, "ds_id") for _, parameter in tracked):
            import deepspeed

            gather_context = deepspeed.zero.GatheredParameters(
                [parameter for _, parameter in tracked], modifier_rank=None
            )
        with gather_context:
            for name, parameter in tracked:
                self._ui5_parameter_baseline[name] = (
                    parameter.detach().float().cpu().clone()
                )

    def _capture_ui5_parameter_updates(self):
        sums = defaultdict(float)
        bases = defaultdict(float)
        changed = defaultdict(int)
        current_parameters = dict(self.model.named_parameters())
        tracked = [
            current_parameters[name]
            for name in self._ui5_parameter_baseline
            if name in current_parameters
        ]
        gather_context = nullcontext()
        if tracked and any(hasattr(parameter, "ds_id") for parameter in tracked):
            import deepspeed

            gather_context = deepspeed.zero.GatheredParameters(
                tracked, modifier_rank=None
            )
        with gather_context:
            for name, baseline in self._ui5_parameter_baseline.items():
                current_parameter = current_parameters.get(name)
                if current_parameter is None or current_parameter.numel() != baseline.numel():
                    continue
                current = current_parameter.detach().float().cpu().reshape_as(baseline)
                delta = current - baseline
                group = self._ui5_parameter_group(name)
                sums[group] += float(delta.square().sum().item())
                bases[group] += float(baseline.square().sum().item())
                changed[group] += int(delta.ne(0).sum().item())
        for group in ("relation", "image_gate", "slot_gate", "pbd"):
            absolute = sums[group] ** 0.5
            relative = absolute / (bases[group] ** 0.5 + 1.0e-12)
            self._set_ui5_scalar(f"{group}_absolute_update_norm", absolute)
            self._set_ui5_scalar(f"{group}_relative_update_norm", relative)
            self._set_ui5_scalar(f"{group}_changed_element_count", changed[group])

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if not self._ui5_enabled:
            return optimizer
        audited_parameters = optimizer_parameters(self.optimizer)
        optimizer_ids = {id(parameter) for parameter in audited_parameters}
        optimizer_ds_ids = {
            int(parameter.ds_id)
            for parameter in audited_parameters
            if hasattr(parameter, "ds_id")
        }
        def in_optimizer(parameter):
            return id(parameter) in optimizer_ids or (
                hasattr(parameter, "ds_id")
                and int(parameter.ds_id) in optimizer_ds_ids
            )
        report = {}
        missing = []
        frozen = []
        for group_name in ("relation", "image_gate", "slot_gate", "pbd"):
            parameters = [
                (name, parameter)
                for name, parameter in self.model.named_parameters()
                if self._ui5_parameter_group(name) == group_name
            ]
            trainable = [(name, parameter) for name, parameter in parameters if parameter.requires_grad]
            group_missing = [name for name, parameter in trainable if not in_optimizer(parameter)]
            group_frozen = [name for name, parameter in parameters if not parameter.requires_grad]
            report[group_name] = {
                "parameter_count": sum(parameter.numel() for _, parameter in parameters),
                "trainable_parameter_count": sum(parameter.numel() for _, parameter in trainable),
                "optimizer_parameter_count": sum(
                    parameter.numel() for _, parameter in trainable if in_optimizer(parameter)
                ),
                "missing_from_optimizer": group_missing,
                "unexpected_frozen_parameters": group_frozen,
            }
            missing.extend(group_missing)
            frozen.extend(group_frozen)
        logger.warning("[UI5 optimizer audit] %s", json.dumps(report, ensure_ascii=False))
        if self.is_world_process_zero():
            audit_path = osp.join(
                self.args.output_dir, "diagnostics", "ui_relation_optimizer_audit.json"
            )
            os.makedirs(osp.dirname(audit_path), exist_ok=True)
            temporary = f"{audit_path}.tmp-{os.getpid()}"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, audit_path)
        if missing:
            raise RuntimeError(f"Trainable UI parameters missing from optimizer: {missing}")
        if frozen:
            raise RuntimeError(f"Unexpected frozen UI parameters: {frozen}")
        return optimizer

    def _add_ui5_weighted_scalar(self, name, value, weight):
        value = self._tensor_float(value)
        weight = self._tensor_float(weight)
        if (
            value is None
            or weight is None
            or weight <= 0
            or not np.isfinite(value)
            or not np.isfinite(weight)
        ):
            return
        state = self._ui5_scalar[name]
        state["sum"] += value * weight
        state["count"] += weight
        state["min"] = min(state["min"], value)
        state["max"] = max(state["max"], value)

    def _capture_ui5_batch(self, outputs, inputs):
        if outputs is None:
            return
        config = self.model.config
        gate_weight = float(getattr(config, "relation_gate_loss_weight", 1.0))
        slot_gate_weight = float(
            getattr(config, "relation_slot_gate_loss_weight", 0.1)
        )
        attention_weight = float(
            getattr(config, "relation_attention_loss_weight", 0.1)
        )
        self._add_ui5_scalar("loss_total", getattr(outputs, "loss", None))
        self._add_ui5_scalar("loss_lm", getattr(outputs, "lm_loss", None))
        self._add_ui5_scalar("loss_gate", getattr(outputs, "gate_loss", None))
        self._add_ui5_scalar("loss_image_gate", getattr(outputs, "image_gate_loss", None))
        self._add_ui5_scalar("loss_slot_gate", getattr(outputs, "slot_gate_loss", None))
        self._add_ui5_scalar(
            "loss_attention", getattr(outputs, "attention_loss", None)
        )
        gate_loss_value = self._tensor_float(getattr(outputs, "gate_loss", None))
        if gate_loss_value is not None:
            self._add_ui5_scalar("weighted_gate_loss", gate_weight * gate_loss_value)
        else:
            self._add_ui5_scalar("weighted_gate_loss", 0.0)
        slot_gate_loss_value = self._tensor_float(getattr(outputs, "slot_gate_loss", None))
        self._add_ui5_scalar(
            "weighted_slot_gate_loss",
            slot_gate_weight * slot_gate_loss_value if slot_gate_loss_value is not None else 0.0,
        )
        attention_loss_value = self._tensor_float(
            getattr(outputs, "attention_loss", None)
        )
        if attention_loss_value is not None:
            self._add_ui5_scalar(
                "weighted_attention_loss",
                attention_weight * attention_loss_value,
            )
        else:
            self._add_ui5_scalar("weighted_attention_loss", 0.0)

        for output_name, metric_name in (
            ("loss_lm_contribution", "loss_lm_contribution"),
            ("loss_image_gate_contribution", "loss_image_gate_contribution"),
            ("loss_slot_gate_contribution", "loss_slot_gate_contribution"),
            ("loss_attention_contribution", "loss_attention_contribution"),
            ("loss_reconstructed", "loss_reconstructed"),
            ("loss_reconstruction_error", "loss_reconstruction_error"),
            ("attention_active", "attention_active_batch_rate"),
        ):
            self._add_ui5_scalar(metric_name, getattr(outputs, output_name, None))

        detail_norm = getattr(outputs, "detail_feature_norm", None)
        if torch.is_tensor(detail_norm) and detail_norm.numel() == 3:
            for name, value in zip(
                (
                    "detail_layer5_norm",
                    "detail_layer15_norm",
                    "detail_layer26_norm",
                ),
                detail_norm.reshape(-1),
            ):
                self._add_ui5_scalar(name, value)
            if not bool(torch.isfinite(detail_norm).all()):
                raise FloatingPointError("Detail Pyramid projected norm is non-finite")
            if bool((detail_norm < 1.0e-4).any()):
                raise RuntimeError(f"Detail Pyramid projected norm below 1e-4: {detail_norm.tolist()}")
        detail_abs_max = getattr(outputs, "detail_feature_abs_max", None)
        if torch.is_tensor(detail_abs_max) and detail_abs_max.numel() == 3:
            for name, value in zip(
                ("detail_layer5_abs_max", "detail_layer15_abs_max", "detail_layer26_abs_max"),
                detail_abs_max.reshape(-1),
            ):
                self._add_ui5_scalar(name, value)
        detail_saturation = getattr(outputs, "detail_saturation_fraction", None)
        if torch.is_tensor(detail_saturation) and detail_saturation.numel() == 3:
            for name, value in zip(
                (
                    "detail_layer5_saturation_fraction",
                    "detail_layer15_saturation_fraction",
                    "detail_layer26_saturation_fraction",
                ),
                detail_saturation.reshape(-1),
            ):
                self._add_ui5_scalar(name, value)
            if float(detail_saturation.detach().float().max().item()) > 0.001:
                raise RuntimeError(
                    f"Detail Pyramid saturation_fraction exceeds 0.001: {detail_saturation.tolist()}"
                )
        detail_norm_ratio = getattr(outputs, "detail_norm_ratio", None)
        self._add_ui5_scalar("detail_norm_ratio", detail_norm_ratio)
        ratio_value = self._tensor_float(detail_norm_ratio)
        if ratio_value is not None and ratio_value > 20.0:
            raise RuntimeError(f"Detail Pyramid norm ratio exceeds 20: {ratio_value}")
        for output_name, metric_name in (
            ("detail_fused_norm", "detail_fused_norm"),
            ("relation_context_norm", "relation_context_norm"),
            ("relation_gate_prob_mean", "relation_gate_prob_mean"),
        ):
            self._add_ui5_scalar(metric_name, getattr(outputs, output_name, None))
        pbd_active_positions = getattr(outputs, "pbd_active_positions", None)
        self._add_ui5_weighted_scalar(
            "pbd_delta_norm",
            getattr(outputs, "pbd_delta_norm", None),
            pbd_active_positions,
        )
        self._add_ui5_scalar("pbd_active_positions", pbd_active_positions)

        if not self._ui5_real_data_audit_logged and torch.is_tensor(detail_norm):
            detail_weights_for_audit = getattr(outputs, "detail_layer_weights", None)
            if torch.is_tensor(detail_weights_for_audit):
                weight_sums = detail_weights_for_audit.detach().float().sum(dim=-1)
                if not bool(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1.0e-5)):
                    raise RuntimeError(
                        f"Detail Pyramid scale weights do not sum to one: {weight_sums.tolist()}"
                    )
                if int(self.state.global_step) == 0 and not bool(
                    torch.allclose(
                        detail_weights_for_audit.detach().float(),
                        torch.full_like(detail_weights_for_audit.detach().float(), 1.0 / 3.0),
                        atol=1.0e-4,
                    )
                ):
                    raise RuntimeError(
                        "Initial Detail Pyramid scale weights are not thirds: "
                        f"{detail_weights_for_audit.detach().float().cpu().tolist()}"
                    )
            audit = {
                "rank": get_rank(),
                "projected_norm": detail_norm.detach().float().cpu().tolist(),
                "projected_abs_max": (
                    detail_abs_max.detach().float().cpu().tolist()
                    if torch.is_tensor(detail_abs_max)
                    else None
                ),
                "saturation_fraction": (
                    detail_saturation.detach().float().cpu().tolist()
                    if torch.is_tensor(detail_saturation)
                    else None
                ),
                "norm_ratio": self._tensor_float(detail_norm_ratio),
                "scale_weights": (
                    detail_weights_for_audit.detach().float().cpu().tolist()
                    if torch.is_tensor(detail_weights_for_audit)
                    else None
                ),
                "relation_context_norm": self._tensor_float(
                    getattr(outputs, "relation_context_norm", None)
                ),
                "pbd_delta_norm": self._tensor_float(
                    getattr(outputs, "pbd_delta_norm", None)
                ),
                "pbd_active_positions": self._tensor_float(pbd_active_positions),
            }
            logger.warning(
                "[UI5 first-real-batch audit] %s",
                json.dumps(audit, ensure_ascii=False),
            )
            audit_path = osp.join(
                self.args.output_dir,
                "diagnostics",
                f"first_real_batch_audit_rank{get_rank()}.json",
            )
            os.makedirs(osp.dirname(audit_path), exist_ok=True)
            temporary = f"{audit_path}.tmp-{os.getpid()}"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(audit, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, audit_path)
            self._ui5_real_data_audit_logged = True

        defect_type = inputs.get("defect_type")
        target_mask = inputs.get("target_box_mask")
        p_defect = getattr(outputs, "p_defect", None)
        if not (
            torch.is_tensor(defect_type)
            and torch.is_tensor(target_mask)
            and torch.is_tensor(p_defect)
        ):
            return
        defect_type = defect_type.detach().reshape(-1).long()
        positive = target_mask.detach().reshape(target_mask.shape[0], -1).any(dim=-1)
        probabilities = p_defect.detach().float().reshape(-1)
        count = min(defect_type.numel(), positive.numel(), probabilities.numel())
        defect_type = defect_type[:count]
        positive = positive[:count]
        probabilities = probabilities[:count]
        detail_weights = getattr(outputs, "detail_layer_weights", None)
        if torch.is_tensor(detail_weights) and detail_weights.shape[-1] == 3:
            detail_weights = detail_weights.detach().float().reshape(-1, 3)
            if detail_weights.shape[0] < count:
                count = detail_weights.shape[0]
                defect_type = defect_type[:count]
                positive = positive[:count]
                probabilities = probabilities[:count]
            detail_weights = detail_weights[:count]
        else:
            detail_weights = None
        threshold = float(getattr(config, "relation_gate_threshold", 0.5))
        predicted = probabilities >= threshold
        slot_targets = getattr(outputs, "gate_targets", None)
        for defect_id, task in self._DEFECT_TO_DIAGNOSTIC_TASK.items():
            mask = defect_type == defect_id
            values = self._ui5_tasks[task]
            values["samples"] += float(mask.sum().item())
            values["positive"] += float((mask & positive).sum().item())
            values["negative"] += float((mask & ~positive).sum().item())
            values["p_defect_pos_sum"] += float(
                probabilities[mask & positive].sum().item()
            )
            values["p_defect_pos_count"] += float((mask & positive).sum().item())
            values["p_defect_neg_sum"] += float(
                probabilities[mask & ~positive].sum().item()
            )
            values["p_defect_neg_count"] += float((mask & ~positive).sum().item())
            values["tp"] += float((mask & positive & predicted).sum().item())
            values["fp"] += float((mask & ~positive & predicted).sum().item())
            values["fn"] += float((mask & positive & ~predicted).sum().item())
            task_probability = probabilities[mask]
            task_positive = positive[mask]
            if task_probability.numel():
                values["pr_scores"].extend(
                    task_probability.detach().float().cpu().tolist()
                )
                values["pr_labels"].extend(
                    task_positive.detach().bool().cpu().tolist()
                )
            if torch.is_tensor(slot_targets) and slot_targets.shape[0] >= count:
                task_slot_targets = (
                    slot_targets.detach().float().reshape(slot_targets.shape[0], -1)[:count][mask]
                )
                values["slot_positive"] += float(task_slot_targets.sum().item())
                values["slot_negative"] += float(task_slot_targets.numel() - task_slot_targets.sum().item())
            if detail_weights is not None:
                task_count = float(mask.sum().item())
                if task_count:
                    task_weights = detail_weights[mask]
                    values["detail_weight_l5_sum"] += float(
                        task_weights[:, 0].sum().item()
                    )
                    values["detail_weight_l15_sum"] += float(
                        task_weights[:, 1].sum().item()
                    )
                    values["detail_weight_l26_sum"] += float(
                        task_weights[:, 2].sum().item()
                    )
                    values["detail_weight_count"] += task_count

        for output_name, prefix in (
            ("per_task_image_gate_loss", "gate_loss"),
            ("per_task_slot_gate_loss", "slot_gate_loss"),
            ("per_task_attention_loss", "attention_loss"),
        ):
            task_losses = getattr(outputs, output_name, None) or {}
            for raw_defect_id, loss_value in task_losses.items():
                task = self._DEFECT_TO_DIAGNOSTIC_TASK.get(int(raw_defect_id))
                loss_float = self._tensor_float(loss_value)
                if task is not None and loss_float is not None:
                    self._ui5_tasks[task][f"{prefix}_sum"] += loss_float
                    self._ui5_tasks[task][f"{prefix}_count"] += 1.0

    def _capture_ui5_gradient_groups(self, model):
        # Hooks run at gradient creation time, before DeepSpeed/MAGI can clear
        # parameter.grad. One seen-step means at least one parameter hook in
        # that group fired during this local-rank training_step.
        for group in ("relation", "image_gate", "slot_gate", "pbd"):
            square = self._ui5_hook_squares.get(group)
            if square is not None:
                self._add_ui5_scalar(
                    f"{group}_grad_norm", float(square.detach().item())
                )
            if group in self._ui5_hook_seen:
                global_step = int(self.state.global_step)
                if self._ui5_last_grad_seen_global.get(group) != global_step:
                    self._add_ui5_scalar(f"{group}_grad_seen_steps", 1.0)
                    self._ui5_last_grad_seen_global[group] = global_step
        self._ui5_hook_squares = {}
        self._ui5_hook_seen = set()

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        if self._ui5_enabled:
            self._capture_ui5_batch(outputs, inputs)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        # 记录开始的step（用于resume时正确计算平均值）
        if self._start_step is None:
            self._start_step = self.state.global_step
        if self._ui5_segment_start_epoch is None:
            self._ui5_segment_start_epoch = float(self.state.epoch or 0.0)
        
        # Count samples in current batch (通过 sub_sample_lengths 获取样本数)
        if 'sub_sample_lengths' in inputs:
            sub_sample_lengths = inputs['sub_sample_lengths']
            if isinstance(sub_sample_lengths, list):
                # sub_sample_lengths 是 list of tensors，每个tensor的长度是样本数
                num_samples = sum(len(ssl) for ssl in sub_sample_lengths)
            elif isinstance(sub_sample_lengths, torch.Tensor):
                num_samples = sub_sample_lengths.size(0)
            else:
                num_samples = 1
            self._total_samples += int(num_samples)
        
        # Log sample count every N steps (rank 0 only)
        if self.state.global_step > 0 and self.state.global_step % self._sample_log_interval == 0:
            if get_rank() == 0:
                steps_since_start = self.state.global_step - self._start_step
                if steps_since_start > 0:
                    avg_samples_per_step = self._total_samples / steps_since_start
                    logger.info(f"[SampleStats] Step {self.state.global_step}: "
                               f"Total samples (this run) = {self._total_samples}, "
                               f"Avg samples/step = {avg_samples_per_step:.2f}")
        
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self._ui5_enabled:
            self._capture_ui5_gradient_groups(model)
        if self._ui5_enabled and torch.cuda.is_available():
            self._ui5_peak_gpu_memory_mb = max(
                self._ui5_peak_gpu_memory_mb,
                torch.cuda.max_memory_allocated() / (1024.0 ** 2),
            )
        return loss

    def _reduce_ui5_window(self):
        scalar_names = list(self._UI5_SCALARS)
        task_keys = (
            "samples", "positive", "negative",
            "p_defect_pos_sum", "p_defect_pos_count",
            "p_defect_neg_sum", "p_defect_neg_count",
            "tp", "fp", "fn",
            "gate_loss_sum", "gate_loss_count",
            "slot_gate_loss_sum", "slot_gate_loss_count",
            "slot_positive", "slot_negative",
            "attention_loss_sum", "attention_loss_count",
            "detail_weight_l5_sum", "detail_weight_l15_sum",
            "detail_weight_l26_sum", "detail_weight_count",
        )
        sums = []
        minima = []
        maxima = []
        for name in scalar_names:
            state = self._ui5_scalar[name]
            sums.extend((state["sum"], state["count"]))
            minima.append(state["min"])
            maxima.append(state["max"])
        for task in TRAIN_TASKS:
            sums.extend(self._ui5_tasks[task][key] for key in task_keys)
        sums.append(self._ui5_peak_gpu_memory_mb)

        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        sum_tensor = torch.tensor(sums, dtype=torch.float64, device=device)
        min_tensor = torch.tensor(minima, dtype=torch.float64, device=device)
        max_tensor = torch.tensor(maxima, dtype=torch.float64, device=device)
        peak_tensor = sum_tensor[-1:].clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_tensor[:-1], op=dist.ReduceOp.SUM)
            dist.all_reduce(min_tensor, op=dist.ReduceOp.MIN)
            dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
            dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
        sum_tensor[-1] = peak_tensor[0]

        local_pr_observations = {
            task: {
                "scores": list(self._ui5_tasks[task]["pr_scores"]),
                "labels": list(self._ui5_tasks[task]["pr_labels"]),
            }
            for task in TRAIN_TASKS
        }
        gathered_pr_observations = [local_pr_observations]
        if dist.is_available() and dist.is_initialized():
            gathered_pr_observations = [None] * dist.get_world_size()
            dist.all_gather_object(gathered_pr_observations, local_pr_observations)
        task_pr_auc = {}
        for task in TRAIN_TASKS:
            scores = []
            labels = []
            for rank_observations in gathered_pr_observations:
                if not rank_observations:
                    continue
                task_observations = rank_observations.get(task, {})
                scores.extend(task_observations.get("scores", ()))
                labels.extend(task_observations.get("labels", ()))
            task_pr_auc[task] = self._average_precision(scores, labels)

        values = sum_tensor.cpu().tolist()
        minima = min_tensor.cpu().tolist()
        maxima = max_tensor.cpu().tolist()
        cursor = 0
        reduced_scalars = {}
        for index, name in enumerate(scalar_names):
            reduced_scalars[name] = {
                "sum": values[cursor],
                "count": values[cursor + 1],
                "min": minima[index],
                "max": maxima[index],
            }
            cursor += 2
        reduced_tasks = {}
        for task in TRAIN_TASKS:
            reduced_tasks[task] = dict(zip(task_keys, values[cursor:cursor + len(task_keys)]))
            cursor += len(task_keys)
        return reduced_scalars, reduced_tasks, values[-1], task_pr_auc

    def _reduce_curriculum_pool_draw_counts(self):
        """Return globally summed, resume-stable sampler draws by pool."""

        schedule = getattr(self.train_dataset, "curriculum_schedule", None)
        dataset_pools = getattr(self.train_dataset, "dataset_pools", None)
        if schedule is None or dataset_pools is None:
            return None
        worker_states = getattr(self.train_dataset, "_worker_states", {})
        configured_workers = getattr(
            self.train_dataset, "_configured_num_workers", None
        )
        local_available = bool(worker_states) and (
            configured_workers is None or len(worker_states) == configured_workers
        )
        try:
            local_counts = (
                curriculum_pool_draw_counts(worker_states, dataset_pools)
                if local_available
                else {"hard": 0, "matched_anchor": 0, "global_replay": 0}
            )
        except ValueError as exc:
            # Every rank must still enter the same collective.  Report N/A
            # globally instead of letting one malformed local snapshot strand
            # its peers inside all_reduce.
            logger.warning("Curriculum pool draw snapshot is unavailable: %s", exc)
            local_available = False
            local_counts = {"hard": 0, "matched_anchor": 0, "global_replay": 0}
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        count_tensor = torch.tensor(
            [
                local_counts[pool]
                for pool in ("hard", "matched_anchor", "global_replay")
            ]
            + [int(local_available)],
            dtype=torch.int64,
            device=device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        values = count_tensor.cpu().tolist()
        expected_available_ranks = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        if int(values[3]) != expected_available_ranks:
            return None
        return {
            "hard": int(values[0]),
            "matched_anchor": int(values[1]),
            "global_replay": int(values[2]),
        }

    @staticmethod
    def _average_precision(scores, labels):
        """Exact average precision over one complete diagnostic window."""

        if not scores or len(scores) != len(labels):
            return None
        paired = sorted(
            zip(scores, labels),
            key=lambda item: float(item[0]),
            reverse=True,
        )
        positive_count = sum(bool(label) for _, label in paired)
        if positive_count == 0:
            return None
        true_positives = 0
        precision_sum = 0.0
        for rank, (_, label) in enumerate(paired, start=1):
            if bool(label):
                true_positives += 1
                precision_sum += true_positives / rank
        return precision_sum / positive_count

    @staticmethod
    def _average(state):
        return state["sum"] / state["count"] if state["count"] else None

    def _flush_ui5_excel(self, step, logs):
        scalars, tasks, peak_memory, task_pr_auc = self._reduce_ui5_window()
        curriculum_pool_counts = self._reduce_curriculum_pool_draw_counts()
        config = self.model.config
        def global_grad_rms(group):
            state = scalars[f"{group}_grad_norm"]
            if not state["count"]:
                return None
            return (
                state["sum"] * max(1, int(self.args.world_size)) / state["count"]
            ) ** 0.5
        relation_grad_norm = global_grad_rms("relation")
        image_gate_grad_norm = global_grad_rms("image_gate")
        slot_gate_grad_norm = global_grad_rms("slot_gate")
        pbd_grad_norm = global_grad_rms("pbd")
        gate_grad_components = (
            value
            for value in (image_gate_grad_norm, slot_gate_grad_norm)
            if value is not None
        )
        gate_grad_squared_sum = sum(value * value for value in gate_grad_components)
        gate_grad_norm = (
            gate_grad_squared_sum ** 0.5
            if image_gate_grad_norm is not None or slot_gate_grad_norm is not None
            else None
        )
        current_epoch = float(self.state.epoch or 0.0)
        segment_start_epoch = float(self._ui5_segment_start_epoch or 0.0)
        segment_epoch = max(0.0, current_epoch - segment_start_epoch)
        global_epoch = self._ui5_global_epoch_offset + segment_epoch
        metrics = {
            "step": step,
            "segment_epoch": segment_epoch,
            "global_epoch": global_epoch,
            "gpu_num": self.args.world_size,
            "max_num_tokens": self._max_num_tokens,
            "learning_rate": logs.get("learning_rate"),
            "gate_loss_weight": float(getattr(config, "relation_gate_loss_weight", 1.0)),
            "slot_gate_loss_weight": float(getattr(config, "relation_slot_gate_loss_weight", 0.1)),
            "attention_loss_weight": float(getattr(config, "relation_attention_loss_weight", 0.1)),
            "gate_threshold": float(getattr(config, "relation_gate_threshold", 0.5)),
            "focal_beta": float(getattr(config, "relation_focal_beta", 0.999)),
            "focal_gamma": float(getattr(config, "relation_focal_gamma", 2.0)),
            "relation_num_slots": int(getattr(config, "relation_num_slots", 8)),
            "loss_total": self._average(scalars["loss_total"]),
            "loss_total_min": scalars["loss_total"]["min"],
            "loss_total_max": scalars["loss_total"]["max"],
            "loss_lm": self._average(scalars["loss_lm"]),
            "loss_gate": self._average(scalars["loss_gate"]),
            "loss_image_gate": self._average(scalars["loss_image_gate"]),
            "loss_slot_gate": self._average(scalars["loss_slot_gate"]),
            "loss_attention": self._average(scalars["loss_attention"]),
            "weighted_gate_loss": self._average(scalars["weighted_gate_loss"]),
            "weighted_slot_gate_loss": self._average(scalars["weighted_slot_gate_loss"]),
            "weighted_attention_loss": self._average(scalars["weighted_attention_loss"]),
            "loss_lm_contribution": self._average(scalars["loss_lm_contribution"]),
            "loss_image_gate_contribution": self._average(scalars["loss_image_gate_contribution"]),
            "loss_slot_gate_contribution": self._average(scalars["loss_slot_gate_contribution"]),
            "loss_attention_contribution": self._average(scalars["loss_attention_contribution"]),
            "loss_reconstructed": self._average(scalars["loss_reconstructed"]),
            "loss_reconstruction_error": self._average(scalars["loss_reconstruction_error"]),
            "attention_active_batch_rate": self._average(scalars["attention_active_batch_rate"]),
            "grad_norm": self._average(scalars["grad_norm"]),
            "grad_norm_max": scalars["grad_norm"]["max"],
            "samples": sum(values["samples"] for values in tasks.values()),
            "positive_samples": sum(values["positive"] for values in tasks.values()),
            "negative_samples": sum(values["negative"] for values in tasks.values()),
            "peak_gpu_memory_mb": peak_memory,
            "detail_layer5_norm": self._average(scalars["detail_layer5_norm"]),
            "detail_layer15_norm": self._average(scalars["detail_layer15_norm"]),
            "detail_layer26_norm": self._average(scalars["detail_layer26_norm"]),
            "detail_layer5_abs_max": self._average(scalars["detail_layer5_abs_max"]),
            "detail_layer15_abs_max": self._average(scalars["detail_layer15_abs_max"]),
            "detail_layer26_abs_max": self._average(scalars["detail_layer26_abs_max"]),
            "detail_layer5_saturation_fraction": self._average(scalars["detail_layer5_saturation_fraction"]),
            "detail_layer15_saturation_fraction": self._average(scalars["detail_layer15_saturation_fraction"]),
            "detail_layer26_saturation_fraction": self._average(scalars["detail_layer26_saturation_fraction"]),
            "detail_norm_ratio": self._average(scalars["detail_norm_ratio"]),
            "detail_fused_norm": self._average(scalars["detail_fused_norm"]),
            "relation_context_norm": self._average(scalars["relation_context_norm"]),
            "relation_gate_prob_mean": self._average(scalars["relation_gate_prob_mean"]),
            "pbd_delta_norm": self._average(scalars["pbd_delta_norm"]),
            "pbd_active_positions": scalars["pbd_active_positions"]["sum"],
            "relation_grad_norm": relation_grad_norm,
            "gate_grad_norm": gate_grad_norm,
            "image_gate_grad_norm": image_gate_grad_norm,
            "slot_gate_grad_norm": slot_gate_grad_norm,
            "pbd_grad_norm": pbd_grad_norm,
            "tasks": {},
        }
        if curriculum_pool_counts is not None:
            metrics.update(
                {
                    "curriculum_hard_samples": curriculum_pool_counts["hard"],
                    "curriculum_anchor_samples": curriculum_pool_counts["matched_anchor"],
                    "curriculum_global_replay_samples": curriculum_pool_counts[
                        "global_replay"
                    ],
                }
            )
        for group in ("relation", "image_gate", "slot_gate", "pbd"):
            metrics[f"{group}_grad_seen_steps"] = scalars[f"{group}_grad_seen_steps"]["sum"]
            for suffix in (
                "absolute_update_norm",
                "relative_update_norm",
                "changed_element_count",
            ):
                metrics[f"{group}_{suffix}"] = self._average(
                    scalars[f"{group}_{suffix}"]
                )
        for task, values in tasks.items():
            tp, fp, fn = values["tp"], values["fp"], values["fn"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            metrics["tasks"][task] = {
                "samples": values["samples"],
                "positive": values["positive"],
                "negative": values["negative"],
                "gate_loss": (
                    values["gate_loss_sum"] / values["gate_loss_count"]
                    if values["gate_loss_count"] else None
                ),
                "attention_loss": (
                    values["attention_loss_sum"] / values["attention_loss_count"]
                    if values["attention_loss_count"] else None
                ),
                "p_defect_pos": (
                    values["p_defect_pos_sum"] / values["p_defect_pos_count"]
                    if values["p_defect_pos_count"] else None
                ),
                "p_defect_neg": (
                    values["p_defect_neg_sum"] / values["p_defect_neg_count"]
                    if values["p_defect_neg_count"] else None
                ),
                "gate_precision": precision,
                "gate_recall": recall,
                "gate_f1": f1,
                "gate_pr_auc": task_pr_auc.get(task),
                "slot_gate_loss": (
                    values["slot_gate_loss_sum"] / values["slot_gate_loss_count"]
                    if values["slot_gate_loss_count"] else None
                ),
                "slot_positive": values["slot_positive"],
                "slot_negative": values["slot_negative"],
                "detail_weight_l5": (
                    values["detail_weight_l5_sum"] / values["detail_weight_count"]
                    if values["detail_weight_count"] else None
                ),
                "detail_weight_l15": (
                    values["detail_weight_l15_sum"] / values["detail_weight_count"]
                    if values["detail_weight_count"] else None
                ),
                "detail_weight_l26": (
                    values["detail_weight_l26_sum"] / values["detail_weight_count"]
                    if values["detail_weight_count"] else None
                ),
            }
        schedule = getattr(self.train_dataset, "curriculum_schedule", None)
        if schedule is not None:
            stage = schedule.stage_for_optimizer_step(int(step))
            next_stage = schedule.stage_after_completed_step(int(step))
            target = stage.to_dict()["pool_weights"]
            curriculum_log = {
                "curriculum_phase": stage.index + 1,
                "curriculum_next_phase": next_stage.index + 1,
                "hard_ratio": target["hard"],
                "anchor_ratio": target["matched_anchor"],
                "global_replay_ratio": target["global_replay"],
                "curriculum_target_llm_lr": stage.llm_lr,
                "curriculum_next_llm_lr": next_stage.llm_lr,
            }
            for name in (
                "curriculum_hard_samples",
                "curriculum_anchor_samples",
                "curriculum_global_replay_samples",
            ):
                if name in metrics:
                    curriculum_log[name] = metrics[name]
            metrics.update(curriculum_log)
            logs.update(curriculum_log)
            for history_row in reversed(self.state.log_history):
                try:
                    history_step = int(history_row.get("step", -1))
                except (TypeError, ValueError):
                    continue
                if history_step == int(step):
                    history_row.update(curriculum_log)
                    break

        if self.is_world_process_zero():
            written = self._ui5_excel.update_train(step, metrics)
            logger.info(
                "[UI5Excel] step=%s written=%s path=%s",
                step,
                written,
                self._ui5_excel.path,
            )
            if schedule is not None:
                segment_target = int(
                    os.environ.get("LOCANY_STOP_AFTER_STEP", schedule.total_steps)
                )
                pool_counts = {
                    "hard": metrics.get("curriculum_hard_samples"),
                    "anchor": metrics.get("curriculum_anchor_samples"),
                    "global_replay": metrics.get(
                        "curriculum_global_replay_samples"
                    ),
                }
                status = {
                    "event": "train_progress",
                    "step": int(step),
                    "phase": stage.index + 1,
                    "curriculum_target": {
                        "hard_ratio": target["hard"],
                        "anchor_ratio": target["matched_anchor"],
                        "global_replay_ratio": target["global_replay"],
                        "llm_lr": stage.llm_lr,
                    },
                    "next_curriculum_target": {
                        "phase": next_stage.index + 1,
                        "hard_ratio": next_stage.pool_weights[0],
                        "anchor_ratio": next_stage.pool_weights[1],
                        "global_replay_ratio": next_stage.pool_weights[2],
                        "llm_lr": next_stage.llm_lr,
                    },
                    "training": {
                        "learning_rate": metrics["learning_rate"],
                        "learning_rate_semantics": "next_optimizer_step",
                        "loss_total": metrics["loss_total"],
                        "loss_lm": metrics["loss_lm"],
                        "grad_norm": metrics["grad_norm"],
                        "window_samples": int(metrics["samples"]),
                        "pool_samples_cumulative": pool_counts,
                        "pool_sample_count_status": (
                            "available"
                            if curriculum_pool_counts is not None
                            else "N/A: incomplete worker iterator snapshots"
                        ),
                    },
                    "next_action": (
                        f"continue_training_to_step_{segment_target}"
                        if int(step) < segment_target
                        else f"exit_training_segment_for_evaluation_step_{step}"
                    ),
                }
                logger.info(
                    "[TRAIN SNAPSHOT] step=%s completed_phase=%s next_phase=%s "
                    "lr_next=%s next_target_lr=%s completed_phase_lr=%s "
                    "loss=%s loss_lm=%s grad_norm=%s window_samples=%s "
                    "pool_draws_cumulative=hard:%s,anchor:%s,global_replay:%s",
                    step,
                    stage.index + 1,
                    next_stage.index + 1,
                    metrics["learning_rate"] if metrics["learning_rate"] is not None else "N/A",
                    next_stage.llm_lr,
                    stage.llm_lr,
                    metrics["loss_total"] if metrics["loss_total"] is not None else "N/A",
                    metrics["loss_lm"] if metrics["loss_lm"] is not None else "N/A",
                    metrics["grad_norm"] if metrics["grad_norm"] is not None else "N/A",
                    int(metrics["samples"]),
                    pool_counts["hard"] if pool_counts["hard"] is not None else "N/A",
                    pool_counts["anchor"] if pool_counts["anchor"] is not None else "N/A",
                    pool_counts["global_replay"] if pool_counts["global_replay"] is not None else "N/A",
                )
                logger.info(
                    "[CURRICULUM STATUS] %s",
                    json.dumps(status, ensure_ascii=False, separators=(",", ":")),
                )
        self._reset_ui5_window()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def log(self, logs, start_time=None):
        if not self._ui5_enabled:
            return super().log(logs, start_time)
        if "grad_norm" in logs:
            self._add_ui5_scalar("grad_norm", logs["grad_norm"])
        result = super().log(logs, start_time)
        step = int(self.state.global_step)
        if step in {1, 20, 100}:
            self._capture_ui5_parameter_updates()
            if step == 20:
                required_groups = {"relation", "image_gate", "slot_gate"}
                if self._ui5_scalar["pbd_active_positions"]["sum"] > 0:
                    required_groups.add("pbd")
                failed_groups = [
                    group
                    for group in sorted(required_groups)
                    if self._ui5_scalar[f"{group}_absolute_update_norm"]["sum"] <= 0
                ]
                if failed_groups:
                    raise RuntimeError(
                        "UI modules had effective supervision but no parameter update "
                        f"through optimizer step 20: {failed_groups}"
                    )
        if (
            step > 0
            and step % 100 == 0
            and step != self._ui5_last_flushed_step
        ):
            self._flush_ui5_excel(step, logs)
            self._ui5_last_flushed_step = step
            self._save_ui5_window_state()
        elif "train_runtime" in logs:
            # Persist a partial 100-step window at a clean segment/final stop.
            self._save_ui5_window_state()
        return result
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        from torch.utils.data import DataLoader
        
        pad_id = 0
        if self.processing_class is not None and hasattr(self.processing_class, 'tokenizer'):
            pad_id = self.processing_class.tokenizer.pad_token_id or 0
        
        collate_fn = PackedCollatorMTP(pad_id=pad_id, dataset=self.train_dataset)
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=self.args.dataloader_num_workers,
            collate_fn=collate_fn,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=self.args.dataloader_prefetch_factor if self.args.dataloader_num_workers > 0 else None,
        )
        
        return StateAwareDataLoader(dataloader, self.train_dataset)


def build_stream_packed_dataset_mtp(
    model_args,
    data_args, 
    processor, 
    base_seed: int = 42,
    total_steps: int = 0,
) -> StreamPackedDatasetMTP:
    """Build StreamPackedDatasetMTP."""
    ds_collections = json.loads(open(data_args.meta_path).read())
    curriculum_schedule = UI5CurriculumSchedule.from_environment(
        os.environ,
        default_total_steps=int(total_steps),
    )
    scheduled_curriculum = curriculum_schedule is not None
    if scheduled_curriculum and curriculum_schedule.total_steps != int(total_steps):
        raise ValueError(
            "Scheduled TOTAL_STEPS must equal Trainer --max_steps: "
            f"TOTAL_STEPS={curriculum_schedule.total_steps}, max_steps={total_steps}"
        )
    curriculum_identity = None
    if scheduled_curriculum:
        curriculum_identity = curriculum_artifact_identity(
            data_args.meta_path, curriculum_schedule
        )
        runtime_sampling_mode = os.environ.get("UI5_UI_SAMPLING_MODE", "fixed_ratio")
        if runtime_sampling_mode != "fixed_ratio":
            raise ValueError(
                "CURRICULUM_MODE=scheduled requires UI5_UI_SAMPLING_MODE=fixed_ratio; "
                "the outer curriculum sampler owns all three pool weights"
            )
        logger.warning(
            "[UI5 curriculum] enabled schedule=%s fingerprint=%s",
            curriculum_schedule.to_dict(),
            curriculum_schedule.fingerprint,
        )
    
    datasets = []
    dataset_weights = []
    dataset_pools = []
    
    for ds_name, meta in ds_collections.items():
        meta = resolve_recipe_entry_paths(meta, data_args.meta_path)
        curriculum_pool = None
        if scheduled_curriculum:
            if "curriculum_pool" not in meta:
                raise ValueError(
                    f"Dataset {ds_name!r} lacks required curriculum_pool; expected "
                    "hard, matched_anchor, or global_replay"
                )
            curriculum_pool = canonical_curriculum_pool(meta["curriculum_pool"])
            # The three exported pools need not each contain all five tasks.
            # Retain every legal record and let only the outer sampler apply the
            # requested stage ratios.
            meta = dict(meta)
            meta["curriculum_pool"] = curriculum_pool
            meta["balance_ui_defects"] = False
            meta["ui_sampling_mode"] = "fixed_ratio"
        repeat_time = meta.get('repeat_time', 1)
        try:
            ds = LazySupervisedDatasetMTP(
                ds_name, meta, processor,
                block_size=model_args.block_size,
                repeat_time=repeat_time,
                target_fps=data_args.target_fps,
                max_frames=data_args.max_frames,
                video_total_pixels=data_args.video_total_pixels,
                balance_ui_defects=(False if scheduled_curriculum else data_args.balance_ui_defects),
                ui_records_per_class=data_args.ui_records_per_class,
                ui_negative_to_positive_ratio=data_args.ui_negative_to_positive_ratio,
                ui_sampling_mode=data_args.ui_sampling_mode,
                curriculum_group_sampling=scheduled_curriculum,
            )
            
            if len(ds) == 0:
                logger.warning(f'Dataset {ds_name} is empty, skipping.')
                continue
            
            datasets.append(ds)
            
            weight = resolve_dataset_sampling_weight(meta, len(ds))
            dataset_weights.append(weight)
            if scheduled_curriculum:
                dataset_pools.append(curriculum_pool)
            
            logger.info(
                f'Added dataset: {ds_name}, length={len(ds)}, '
                f'repeat_time={repeat_time}, sampling_weight={weight:g}, '
                f'explicit_sampling_weight={meta.get("sampling_weight")!r}'
            )
            
        except Exception as e:
            traceback.print_exc()
            logger.error(f'Error loading dataset {ds_name}: {e}')
            raise

    if len(datasets) == 0:
        raise ValueError("No valid datasets found!")

    buffer_size = getattr(data_args, 'packing_buffer_size', 32)

    return StreamPackedDatasetMTP(
        tokenizer=processor.tokenizer,
        data_rank=get_rank(),
        data_world_size=get_world_size(),
        datasets=datasets,
        dataset_weight=dataset_weights,
        max_num_tokens_per_sample=getattr(data_args, 'max_num_tokens_per_sample', 16384),
        max_num_tokens=getattr(data_args, 'max_num_tokens', data_args.max_seq_length),
        log_freq=10000,
        base_seed=base_seed,
        buffer_size=buffer_size,
        curriculum_schedule=curriculum_schedule,
        dataset_pools=(dataset_pools if scheduled_curriculum else None),
        curriculum_identity=curriculum_identity,
    )


def main():
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if os.path.exists(osp.join(training_args.output_dir, 'done.txt')):
        logger.info("Training done (done.txt exists), exiting!")
        return
    
    # Patches - Note: NOT applying patch_packing_attention since we use custom attention
    # patch_packing_attention()  # Disabled - using MTP-specific attention
    replace_train_dataloader()
    replace_train_sampler()
    
    # Logging setup
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, '
        f'n_gpu: {training_args.n_gpu}, distributed: {bool(training_args.local_rank != -1)}, '
        f'fp16: {training_args.fp16}'
    )
    logger.info(f'Training parameters: {training_args}')

    training_args_serialization_preflight(training_args)

    # Checkpoint detection
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint_guard(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f'Checkpoint detected at {last_checkpoint}, resuming training.')
            
    set_seed(training_args.seed)
    
    # Load model and tokenizer
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False)
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    num_new_tokens = tokenizer.add_tokens(special_tokens_list + number_tokens_list, special_tokens=True)
    
    if len(tokenizer.encode("assistant")) > 1:
        tokenizer.add_tokens(["assistant"], special_tokens=False)
        num_new_tokens += 1
        
    image_token_index = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    text_mask_token_id = tokenizer.convert_tokens_to_ids(TEXT_MASK_TOKEN)
    null_token_id = tokenizer.convert_tokens_to_ids(NULL_TOKEN)
    box_start_token_id = tokenizer.convert_tokens_to_ids(BOX_START_TOKEN)
    box_end_token_id = tokenizer.convert_tokens_to_ids(BOX_END_TOKEN)
    ref_start_token_id = tokenizer.convert_tokens_to_ids(REF_START_TOKEN)
    ref_end_token_id = tokenizer.convert_tokens_to_ids(REF_END_TOKEN)
    coord_start_token_id = tokenizer.convert_tokens_to_ids(number_tokens_list[0])
    coord_end_token_id = tokenizer.convert_tokens_to_ids(number_tokens_list[-1])
    none_token_ids = tokenizer.encode("none", add_special_tokens=False)
    none_token_id = none_token_ids[0] if len(none_token_ids) == 1 else 4064
    relation_detail_layers = None
    if model_args.relation_detail_layers:
        relation_detail_layers = [
            int(value.strip())
            for value in model_args.relation_detail_layers.split(",")
            if value.strip()
        ]
        if len(relation_detail_layers) != 3:
            raise ValueError("--relation_detail_layers must contain exactly three comma-separated indices")
    
    if model_args.model_name_or_path is not None:
        # ===== LocateAnything (MoonVit + Qwen2/Qwen3) Loading Path =====
        logger.info('Loading LocateAnythingForConditionalGeneration...')
        config = LocateAnythingConfig.from_pretrained(model_args.model_name_or_path)
        configure_ui5_model_config(
            config,
            attn_implementation=model_args.attn_implementation,
            image_token_index=image_token_index,
            block_size=model_args.block_size,
            causal_attn=model_args.causal_attn,
            text_mask_token_id=text_mask_token_id,
            null_token_id=null_token_id,
            box_start_token_id=box_start_token_id,
            box_end_token_id=box_end_token_id,
            coord_start_token_id=coord_start_token_id,
            coord_end_token_id=coord_end_token_id,
            ref_start_token_id=ref_start_token_id,
            ref_end_token_id=ref_end_token_id,
            none_token_id=none_token_id,
            enable_ui_relation=model_args.enable_ui_relation,
            relation_detail_hidden_size=model_args.relation_detail_hidden_size,
            relation_num_slots=model_args.relation_num_slots,
            relation_adapter_bottleneck=model_args.relation_adapter_bottleneck,
            relation_detail_layers=relation_detail_layers,
            relation_gate_loss_weight=model_args.relation_gate_loss_weight,
            relation_slot_gate_loss_weight=model_args.relation_slot_gate_loss_weight,
            relation_attention_loss_weight=model_args.relation_attention_loss_weight,
            relation_gate_threshold=model_args.relation_gate_threshold,
            relation_focal_beta=model_args.relation_focal_beta,
            relation_focal_gamma=model_args.relation_focal_gamma,
        )
        logger.info(f'Text attn: {model_args.attn_implementation}, Vision attn: flash_attention_2')

        loaded_model = LocateAnythingForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path, 
            torch_dtype=torch.bfloat16, config=config, 
            attn_implementation=model_args.attn_implementation,
            output_loading_info=True,
        )
        model, loading_info = loaded_model
        ui_load_report = initialize_or_validate_ui_relation(
            model,
            loading_info,
            seed=training_args.seed,
            all_missing_reason="all-ui-relation-keys-missing-from-base-checkpoint",
        )
        logger.warning("UI Relation/Gate/PBD load policy report: %s", ui_load_report)
            
        model.text_mask_token_id = text_mask_token_id
        model.language_model.block_size = int(model_args.block_size)
        model.language_model.causal_attn = model_args.causal_attn
        model.language_model.training = True
        
        try:
            processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, use_fast=True)
            processor.tokenizer = tokenizer
        except Exception as e:
            logger.warning(f'AutoProcessor failed ({e}), building processor from local configs...')
            chat_template_data = load_config(model_args.chat_template_path)
            processor_config = load_config(model_args.processor_config_path)
            preprocessor_config = load_config(model_args.preprocessor_config_path)
            image_processor = LocateAnythingImageProcessor(**preprocessor_config)
            processor_config["chat_template"] = chat_template_data["chat_template"]
            processor = LocateAnythingProcessor(tokenizer=tokenizer, image_processor=image_processor, **processor_config)
    else:
        logger.info(f"Loading vision backbone from {model_args.vision_path}")
        vision_config = AutoConfig.from_pretrained(model_args.vision_path, trust_remote_code=True)

        if vision_config.model_type == 'moonvit':
            logger.info('Loading MoonVit...')
            vision_config._attn_implementation = 'flash_attention_2'
            vision_model = MoonVitPretrainedModel.from_pretrained(
                model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config)
        else:
            raise ValueError(f"Unsupported vision model type: {vision_config.model_type}")
            
        logger.info('Loading LLM...')
        text_config = AutoConfig.from_pretrained(model_args.llm_path, trust_remote_code=True)
        text_config._attn_implementation = 'magi'

        llm = AutoModelForCausalLM.from_pretrained(
            model_args.llm_path, torch_dtype=torch.bfloat16,
            config=text_config, trust_remote_code=True)
        
        locateanything_config = LocateAnythingConfig(
            vision_config.to_dict(), text_config.to_dict(), 
            image_token_index=image_token_index, 
            mlp_connector_layers=model_args.mlp_connector_layers,
            box_start_token_id=box_start_token_id,
            box_end_token_id=box_end_token_id,
            coord_start_token_id=coord_start_token_id,
            coord_end_token_id=coord_end_token_id,
            ref_start_token_id=ref_start_token_id,
            ref_end_token_id=ref_end_token_id,
            none_token_id=none_token_id,
            enable_ui_relation=model_args.enable_ui_relation,
            relation_detail_hidden_size=model_args.relation_detail_hidden_size,
            relation_num_slots=model_args.relation_num_slots,
            relation_adapter_bottleneck=model_args.relation_adapter_bottleneck,
            relation_detail_layers=relation_detail_layers,
            relation_gate_loss_weight=model_args.relation_gate_loss_weight,
            relation_slot_gate_loss_weight=model_args.relation_slot_gate_loss_weight,
            relation_attention_loss_weight=model_args.relation_attention_loss_weight,
            relation_gate_threshold=model_args.relation_gate_threshold,
            relation_gate_mode="observe",
            relation_focal_beta=model_args.relation_focal_beta,
            relation_focal_gamma=model_args.relation_focal_gamma)
        locateanything_config._attn_implementation = 'magi'
        model = LocateAnythingForConditionalGeneration(locateanything_config, vision_model, llm)
        model.initialize_ui_relation_modules(
            training_args.seed, "new-model-without-composite-checkpoint"
        )

        chat_template_data = load_config(model_args.chat_template_path)
        processor_config = load_config(model_args.processor_config_path)
        preprocessor_config = load_config(model_args.preprocessor_config_path)
        image_processor = LocateAnythingImageProcessor(**preprocessor_config)
        processor_config["chat_template"] = chat_template_data["chat_template"]
        processor = LocateAnythingProcessor(tokenizer=tokenizer, image_processor=image_processor, **processor_config)
        
    model.neftune_alpha = data_args.neftune_alpha
    if getattr(model, "enable_ui_relation", False):
        validation_report = model.validate_ui_relation_parameters()
        logger.info("Validated UI relation parameters without mutation: %s", validation_report)
    
    # Enable packing mode for stream packing (works for both pretrained and scratch models)
    model.language_model.model.is_packing_mode = True
    
    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        model.config.text_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
        dist.barrier()

    model.language_model.config.use_cache = False

    if model_args.grad_checkpoint:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    logger.info("model init done")
    if getattr(model, "enable_ui_relation", False) and get_rank() == 0:
        logger.info(f"UI relation parameter report: {model.ui_relation_parameter_report()}")
    
    # Sequence parallelism
    if data_args.sequence_parallel_degree > 1:
        set_pg_manager(model, data_args.sequence_parallel_degree)
        logger.info(f'Sequence parallelism enabled: SP={data_args.sequence_parallel_degree}')

    # Multi-node info
    hostnames = [None] * dist.get_world_size()
    if get_pg_manager() is not None:
        local_info = f"sp_rank: {get_pg_manager().sequence_parallel_rank}, host: {socket.gethostname()}"
    else:
        local_info = f"sp_rank: None, host: {socket.gethostname()}"
    dist.all_gather_object(hostnames, local_info)
    
    if dist.get_rank() == 0:
        for i, info in enumerate(hostnames):
            logger.info(f"global rank[{i}]: {info}")

    # Build dataset
    logger.info("Building stream packed MTP dataset...")
    t_start = time.time()
    train_dataset = build_stream_packed_dataset_mtp(
        model_args,
        data_args,
        processor,
        base_seed=training_args.seed,
        total_steps=training_args.max_steps,
    )
    train_dataset.configure_num_workers(training_args.dataloader_num_workers)
    logger.info(f"Dataset built in {time.time() - t_start:.2f}s")

    # Freeze params
    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora)
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora)
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers:]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    # Verify parameter order consistency across all ranks (critical for ZeRO-3)
    param_names = [name for name, param in model.named_parameters()]
    param_names_list = [None] * dist.get_world_size()
    dist.all_gather_object(param_names_list, param_names)
    
    if dist.get_rank() == 0:
        logger.info("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"  {name}")
        
        # Verify all ranks have the same parameter names
        for rank, names in enumerate(param_names_list):
            if names != param_names:
                logger.warning(f"Rank {rank} has different parameter order! This may cause ZeRO-3 errors.")
                logger.warning(f"Rank 0 has {len(param_names)} params, Rank {rank} has {len(names)} params")
    
    # Critical: Synchronize all ranks before DeepSpeed initialization
    # This ensures parameter order consistency across ranks for ZeRO-3
    dist.barrier()
    logger.info(f"Rank {dist.get_rank()}: Model initialization synchronized across all ranks")
    if getattr(model, "enable_ui_relation", False):
        rank_report = model.assert_ui_relation_rank_consistency()
        if dist.get_rank() == 0:
            logger.info("UI relation cross-rank checksum: %s", rank_report)

    set_seed(training_args.seed)

    if model_args.lr_scale is not None:
        training_args.lr_scale = model_args.lr_scale
        replace_create_optimizer_with_various_lr()

    # Callbacks
    my_callbacks = []
    if model_args.save_every_n_hours > 0:
        my_callbacks.append(
            SaveCheckpointCallback(interval_hours=model_args.save_every_n_hours)
        )
    my_callbacks.append(MemoryLoggerCallback())
    my_callbacks.append(DataloaderStateCallback(train_dataset))
    my_callbacks.append(SamplingCoverageCallback(train_dataset, interval=1000))
    checkpoint_completion_callback = CheckpointCompletionCallback(
        train_dataset,
        model_args=model_args,
        data_args=data_args,
    )
    my_callbacks.append(checkpoint_completion_callback)
    stop_after_step = int(os.environ.get("LOCANY_STOP_AFTER_STEP", "0"))
    if stop_after_step:
        if stop_after_step > training_args.max_steps:
            raise ValueError(
                f"LOCANY_STOP_AFTER_STEP={stop_after_step} exceeds "
                f"max_steps={training_args.max_steps}"
            )
        my_callbacks.append(SegmentStopCallback(stop_after_step=stop_after_step))
    if _env_flag("LOCANY_ENABLE_MILESTONE_COPIES", default=True):
        milestone_interval = int(os.environ.get("LOCANY_MILESTONE_INTERVAL", "2000"))
        my_callbacks.append(
            MilestoneCheckpointCallback(milestone_interval=milestone_interval)
        )
    
    CustomTrainer = StreamPackingMTPTrainer

    assert processor is not None, "Processor is required"
    
    collate_fn = PackedCollatorMTP(pad_id=processor.tokenizer.pad_token_id, dataset=train_dataset)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        data_collator=collate_fn,
        callbacks=my_callbacks,
        processing_class=processor,
        sample_log_interval=getattr(data_args, 'sample_log_interval', 100),
        max_num_tokens=getattr(data_args, 'max_num_tokens', 0),
    )

    # Training
    if training_args.do_train:
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint
        completed_global_step = 0

        if checkpoint is not None:
            checkpoint = osp.abspath(osp.expanduser(os.fspath(checkpoint)))
            curriculum_schedule = train_dataset.curriculum_schedule
            resume_report = validate_checkpoint(
                Path(checkpoint),
                mode="resume",
                expected_ranks=(dist.get_world_size() if dist.is_initialized() else 1),
                strict=curriculum_schedule is not None,
                scaler_required=bool(training_args.fp16),
                expected_curriculum_fingerprint=(
                    curriculum_schedule.fingerprint
                    if curriculum_schedule is not None
                    else None
                ),
                require_completion_marker=curriculum_schedule is not None,
            )
            if not resume_report["valid"]:
                raise RuntimeError(
                    "Refusing unsafe checkpoint resume: "
                    f"checkpoint={checkpoint}; "
                    f"errors={'; '.join(resume_report['errors'])}"
                )
            completed_global_step = int(resume_report["details"]["global_step"])
            if curriculum_schedule is not None:
                saved_training_config = resume_report["details"][
                    "continuity_manifest"
                ].get("training_continuity_config")
                current_training_config = training_continuity_config(
                    training_args,
                    curriculum_schedule,
                    model_args=model_args,
                    data_args=data_args,
                )
                if saved_training_config != current_training_config:
                    raise RuntimeError(
                        "Training/optimizer configuration changed across curriculum "
                        f"resume: saved={saved_training_config}, "
                        f"current={current_training_config}"
                    )
            training_args.ignore_data_skip = True
            logger.info("Enabled ignore_data_skip=True for stateful dataloader resume.")
            
            rank = get_rank()
            dataloader_state_path = os.path.join(checkpoint, f"dataloader_state_rank{rank}.pt")
            
            if os.path.exists(dataloader_state_path):
                try:
                    dataloader_state = torch.load(dataloader_state_path, weights_only=False)
                    train_dataset.load_state_dict(
                        dataloader_state,
                        expected_global_step=completed_global_step,
                    )
                    logger.info(f"Rank {rank}: Loaded dataloader state from {dataloader_state_path}")
                except Exception as e:
                    logger.error(f"Rank {rank}: Failed to load dataloader state: {e}")
                    traceback.print_exc()
                    if train_dataset.curriculum_schedule is not None:
                        raise RuntimeError(
                            f"Rank {rank}: strict curriculum state restore failed"
                        ) from e
            else:
                message = f"Rank {rank}: No dataloader state found at {dataloader_state_path}"
                if train_dataset.curriculum_schedule is not None:
                    raise RuntimeError(message)
                logger.warning(message)

        if train_dataset.curriculum_schedule is not None:
            declared_start = os.environ.get("CURRICULUM_START_STEP")
            if (
                declared_start is not None
                and int(declared_start) != completed_global_step
            ):
                raise RuntimeError(
                    "CURRICULUM_START_STEP does not match the restored trainer state: "
                    f"declared={declared_start}, restored={completed_global_step}"
                )
            segment_target = int(
                os.environ.get("LOCANY_STOP_AFTER_STEP", training_args.max_steps)
            )
            train_dataset.curriculum_schedule.validate_segment(
                completed_global_step, segment_target
            )
            checkpoint_completion_callback.set_segment_bounds(
                source_global_step=completed_global_step,
                target_global_step=segment_target,
            )
            active_stage = train_dataset.curriculum_schedule.stage_after_completed_step(
                completed_global_step
            )
            logger.warning(
                "[UI5 curriculum] segment completed=%s target=%s active_stage=%s "
                "optimizer_steps=%s-%s dataset_weights=%s llm_lr=%s",
                completed_global_step,
                segment_target,
                active_stage.index,
                active_stage.first_optimizer_step,
                active_stage.last_optimizer_step,
                train_dataset.dataset_weight,
                active_stage.llm_lr,
            )

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        segment_mode = _env_flag("LOCANY_SEGMENT_MODE", default=False)
        if should_export_model_at_training_end(segment_mode=segment_mode):
            trainer.save_model()
        elif get_rank() == 0:
            logger.info(
                "Segment mode stopped at step %s; skipped duplicate model export to "
                "output root (including the final segment)",
                trainer.state.global_step,
            )

        if get_rank() == 0:
            output_dir = training_args.output_dir

            locany_utils_src = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'utils', 'locany')
            skip_files = {'config.json', 'README.md', '__init__.py', '__pycache__'}
            if osp.isdir(locany_utils_src):
                for file in os.listdir(locany_utils_src):
                    if file in skip_files or file.startswith('__'):
                        continue
                    src_file = osp.join(locany_utils_src, file)
                    dst_file = osp.join(output_dir, file)
                    if osp.isfile(src_file):
                        shutil.copy2(src_file, dst_file)
                logger.info(f"Copied inference files from {locany_utils_src} to {output_dir}")

            relation_module_src = osp.join(
                osp.dirname(osp.dirname(osp.abspath(__file__))),
                'model', 'locany', 'relation_modules.py'
            )
            if osp.isfile(relation_module_src):
                shutil.copy2(relation_module_src, osp.join(output_dir, 'relation_modules.py'))
                logger.info("Copied self-contained UI relation module to checkpoint")

            config_path = osp.join(output_dir, 'config.json')
            if osp.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                config_data['auto_map'] = {
                    "AutoConfig": "configuration_locateanything.LocateAnythingConfig",
                    "AutoModel": "modeling_locateanything.LocateAnythingForConditionalGeneration",
                    "AutoModelForCausalLM": "modeling_locateanything.LocateAnythingForConditionalGeneration",
                    "AutoImageProcessor": "image_processing_locateanything.LocateAnythingImageProcessor",
                    "AutoProcessor": "processing_locateanything.LocateAnythingProcessor",
                }
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                    f.write('\n')
                logger.info("Updated config.json with auto_map")

        metrics = train_result.metrics
        metrics['train_samples'] = 'streaming'

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        trainer.save_state()
        
    segment_mode = _env_flag("LOCANY_SEGMENT_MODE", default=False)
    if should_write_training_done_marker(segment_mode=segment_mode):
        with open(osp.join(training_args.output_dir, 'done.txt'), 'w') as f:
            f.write('done: ' + time.ctime())
    elif get_rank() == 0:
        logger.info(
            "Segment mode never writes training done.txt; the orchestrator must "
            "publish completion only after final evaluation and artifact updates"
        )


if __name__ == '__main__':
    main()
