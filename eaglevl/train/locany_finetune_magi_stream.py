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
import errno
import copy
import logging
import random
import sys
import warnings
from contextlib import nullcontext
import numpy as np
from typing import Dict, Optional, List, Tuple, Any, Mapping
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
from torch.distributed.elastic.multiprocessing.errors import record
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
from eaglevl.train.cpt_observability import (
    CPT_ID_TO_TASK,
    CPT_TASKS,
    CPT_TASK_TO_ID,
    add_sample_to_counter,
    aggregate_token_losses,
    empty_task_counter,
    merge_task_counters,
    restore_counter,
    sample_length_metadata,
    serializable_counter,
    stable_hash64,
    summarize_task_counter,
    supervision_kinds,
)
from eaglevl.train.cpt_sampling import (
    assert_sampling_resume_compatible,
    resolve_cpt_sampling,
)
from eaglevl.train.cpt_eval_queue import enqueue_pending_eval
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
    extract_ui_defect_targets,
    identify_ui_defect_task,
    is_positive_ui_defect,
)
from eaglevl.train.ui5_excel_logger import UI5ExcelLogger, TRAIN_TASKS
from eaglevl.train.ui5_checkpoint_utils import validate_checkpoint
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
                 ui_negative_to_positive_ratio: float = 2.0):
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
        self.balance_ui_defects = bool(meta.get("balance_ui_defects", balance_ui_defects))
        self.cpt_enabled = _env_flag("LOCANY_CPT_MODE", default=False)
        self.cpt_task = (
            str(meta.get("cpt_task") or ds_name.removeprefix("locany_cpt_"))
            if self.cpt_enabled
            else None
        )
        if self.cpt_enabled and self.cpt_task not in CPT_TASK_TO_ID:
            raise ValueError(f"Unknown CPT task for {ds_name}: {self.cpt_task!r}")
        self.cpt_task_id = CPT_TASK_TO_ID.get(self.cpt_task, -1)
        self.cpt_dataset_groups = int(meta.get("dataset_groups", 0) or 0)

        ann_paths = meta["annotation"]
        if not isinstance(ann_paths, (list, tuple)):
            ann_paths = [ann_paths]
        self.root = meta.get("root", "")
        
        logger.info(f"[Dataset] {self.ds_name} Indexing JSONL files...")
        start_time = time.time()
        self.lazy_loader = LazyJsonlLoader(ann_paths)
        logger.info(f"[Dataset] {self.ds_name} Indexing done in {time.time() - start_time:.2f}s.")
        
        original_num_rows = len(self.lazy_loader)
        logger.info(
            f"[Dataset] {self.ds_name} Found {original_num_rows} samples. "
            f"visual_prompt={self.visual_prompt}"
        )
        self.active_indices = list(range(original_num_rows))
        self._balanced_logical_buckets = None

        if self.balance_ui_defects:
            logger.info(
                f"[Dataset] {self.ds_name} building balanced UI index: "
                f"records_per_class={ui_records_per_class}, "
                f"negative:positive={ui_negative_to_positive_ratio}:1"
            )
            index_records = [self.lazy_loader[index] for index in range(original_num_rows)]
            self.active_indices = build_balanced_ui_indices(
                index_records,
                records_per_class=ui_records_per_class,
                negative_to_positive_ratio=ui_negative_to_positive_ratio,
            )
            logger.info(
                f"[Dataset] {self.ds_name} balanced to {len(self.active_indices)} records "
                f"({len(self.active_indices) // 5} per class)."
            )
            self._balanced_logical_buckets = {
                defect_type: {"positive": [], "negative": []}
                for defect_type in range(5)
            }
            for logical_index, raw_index in enumerate(self.active_indices):
                record = index_records[raw_index]
                defect_type = identify_ui_defect_task(record)[1]
                label = "positive" if is_positive_ui_defect(record) else "negative"
                self._balanced_logical_buckets[defect_type][label].append(logical_index)
        elif repeat_time < 1:
            if original_num_rows > 0:
                partial_len = int(original_num_rows * repeat_time)
                if partial_len > 0:
                    rnd = random.Random(10086)
                    sampled_indices = set(rnd.sample(range(original_num_rows), partial_len))
                    self.active_indices = [i for i in range(original_num_rows) if i in sampled_indices]
                    logger.info(f"[Dataset] {self.ds_name} Downsampled to {len(self.active_indices)} samples.")
                else:
                    self.active_indices = []
        
        self._length = len(self.active_indices)

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

    def get_epoch_indices(self, shuffle_seed: int) -> List[int]:
        rng = random.Random(shuffle_seed)
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
                cpt_supervision_kind=torch.tensor(
                    supervision_kinds(targets_out.tolist(), len_input_ids),
                    dtype=torch.long,
                ),
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
            cpt_supervision_kind=torch.tensor(
                supervision_kinds(targets_out.tolist(), len_input_ids),
                dtype=torch.long,
            ),
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
        self,
        messages: list,
        ui_targets: Optional[Dict[str, torch.Tensor]] = None,
        cpt_metadata: Optional[Dict[str, Any]] = None,
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
            text=message_text,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=False,
            # CPT must observe the real post-MTP length and explicitly skip an
            # oversize sample.  Preserve historical SFT truncation elsewhere.
            truncation=not self.cpt_enabled,
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
        if cpt_metadata is not None:
            image_token_id = getattr(self.processor, "image_token_id", None)
            if image_token_id is None:
                image_token_id = self.processor.tokenizer.convert_tokens_to_ids(
                    getattr(self.processor, "image_token", IMG_CONTEXT_TOKEN)
                )
            pre_mtp_length = int(inputs["input_ids"][0].numel())
            vision_tokens = int(inputs["input_ids"][0].eq(image_token_id).sum().item())
            lengths = sample_length_metadata(
                result["labels"].tolist(),
                pre_mtp_length=pre_mtp_length,
                vision_tokens=vision_tokens,
                ignore_index=IGNORE_TOKEN_ID,
            )
            result["cpt_task_token_ids"] = torch.full_like(
                result["labels"], self.cpt_task_id, dtype=torch.long
            )
            result["cpt_supervision_kind"] = labels_dict["cpt_supervision_kind"]
            result["_sample_task_ids"] = [self.cpt_task_id]
            result["_sample_task_names"] = [self.cpt_task]
            result["_sample_record_ids"] = [int(cpt_metadata["record_hash"])]
            result["_sample_group_ids"] = [int(cpt_metadata["group_hash"])]
            for key, value in lengths.items():
                result[f"_sample_{key}"] = [int(value)]
        return result

    def _cpt_record_metadata(self, data_item: dict, real_idx: int) -> dict[str, Any]:
        record_id = data_item.get("cpt_record_id") or data_item.get("id")
        if record_id is None:
            record_id = f"{self.ds_name}:{real_idx}"
        group_id = data_item.get("cpt_group_id")
        if self.cpt_enabled and not group_id:
            raise ValueError(
                f"[{self.ds_name}] record={record_id!r} has no cpt_group_id; "
                "prepare the group-level CPT split first"
            )
        return {
            "record_id": str(record_id),
            "group_id": str(group_id or record_id),
            "record_hash": stable_hash64(record_id),
            "group_hash": stable_hash64(group_id or record_id),
            "source": data_item.get("cpt_source"),
            "line": data_item.get("cpt_source_line"),
        }

    def _get_item_once(self, real_idx: int) -> Dict[str, torch.Tensor]:
        data_item = self.lazy_loader[real_idx]
        cpt_metadata = self._cpt_record_metadata(data_item, real_idx) if self.cpt_enabled else None
        ui_targets = extract_ui_defect_targets(data_item, max_boxes=8)
        processed = process_multimodal_sample(
            data_item,
            self.root,
            self.max_frames,
            self.target_fps,
            self.video_total_pixels,
            visual_prompt=self.visual_prompt,
        )
        return self.multi_modal_get_item(
            processed, ui_targets=ui_targets, cpt_metadata=cpt_metadata
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        retry_count = 0
        current_idx = idx
        
        seed = int(idx + 10086)
        random.seed(seed)
        np.random.seed(seed)

        if self.cpt_enabled:
            real_idx = self.active_indices[current_idx % self._length]
            try:
                return self._get_item_once(real_idx)
            except Exception as exc:
                raw = self.lazy_loader[real_idx]
                record_id = raw.get("cpt_record_id") or raw.get("id") or f"{self.ds_name}:{real_idx}"
                raise RuntimeError(
                    f"CPT sample failure: task={self.cpt_task}, "
                    f"record_id={record_id}, source={raw.get('cpt_source')}, "
                    f"line={raw.get('cpt_source_line')}: {type(exc).__name__}: {exc}"
                ) from exc
        
        while retry_count <= 10:
            real_idx = self.active_indices[(current_idx + retry_count) % self._length]
            try:
                return self._get_item_once(real_idx)
            except Exception as e:
                tb = traceback.format_exc()
                logger.warning(f"[{self.ds_name}] idx {real_idx} failed: {e}\n{tb}")
                retry_count += 1
        
        raise RuntimeError(f"[{self.ds_name}] Failed after 10 retries")
    
    def get_sample_at_global_idx(self, global_idx: int, seed: int) -> Dict[str, torch.Tensor]:
        """Get a sample by global index (used for resume)."""
        ds_len = self._length
        if ds_len == 0:
            raise ValueError("Dataset is empty")
        
        epoch = global_idx // ds_len
        pos = global_idx % ds_len
        
        shuffle_seed = seed + epoch * 999983
        indices = self.get_epoch_indices(shuffle_seed)
        
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
        self.ds_len = len(dataset)
        self.ds_name = getattr(dataset, 'ds_name', 'unknown')
        self.global_idx = start_global_idx
        
        self._cached_epoch = -1
        self._cached_indices = None
    
    def _get_epoch_indices(self, epoch: int) -> list:
        if self._cached_epoch == epoch and self._cached_indices is not None:
            return self._cached_indices
        
        shuffle_seed = self.seed + epoch * 999983
        indices = self.dataset.get_epoch_indices(shuffle_seed)
        
        self._cached_epoch = epoch
        self._cached_indices = indices
        return indices
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[dict, int]:
        if self.ds_len == 0:
            raise StopIteration
        
        current_global_idx = self.global_idx
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
        return IteratorState(seed=self.seed, global_idx=self.global_idx).to_dict()
    
    @classmethod
    def from_state_dict(cls, dataset: LazySupervisedDatasetMTP, state: dict) -> 'DeterministicIterator':
        return cls(dataset=dataset, seed=state['seed'], start_global_idx=state['global_idx'])


@dataclass
class WorkerState:
    """Complete state of a worker, including buffer state."""
    iterator_states: List[dict]
    sample_rng_state: tuple
    samples_produced: int
    batches_produced: int
    cpt_task_counters: Dict[str, dict]
    current_batch_locations: List[Tuple[int, int]]
    buffer_locations: List[Tuple[int, int]]
    
    def to_dict(self) -> dict:
        return {
            'iterator_states': self.iterator_states,
            'sample_rng_state': self.sample_rng_state,
            'samples_produced': self.samples_produced,
            'batches_produced': self.batches_produced,
            'cpt_task_counters': {
                task: serializable_counter(counter)
                for task, counter in self.cpt_task_counters.items()
            },
            'current_batch_locations': self.current_batch_locations,
            'buffer_locations': self.buffer_locations,
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
        
        return cls(
            iterator_states=d['iterator_states'],
            sample_rng_state=sample_rng_state,
            samples_produced=d.get('samples_produced', 0),
            batches_produced=d.get('batches_produced', 0),
            cpt_task_counters={
                task: restore_counter(counter)
                for task, counter in d.get('cpt_task_counters', {}).items()
            },
            current_batch_locations=d.get('current_batch_locations', []),
            buffer_locations=d.get('buffer_locations', []),
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
        cpt_sampling_config: Optional[dict] = None,
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
        self.cpt_enabled = _env_flag("LOCANY_CPT_MODE", default=False)
        self.cpt_sampling_config = copy.deepcopy(cpt_sampling_config)
        self._cpt_trained_counters = {
            task: empty_task_counter() for task in CPT_TASKS
        }
        self._cpt_ce_counters = {
            task: empty_task_counter() for task in CPT_TASKS
        }

        if dataset_weight is None:
            dataset_weight = [1] * len(datasets)
        total_weight = sum(dataset_weight)
        self.dataset_weight = [w / total_weight for w in dataset_weight]
        
        self._worker_states: Dict[str, dict] = {}
        self._resume_states: Dict[str, dict] = {}
        self._saved_num_workers: Optional[int] = None
        self._configured_num_workers: Optional[int] = None
        
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
                    "task": dataset.cpt_task,
                    "rows": len(dataset),
                    "probability": float(probability),
                }
                for dataset, probability in zip(self.datasets, self.dataset_weight)
            ],
        }

    def state_dict(self) -> dict:
        return {
            'worker_states': copy.deepcopy(self._worker_states),
            'base_seed': self.base_seed,
            'stream_resume_config': self._stream_resume_config(),
            'num_workers': (
                self._configured_num_workers
                if self._configured_num_workers is not None
                else len(self._worker_states)
            ),
            'cpt_sampling_config': copy.deepcopy(self.cpt_sampling_config),
            'cpt_trained_counters': {
                task: serializable_counter(counter)
                for task, counter in self._cpt_trained_counters.items()
            },
            'cpt_ce_counters': {
                task: serializable_counter(counter)
                for task, counter in self._cpt_ce_counters.items()
            },
            'version': 6,
        }
    
    def load_state_dict(self, state: dict):
        version = state.get('version', 1)
        if self.cpt_enabled and version < 6:
            raise RuntimeError(
                "CPT resume requires dataloader state version >= 6; "
                f"checkpoint has version={version}. Restart this v2 run from a clean output directory."
            )
        if version < 3:
            logger.warning(f"Loading old state version {version}, perfect resume not available.")
        saved_stream_config = state.get("stream_resume_config")
        current_stream_config = self._stream_resume_config()
        if self.cpt_enabled and saved_stream_config != current_stream_config:
            raise RuntimeError(
                "CPT stream configuration changed across resume: "
                f"saved={saved_stream_config}, current={current_stream_config}"
            )
        saved_num_workers = state.get("num_workers")
        self._saved_num_workers = (
            int(saved_num_workers) if saved_num_workers is not None else None
        )
        
        if 'worker_states' in state:
            self._resume_states = copy.deepcopy(state['worker_states'])
            self._worker_states = copy.deepcopy(state['worker_states'])
            if get_rank() == 0:
                logger.info(f"Loaded resume states for {len(self._resume_states)} workers")
        saved_sampling = state.get("cpt_sampling_config")
        if self.cpt_enabled and saved_sampling is not None:
            assert_sampling_resume_compatible(self.cpt_sampling_config, saved_sampling)
        for destination, key in (
            (self._cpt_trained_counters, "cpt_trained_counters"),
            (self._cpt_ce_counters, "cpt_ce_counters"),
        ):
            for task, counter in state.get(key, {}).items():
                if task in destination:
                    destination[task] = restore_counter(counter)

    @staticmethod
    def _sample_cpt_metadata(sample: dict, index: int = 0) -> dict[str, Any]:
        return {
            "task_id": int(sample["_sample_task_ids"][index]),
            "task": str(sample["_sample_task_names"][index]),
            "record_hash": int(sample["_sample_record_ids"][index]),
            "group_hash": int(sample["_sample_group_ids"][index]),
            "raw_text_tokens": int(sample["_sample_raw_text_tokens"][index]),
            "vision_tokens": int(sample["_sample_vision_tokens"][index]),
            "pre_mtp_seq_len": int(sample["_sample_pre_mtp_seq_len"][index]),
            "post_mtp_seq_len": int(sample["_sample_post_mtp_seq_len"][index]),
            "main_supervised_tokens": int(sample["_sample_main_supervised_tokens"][index]),
            "mtp_supervised_tokens": int(sample["_sample_mtp_supervised_tokens"][index]),
            "total_supervised_tokens": int(sample["_sample_total_supervised_tokens"][index]),
        }

    def record_trained_batch(self, batch: dict) -> None:
        if not self.cpt_enabled or "_sample_task_ids" not in batch:
            return
        tasks_seen = set()
        for index in range(len(batch["_sample_task_ids"])):
            metadata = self._sample_cpt_metadata(batch, index)
            counter = self._cpt_trained_counters[metadata["task"]]
            add_sample_to_counter(counter, metadata, outcome="trained")
            counter["packed_tokens"] += metadata["post_mtp_seq_len"]
            tasks_seen.add(metadata["task"])
        for task in tasks_seen:
            self._cpt_trained_counters[task]["packed_batches"] += 1

    def record_cpt_ce(self, values: Mapping[int, Mapping[str, Any]]) -> None:
        if not self.cpt_enabled:
            return
        for task_id, metrics in values.items():
            task = CPT_ID_TO_TASK[int(task_id)]
            counter = self._cpt_ce_counters[task]
            for key in (
                "main_loss_sum",
                "main_loss_tokens",
                "mtp_loss_sum",
                "mtp_loss_tokens",
            ):
                counter[key] += metrics[key]

    def local_cpt_counters(self) -> dict[str, dict]:
        output = {}
        for task in CPT_TASKS:
            worker_values = []
            for state in self._worker_states.values():
                counter = state.get("cpt_task_counters", {}).get(task)
                if counter is not None:
                    worker_values.append(counter)
            output[task] = merge_task_counters(
                [
                    *worker_values,
                    serializable_counter(self._cpt_trained_counters[task]),
                    serializable_counter(self._cpt_ce_counters[task]),
                ]
            )
        return output

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
            elif k.startswith('_sample_'):
                result[k] = batch[k] + sample[k]
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
        cpt_task_counters = {task: empty_task_counter() for task in CPT_TASKS}
        
        current_batch = None
        current_batch_locations: List[Tuple[int, int]] = []
        buffer: List[Tuple[dict, int, int]] = []
        
        # Resume handling
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
                for task, counter in ws.cpt_task_counters.items():
                    if task in cpt_task_counters:
                        cpt_task_counters[task] = restore_counter(counter)
                
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
                            if self.cpt_enabled:
                                raise RuntimeError(
                                    f"[{worker_key}] CPT resume could not rebuild batch sample {loc}"
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
                            if self.cpt_enabled:
                                raise RuntimeError(
                                    f"[{worker_key}] CPT resume could not rebuild buffer sample {loc}"
                                ) from e
                            logger.warning(f'[{worker_key}] Failed to restore buffer sample {loc}: {e}')
                
                if is_main_log:
                    logger.info(f'[{worker_key}] Resume complete. Buffer size: {len(buffer)}')
                    
            except Exception as e:
                logger.error(f'[{worker_key}] Failed to resume: {e}')
                traceback.print_exc()
                if self.cpt_enabled:
                    raise
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
                cpt_task_counters=cpt_task_counters,
                current_batch_locations=list(current_batch_locations),
                buffer_locations=[(b[1], b[2]) for b in buffer],
            ).to_dict()

        def fetch_next_sample() -> Tuple[dict, int, int]:
            nonlocal samples_produced, skipped_count
            while True:
                ds_idx = sample_rng.choices(range(len(self.datasets)), weights=self.dataset_weight)[0]
                try:
                    sample, global_idx = next(iterators[ds_idx])
                    samples_produced += 1
                    metadata = (
                        self._sample_cpt_metadata(sample)
                        if self.cpt_enabled
                        else None
                    )
                    if metadata is not None:
                        add_sample_to_counter(
                            cpt_task_counters[metadata["task"]],
                            metadata,
                            outcome="attempted",
                        )
                    if self._get_sample_length(sample) > self.max_num_tokens_per_sample:
                        skipped_count += 1
                        if metadata is not None:
                            metadata["pre_mtp_oversize"] = (
                                metadata["pre_mtp_seq_len"]
                                > self.max_num_tokens_per_sample
                            )
                            add_sample_to_counter(
                                cpt_task_counters[metadata["task"]],
                                metadata,
                                outcome="oversize",
                            )
                        continue
                    if metadata is not None:
                        add_sample_to_counter(
                            cpt_task_counters[metadata["task"]],
                            metadata,
                            outcome="accepted",
                        )
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

    if "cpt_task_token_ids" in feat or "cpt_supervision_kind" in feat:
        if not (
            "cpt_task_token_ids" in feat
            and "cpt_supervision_kind" in feat
            and feat["cpt_task_token_ids"].numel() == label_len
            and feat["cpt_supervision_kind"].numel() == label_len
        ):
            raise ValueError("CPT task/supervision token metadata is not aligned with labels")
        result["cpt_task_token_ids"] = feat["cpt_task_token_ids"].unsqueeze(0)
        result["cpt_supervision_kind"] = feat["cpt_supervision_kind"].unsqueeze(0)
        for key, value in feat.items():
            if key.startswith("_sample_"):
                result[key] = value

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
                    self.dataset._worker_states[worker_key] = state_snapshot

            batch.pop('_batch_idx', None)
            if self.dataset is not None:
                self.dataset.record_trained_batch(batch)
            
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
        
        checkpoint_folder = f"checkpoint-{state.global_step}"
        output_dir = os.path.join(args.output_dir, checkpoint_folder)
        rank = get_rank()
        state_path = os.path.join(output_dir, f"dataloader_state_rank{rank}.pt")
        temp_path = state_path + ".tmp"
        
        try:
            ds_state = self.train_dataset.state_dict()
            
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
            
            torch.save(ds_state, temp_path)
            os.replace(temp_path, state_path)
            
            if rank == 0:
                logger.info(f"Saved dataloader state to {state_path}")
                
        except Exception as e:
            logger.error(f"Rank {rank}: Failed to save dataloader state: {e}")
            traceback.print_exc()
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
        return control


_UNSUPPORTED_FSYNC_ERRNOS = {
    errno.ENOSYS,
    errno.EINVAL,
    getattr(errno, "ENOTSUP", errno.EINVAL),
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
}


def _checkpoint_fsync(handle, *, path: Path) -> None:
    """Best-effort durability on filesystems that implement regular-file fsync."""

    try:
        os.fsync(handle.fileno())
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_FSYNC_ERRNOS:
            raise
        logger.warning(
            "[Checkpoint] filesystem does not support fsync for %s (%s); "
            "continuing with close + atomic replace",
            path,
            exc,
        )


def publish_cpt_checkpoint_completion(
    checkpoint_dir: Path,
    *,
    global_step: int,
    world_size: int,
    output_dir: Path,
    recovered: bool = False,
) -> dict[str, Any]:
    """Validate, mark, and enqueue one checkpoint in crash-safe order."""

    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    report = validate_checkpoint(
        checkpoint_dir,
        mode="resume",
        expected_ranks=world_size,
    )
    if not report["valid"]:
        raise RuntimeError(
            "checkpoint resume validation failed before marker publish: "
            + "; ".join(report["errors"])
        )

    marker = checkpoint_dir / "checkpoint_complete.json"
    temporary = marker.with_name(f"{marker.name}.tmp-{os.getpid()}")
    payload = {
        "schema_version": 1,
        "global_step": int(global_step),
        "completed_at_unix": time.time(),
        "hostname": socket.gethostname(),
        "world_size": int(world_size),
        "recovered_after_interrupted_publish": bool(recovered),
        "validation": report,
    }
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            _checkpoint_fsync(handle, path=temporary)
        # The marker is published before the queue row. An evaluator can never
        # observe an advertised checkpoint whose resume files were incomplete.
        os.replace(temporary, marker)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logger.exception(
                "[Checkpoint] could not remove marker temporary file %s",
                temporary,
            )

    if _env_flag("LOCANY_CPT_MODE", default=False):
        queue_path = output_dir / "diagnostics" / "cpt_eval_queue.jsonl"
        try:
            enqueue_pending_eval(
                queue_path,
                {
                    "schema_version": 1,
                    "step": int(global_step),
                    "checkpoint": str(checkpoint_dir),
                    "split": "heldout",
                    "recommended_recipe": "locany_cpt_val_fast.json",
                    "status": "pending",
                    "created_at_unix": time.time(),
                },
            )
        except BaseException as exc:
            raise RuntimeError(
                "checkpoint marker was published, but eval queue publish failed: "
                f"queue={queue_path}; {type(exc).__name__}: {exc}"
            ) from exc
    return report


def reconcile_cpt_checkpoint_completion(
    checkpoint_dir: Path,
    *,
    global_step: int,
    world_size: int,
    output_dir: Path,
) -> None:
    """Repair marker/queue publication after a previously interrupted save."""

    error = None
    if get_rank() == 0:
        try:
            report = publish_cpt_checkpoint_completion(
                checkpoint_dir,
                global_step=global_step,
                world_size=world_size,
                output_dir=output_dir,
                recovered=True,
            )
            logger.info(
                "[Checkpoint] reconciled resumable checkpoint before restart: "
                "step=%s path=%s warnings=%s",
                global_step,
                checkpoint_dir,
                report["warnings"],
            )
        except BaseException as exc:
            error = (
                "CPT checkpoint completion recovery failed: "
                f"checkpoint={checkpoint_dir}; {type(exc).__name__}: {exc}"
            )
            logger.exception("[Checkpoint] restart reconciliation FAILED")
    if dist.is_available() and dist.is_initialized():
        message = [error]
        dist.broadcast_object_list(message, src=0)
        error = message[0]
    if error is not None:
        raise RuntimeError(error)


class CheckpointCompletionCallback(TrainerCallback):
    """Validate every rank's resume state before declaring a save complete."""

    def on_save(self, args, state, control, **kwargs):
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        rank = get_rank()
        checkpoint_dir = osp.join(args.output_dir, f"checkpoint-{state.global_step}")
        validation_error = None
        if rank == 0:
            try:
                report = publish_cpt_checkpoint_completion(
                    Path(checkpoint_dir),
                    global_step=int(state.global_step),
                    world_size=(dist.get_world_size() if dist.is_initialized() else 1),
                    output_dir=Path(args.output_dir),
                )
                logger.info(
                    "[Checkpoint] COMPLETE step=%s path=%s details=%s warnings=%s",
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
        "loss_box_l1",
        "loss_box_giou",
        "loss_attn_kl",
        "loss_coverage",
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
        "box_anchor_count",
        "pbd_to_hidden_ratio",
        "coarse_iou_mean",
        "coarse_recall_03",
        "coarse_recall_05",
        "matched_slots",
        "unmatched_slots",
        "slot_usage_entropy",
        "unique_slot_count",
        "duplicate_slot_rate",
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
        "coarse_box_grad_norm",
        "coord_bridge_grad_norm",
        "coarse_box_grad_seen_steps",
        "coord_bridge_grad_seen_steps",
        "coarse_box_absolute_update_norm",
        "coarse_box_relative_update_norm",
        "coarse_box_changed_element_count",
        "coord_bridge_absolute_update_norm",
        "coord_bridge_relative_update_norm",
        "coord_bridge_changed_element_count",
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
        self._cpt_enabled = _env_flag("LOCANY_CPT_MODE", default=False)
        self._cpt_metrics_interval = int(
            os.environ.get("CPT_METRICS_INTERVAL", "100")
        )
        self._cpt_table_interval = int(
            os.environ.get("CPT_TABLE_INTERVAL", "500")
        )
        if self._cpt_enabled and (
            self._cpt_metrics_interval <= 0 or self._cpt_table_interval <= 0
        ):
            raise ValueError("CPT metrics/table intervals must be positive")
        self._cpt_last_written_step = -1
        self._cpt_last_numeric = None
        self._cpt_unique_cache = {}
        self._cpt_seen_oversize_record_hashes = {
            task: set() for task in CPT_TASKS
        }
        self._cpt_seen_oversize_group_hashes = {
            task: set() for task in CPT_TASKS
        }
        self._cpt_started_at = time.time()
        self._cpt_last_write_time = self._cpt_started_at
        self._cpt_metrics_path = osp.join(
            self.args.output_dir, "diagnostics", "cpt_train_metrics.jsonl"
        )
        if self._cpt_enabled:
            self._load_cpt_metrics_baseline()
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
        if self._cpt_enabled and self.is_world_process_zero():
            self._write_cpt_run_config()

    def _write_cpt_run_config(self):
        diagnostics = osp.join(self.args.output_dir, "diagnostics")
        os.makedirs(diagnostics, exist_ok=True)
        datasets = []
        if isinstance(self.train_dataset, StreamPackedDatasetMTP):
            for dataset, probability in zip(
                self.train_dataset.datasets, self.train_dataset.dataset_weight
            ):
                datasets.append(
                    {
                        "task": dataset.cpt_task,
                        "rows": len(dataset),
                        "groups": dataset.cpt_dataset_groups or None,
                        "probability": probability,
                    }
                )
        payload = {
            "schema_version": 1,
            "run_name": self.args.run_name,
            "output_dir": self.args.output_dir,
            "world_size": int(self.args.world_size),
            "seed": int(self.args.seed),
            "max_steps": int(self.args.max_steps),
            "learning_rate": float(self.args.learning_rate),
            "max_num_tokens": self._max_num_tokens,
            "max_num_tokens_per_sample": getattr(
                self.train_dataset, "max_num_tokens_per_sample", None
            ),
            "packing_buffer_size": getattr(self.train_dataset, "buffer_size", None),
            "metrics_interval": self._cpt_metrics_interval,
            "table_interval": self._cpt_table_interval,
            "sampling": copy.deepcopy(
                getattr(self.train_dataset, "cpt_sampling_config", None)
            ),
            "datasets": datasets,
        }
        path = osp.join(diagnostics, "cpt_run_config.json")
        if osp.isfile(path):
            with open(path, "r", encoding="utf-8") as handle:
                previous = json.load(handle)
            previous_sampling = (previous.get("sampling") or {}).get("config_hash")
            current_sampling = (payload.get("sampling") or {}).get("config_hash")
            immutable_fields = {
                "sampling_config_hash": (previous_sampling, current_sampling),
                "max_num_tokens": (
                    previous.get("max_num_tokens"),
                    payload.get("max_num_tokens"),
                ),
                "max_num_tokens_per_sample": (
                    previous.get("max_num_tokens_per_sample"),
                    payload.get("max_num_tokens_per_sample"),
                ),
                "packing_buffer_size": (
                    previous.get("packing_buffer_size"),
                    payload.get("packing_buffer_size"),
                ),
                "world_size": (
                    previous.get("world_size"),
                    payload.get("world_size"),
                ),
                "datasets": (previous.get("datasets"), payload.get("datasets")),
            }
            changed = {
                key: {"previous": left, "current": right}
                for key, (left, right) in immutable_fields.items()
                if left != right
            }
            if changed:
                raise RuntimeError(
                    "CPT immutable run configuration changed in an existing output "
                    f"directory: {changed}"
                )
            return
        temporary = f"{path}.tmp-{os.getpid()}"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _load_cpt_metrics_baseline(self):
        """Restore rolling-window baselines without changing lifetime counters."""
        if not osp.isfile(self._cpt_metrics_path):
            return
        latest = {}
        try:
            with open(self._cpt_metrics_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    task = row.get("task")
                    if task in CPT_TASK_TO_ID:
                        self._cpt_seen_oversize_record_hashes[task].update(
                            int(value)
                            for value in row.get("window_oversize_record_hashes", [])
                        )
                        self._cpt_seen_oversize_group_hashes[task].update(
                            int(value)
                            for value in row.get("window_oversize_group_hashes", [])
                        )
                    if task in CPT_TASK_TO_ID and (
                        task not in latest
                        or int(row.get("step") or -1)
                        >= int(latest[task].get("step") or -1)
                    ):
                        latest[task] = row
            if not latest:
                return
            numeric_keys = {
                key
                for key, value in empty_task_counter().items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            self._cpt_last_numeric = {
                task: {
                    key: row[key]
                    for key in numeric_keys
                    if isinstance(row.get(key), (int, float))
                }
                for task, row in latest.items()
            }
            last_step = max(int(row.get("step") or -1) for row in latest.values())
            self._cpt_last_written_step = last_step
            self._cpt_unique_cache = {
                task: (
                    int(row["unique_record_count"]),
                    int(row["unique_group_count"]),
                )
                for task, row in latest.items()
                if isinstance(row.get("unique_record_count"), (int, float))
                and isinstance(row.get("unique_group_count"), (int, float))
            }
            logger.info("Restored CPT metric baseline at step %s", last_step)
        except Exception as exc:
            raise RuntimeError(
                f"Invalid CPT metrics history at {self._cpt_metrics_path}: {exc}"
            ) from exc

    def _global_cpt_counters(self, include_unique: bool) -> dict[str, dict]:
        if not isinstance(self.train_dataset, StreamPackedDatasetMTP):
            return {task: empty_task_counter() for task in CPT_TASKS}
        local = self.train_dataset.local_cpt_counters()
        payload = {}
        for task, counter in local.items():
            value = serializable_counter(counter)
            if not include_unique:
                value["unique_record_hashes"] = []
                value["unique_group_hashes"] = []
            payload[task] = value
        gathered = [payload]
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, payload)
        return {
            task: merge_task_counters(
                rank_payload[task] for rank_payload in gathered
            )
            for task in CPT_TASKS
        }

    @staticmethod
    def _numeric_delta(current: Mapping[str, Any], previous: Mapping[str, Any]):
        output = {}
        for key, value in current.items():
            if isinstance(value, (int, float)) and isinstance(
                previous.get(key), (int, float)
            ):
                delta = value - previous[key]
                output[key] = delta if delta >= 0 else None
        return output

    def _write_cpt_metrics(self, step: int, logs: Mapping[str, Any]):
        include_unique = step % self._cpt_table_interval == 0
        counters = self._global_cpt_counters(include_unique=include_unique)
        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated() / (1024.0 ** 2)
            if torch.cuda.is_available()
            else 0.0
        )
        if dist.is_available() and dist.is_initialized():
            peak_tensor = torch.tensor(
                peak_gpu_memory_mb,
                dtype=torch.float64,
                device=(torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")),
            )
            dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
            peak_gpu_memory_mb = float(peak_tensor.item())
        local_physical = (
            sum(
                int(state.get("batches_produced", 0))
                for state in self.train_dataset._worker_states.values()
            )
            if isinstance(self.train_dataset, StreamPackedDatasetMTP)
            else 0
        )
        physical_tensor = torch.tensor(
            local_physical,
            dtype=torch.long,
            device=(torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")),
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(physical_tensor, op=dist.ReduceOp.SUM)
        physical_packed_batches = int(physical_tensor.item())
        if not self.is_world_process_zero():
            return {}
        os.makedirs(osp.dirname(self._cpt_metrics_path), exist_ok=True)
        dataset_info = {
            dataset.cpt_task: (len(dataset), dataset.cpt_dataset_groups or None)
            for dataset in self.train_dataset.datasets
        }
        if include_unique:
            self._cpt_unique_cache = {
                task: (
                    len(counter["unique_record_hashes"]),
                    len(counter["unique_group_hashes"]),
                )
                for task, counter in counters.items()
            }
        total_trained = sum(counter["trained_samples"] for counter in counters.values())
        total_tokens = sum(
            counter["total_supervised_tokens"] for counter in counters.values()
        )
        total_main_tokens = sum(
            counter["main_supervised_tokens"] for counter in counters.values()
        )
        total_mtp_tokens = sum(
            counter["mtp_supervised_tokens"] for counter in counters.values()
        )
        global_attempted = sum(
            counter["attempted_samples"] for counter in counters.values()
        )
        global_accepted = sum(
            counter["accepted_samples"] for counter in counters.values()
        )
        global_skipped = sum(
            counter["oversize_skipped_samples"] for counter in counters.values()
        )
        global_main_loss_sum = sum(
            counter["main_loss_sum"] for counter in counters.values()
        )
        global_main_loss_tokens = sum(
            counter["main_loss_tokens"] for counter in counters.values()
        )
        global_mtp_loss_sum = sum(
            counter["mtp_loss_sum"] for counter in counters.values()
        )
        global_mtp_loss_tokens = sum(
            counter["mtp_loss_tokens"] for counter in counters.values()
        )
        # A mixed-task packed batch is counted once for each task it contains,
        # so the task counters cannot recover the number of physical batches.
        # Use the exact trained sample stream's per-rank dataloader snapshots.
        global_packing_efficiency = (
            sum(counter["packed_tokens"] for counter in counters.values())
            / (physical_packed_batches * self._max_num_tokens)
            if physical_packed_batches and self._max_num_tokens
            else None
        )
        now = time.time()
        window_seconds = max(now - self._cpt_last_write_time, 1.0e-9)
        previous = self._cpt_last_numeric or {}
        rows = []
        new_oversize_hashes = {}
        for task in CPT_TASKS:
            dataset_rows, dataset_groups = dataset_info.get(task, (0, None))
            summary = summarize_task_counter(
                task,
                counters[task],
                dataset_rows=dataset_rows,
                dataset_groups=dataset_groups,
            )
            unique = self._cpt_unique_cache.get(task)
            if not include_unique and unique is None:
                for key in (
                    "unique_record_count",
                    "unique_group_count",
                    "row_coverage",
                    "group_coverage",
                    "repeat_factor",
                ):
                    summary[key] = None
            elif unique is not None:
                summary["unique_record_count"], summary["unique_group_count"] = unique
                summary["row_coverage"] = unique[0] / dataset_rows if dataset_rows else None
                summary["group_coverage"] = (
                    unique[1] / dataset_groups if dataset_groups else None
                )
                summary["repeat_factor"] = summary["trained_samples"] / max(unique[0], 1)
            sample_share = summary["trained_samples"] / total_trained if total_trained else 0.0
            token_share = summary["total_supervised_tokens"] / total_tokens if total_tokens else 0.0
            dominance = token_share / sample_share if sample_share else None
            window = (
                self._numeric_delta(counters[task], previous.get(task, {}))
                if task in previous
                else None
            )
            window_oversize_records = sorted(
                set(counters[task]["oversize_record_hashes"])
                - self._cpt_seen_oversize_record_hashes[task]
            )
            window_oversize_groups = sorted(
                set(counters[task]["oversize_group_hashes"])
                - self._cpt_seen_oversize_group_hashes[task]
            )
            new_oversize_hashes[task] = (
                window_oversize_records,
                window_oversize_groups,
            )
            row = {
                "schema_version": 1,
                "scope": "lifetime_global",
                "step": step,
                "epoch": self.state.epoch,
                "task": task,
                "learning_rate": logs.get("learning_rate"),
                "global_loss": logs.get("loss"),
                "global_attempted_samples": global_attempted,
                "global_accepted_samples": global_accepted,
                "global_trained_samples": total_trained,
                "global_oversize_skipped_samples": global_skipped,
                "global_main_supervised_tokens": total_main_tokens,
                "global_mtp_supervised_tokens": total_mtp_tokens,
                "global_total_supervised_tokens": total_tokens,
                "global_train_main_token_ce": (
                    global_main_loss_sum / global_main_loss_tokens
                    if global_main_loss_tokens
                    else None
                ),
                "global_train_mtp_token_ce": (
                    global_mtp_loss_sum / global_mtp_loss_tokens
                    if global_mtp_loss_tokens
                    else None
                ),
                "global_train_total_token_ce": (
                    (global_main_loss_sum + global_mtp_loss_sum)
                    / (global_main_loss_tokens + global_mtp_loss_tokens)
                    if global_main_loss_tokens + global_mtp_loss_tokens
                    else None
                ),
                "sample_share": sample_share,
                "main_token_share": (
                    summary["main_supervised_tokens"] / total_main_tokens
                    if total_main_tokens
                    else 0.0
                ),
                "mtp_token_share": (
                    summary["mtp_supervised_tokens"] / total_mtp_tokens
                    if total_mtp_tokens
                    else 0.0
                ),
                "total_token_share": token_share,
                "token_dominance_ratio": dominance,
                "token_dominant": dominance is not None and dominance > 2.0,
                **summary,
                "packing_efficiency": global_packing_efficiency,
                "task_conditional_packing_efficiency": (
                    summary["packed_tokens"]
                    / (summary["packed_batches"] * self._max_num_tokens)
                    if summary["packed_batches"] and self._max_num_tokens
                    else None
                ),
                "window_seconds": window_seconds,
                "samples_per_second": (
                    window.get("trained_samples", 0) / window_seconds
                    if window is not None
                    else None
                ),
                "supervised_tokens_per_second": (
                    window.get("total_supervised_tokens", 0) / window_seconds
                    if window is not None
                    else None
                ),
                "packed_tokens_per_second": (
                    window.get("packed_tokens", 0) / window_seconds
                    if window is not None
                    else None
                ),
                "peak_gpu_memory_mb": peak_gpu_memory_mb or None,
                "window": window,
                "window_oversize_record_hashes": window_oversize_records,
                "window_oversize_group_hashes": window_oversize_groups,
            }
            rows.append(row)
        with open(self._cpt_metrics_path, "a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        for task, (record_hashes, group_hashes) in new_oversize_hashes.items():
            self._cpt_seen_oversize_record_hashes[task].update(record_hashes)
            self._cpt_seen_oversize_group_hashes[task].update(group_hashes)
        self._cpt_last_numeric = {
            task: {
                key: value
                for key, value in counter.items()
                if isinstance(value, (int, float))
            }
            for task, counter in counters.items()
        }
        self._cpt_last_write_time = now
        if global_attempted and global_skipped / global_attempted > 0.005:
            logger.warning(
                "[CPT] global oversize skip rate %.3f%% exceeds 0.5%%",
                global_skipped / global_attempted * 100.0,
            )
        for row in rows:
            if row["oversize_skip_rate"] > 0.02:
                logger.warning(
                    "[CPT] task=%s oversize skip rate %.3f%% p95=%s p99=%s max=%s",
                    row["task"],
                    row["oversize_skip_rate"] * 100.0,
                    row["oversize_p95_post_mtp_length"],
                    row["oversize_p99_post_mtp_length"],
                    row["oversize_max_post_mtp_length"],
                )
        if include_unique:
            logger.info(
                "[CPT metrics step=%s]\n%s",
                step,
                "\n".join(
                    f"{row['task']:20s} trained={row['trained_samples']:8d} "
                    f"skip={row['oversize_skip_rate']:.3%} "
                    f"tokens={row['total_supervised_tokens']:10d} "
                    f"CE={row['train_total_token_ce']} "
                    f"coverage={row['row_coverage']} repeat={row['repeat_factor']}"
                    for row in rows
                ),
            )
            # Excel is an optional projection.  The helper imports openpyxl
            # lazily and converts every failure into a warning.
            from eaglevl.train.cpt_excel import build_cpt_workbook

            build_cpt_workbook(osp.dirname(self._cpt_metrics_path))
        tracker_metrics = {}
        for row in rows:
            task = row["task"]
            for key in (
                "sample_share",
                "main_token_share",
                "mtp_token_share",
                "total_token_share",
                "oversize_skip_rate",
                "train_main_token_ce",
                "train_mtp_token_ce",
                "train_total_token_ce",
                "effective_epoch",
                "repeat_factor",
                "samples_per_second",
                "supervised_tokens_per_second",
                "packing_efficiency",
                "attempted_samples",
                "accepted_samples",
                "trained_samples",
                "oversize_skipped_samples",
                "main_supervised_tokens",
                "mtp_supervised_tokens",
                "total_supervised_tokens",
                "unique_record_count",
                "unique_group_count",
                "row_coverage",
                "group_coverage",
                "avg_post_mtp_length",
                "p95_post_mtp_length",
                "token_dominance_ratio",
                "packed_tokens_per_second",
                "peak_gpu_memory_mb",
            ):
                value = row.get(key)
                if isinstance(value, (int, float)):
                    tracker_metrics[f"cpt/{task}/{key}"] = value
        tracker_metrics.update(
            {
                "cpt/global/attempted_samples": global_attempted,
                "cpt/global/accepted_samples": global_accepted,
                "cpt/global/trained_samples": total_trained,
                "cpt/global/oversize_skipped_samples": global_skipped,
                "cpt/global/oversize_skip_rate": (
                    global_skipped / global_attempted if global_attempted else 0.0
                ),
                "cpt/global/main_supervised_tokens": total_main_tokens,
                "cpt/global/mtp_supervised_tokens": total_mtp_tokens,
                "cpt/global/total_supervised_tokens": total_tokens,
                "cpt/global/packing_efficiency": global_packing_efficiency,
                "cpt/global/peak_gpu_memory_mb": peak_gpu_memory_mb or None,
                "cpt/global/train_main_token_ce": (
                    global_main_loss_sum / global_main_loss_tokens
                    if global_main_loss_tokens
                    else None
                ),
                "cpt/global/train_mtp_token_ce": (
                    global_mtp_loss_sum / global_mtp_loss_tokens
                    if global_mtp_loss_tokens
                    else None
                ),
                "cpt/global/train_total_token_ce": (
                    (global_main_loss_sum + global_mtp_loss_sum)
                    / (global_main_loss_tokens + global_mtp_loss_tokens)
                    if global_main_loss_tokens + global_mtp_loss_tokens
                    else None
                ),
                "cpt/global/samples_per_second": sum(
                    float(row.get("samples_per_second") or 0.0) for row in rows
                ),
                "cpt/global/supervised_tokens_per_second": sum(
                    float(row.get("supervised_tokens_per_second") or 0.0)
                    for row in rows
                ),
                "cpt/global/packed_tokens_per_second": sum(
                    float(row.get("packed_tokens_per_second") or 0.0)
                    for row in rows
                ),
            }
        )
        tracker_metrics = {
            key: value
            for key, value in tracker_metrics.items()
            if isinstance(value, (int, float))
        }
        return tracker_metrics

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
                    temporary_target = f"{target_text}.tmp-{os.getpid()}"
                    result = original_torch_save(
                        obj, temporary_target, *args, **kwargs
                    )
                    with open(temporary_target, "rb+") as handle:
                        os.fsync(handle.fileno())
                    os.replace(temporary_target, target_text)
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
        if "relation_pbd.coord_prior_lambda" in name:
            return "coord_bridge"
        if "relation_pyramid.coarse_box_head" in name:
            return "coarse_box"
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
        for group in ("relation", "image_gate", "slot_gate", "pbd", "coarse_box", "coord_bridge"):
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
        for group_name in ("relation", "image_gate", "slot_gate", "pbd", "coarse_box", "coord_bridge"):
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
        for output_name, metric_name in (
            ("box_l1_loss", "loss_box_l1"),
            ("box_giou_loss", "loss_box_giou"),
            ("attention_kl_loss", "loss_attn_kl"),
            ("coverage_loss", "loss_coverage"),
        ):
            self._add_ui5_scalar(metric_name, getattr(outputs, output_name, None))
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
        self._add_ui5_scalar(
            "box_anchor_count", getattr(outputs, "box_anchor_count", None)
        )
        for output_name, metric_name in (
            ("pbd_to_hidden_ratio", "pbd_to_hidden_ratio"),
            ("coarse_iou_mean", "coarse_iou_mean"),
            ("coarse_recall_03", "coarse_recall_03"),
            ("coarse_recall_05", "coarse_recall_05"),
            ("matched_slots", "matched_slots"),
            ("unmatched_slots", "unmatched_slots"),
            ("slot_usage_entropy", "slot_usage_entropy"),
            ("unique_slot_count", "unique_slot_count"),
            ("duplicate_slot_rate", "duplicate_slot_rate"),
        ):
            self._add_ui5_scalar(metric_name, getattr(outputs, output_name, None))

        if not self._ui5_real_data_audit_logged and torch.is_tensor(detail_norm):
            detail_weights_for_audit = getattr(outputs, "detail_layer_weights", None)
            if torch.is_tensor(detail_weights_for_audit):
                weight_sums = detail_weights_for_audit.detach().float().sum(dim=-1)
                if not bool(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1.0e-5)):
                    raise RuntimeError(
                        f"Detail Pyramid scale weights do not sum to one: {weight_sums.tolist()}"
                    )
                if int(self.state.global_step) == 0:
                    relation_family_for_audit = inputs.get("relation_family")
                    if torch.is_tensor(relation_family_for_audit):
                        families = relation_family_for_audit.detach().reshape(-1).long().clamp(
                            0, self.model.relation_pyramid.family_scale_prior.shape[0] - 1
                        )
                        expected_prior = self.model.relation_pyramid.family_scale_prior[
                            families[: detail_weights_for_audit.shape[0]]
                        ].to(
                            device=detail_weights_for_audit.device,
                            dtype=torch.float32,
                        )
                        actual_prior = detail_weights_for_audit.detach().float()[
                            : expected_prior.shape[0]
                        ]
                        if not bool(
                            torch.allclose(actual_prior, expected_prior, atol=1.0e-4, rtol=0.0)
                        ):
                            raise RuntimeError(
                                "Initial Detail Pyramid scale weights do not preserve family prior: "
                                f"actual={actual_prior.cpu().tolist()}, "
                                f"expected={expected_prior.cpu().tolist()}"
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
                    task_entropy = -(
                        task_weights * task_weights.clamp_min(1.0e-7).log()
                    ).sum(dim=-1)
                    values["scale_entropy_sum"] += float(task_entropy.sum().item())
                    values["scale_entropy_count"] += task_count
                    task_std = task_weights.std(dim=0, unbiased=False)
                    for level, std_value in zip(("l5", "l15", "l26"), task_std):
                        values[f"scale_batch_std_{level}_sum"] += float(std_value.item())
                        values[f"scale_batch_std_{level}_count"] += 1.0
            coord_lambdas = getattr(outputs, "coord_prior_lambdas", None)
            if torch.is_tensor(coord_lambdas) and defect_id < coord_lambdas.numel():
                values["coord_prior_lambda_sum"] += float(coord_lambdas[defect_id].detach().float().item())
                values["coord_prior_lambda_count"] += 1.0
            soft_betas = getattr(outputs, "soft_gate_betas", None)
            if torch.is_tensor(soft_betas) and defect_id < soft_betas.numel():
                values["soft_gate_beta_sum"] += float(soft_betas[defect_id].detach().float().item())
                values["soft_gate_beta_count"] += 1.0

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
        for group in ("relation", "image_gate", "slot_gate", "pbd", "coarse_box", "coord_bridge"):
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
        cpt_task_token_ids = inputs.pop("cpt_task_token_ids", None)
        cpt_supervision_kind = inputs.pop("cpt_supervision_kind", None)
        for key in list(inputs):
            if key.startswith("_sample_"):
                inputs.pop(key)
        labels_for_cpt = inputs.get("labels")
        if self._cpt_enabled and (
            cpt_task_token_ids is None or cpt_supervision_kind is None
        ):
            raise RuntimeError(
                "CPT batch is missing task-token/supervision-kind metadata"
            )
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        if self._cpt_enabled:
            token_losses = getattr(outputs, "cpt_token_losses", None)
            if token_losses is None:
                raise RuntimeError(
                    "CPT observability is enabled but the model returned no per-token CE"
                )
            values = aggregate_token_losses(
                token_losses,
                labels_for_cpt,
                cpt_task_token_ids,
                cpt_supervision_kind,
                ignore_index=IGNORE_TOKEN_ID,
            )
            self.train_dataset.record_cpt_ce(values)
        if self._ui5_enabled:
            self._capture_ui5_batch(outputs, inputs)
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        # 记录开始的step（用于resume时正确计算平均值）
        if self._start_step is None:
            self._start_step = self.state.global_step
        
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
            "scale_entropy_sum", "scale_entropy_count",
            "scale_batch_std_l5_sum", "scale_batch_std_l5_count",
            "scale_batch_std_l15_sum", "scale_batch_std_l15_count",
            "scale_batch_std_l26_sum", "scale_batch_std_l26_count",
            "coord_prior_lambda_sum", "coord_prior_lambda_count",
            "soft_gate_beta_sum", "soft_gate_beta_count",
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
        coarse_box_grad_norm = global_grad_rms("coarse_box")
        coord_bridge_grad_norm = global_grad_rms("coord_bridge")
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
        metrics = {
            "step": step,
            "task_id": "mixed",
            "tc_msed_stage": str(getattr(config, "tc_msed_stage", "v4")),
            "epoch": self.state.epoch,
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
            "loss_box_l1": self._average(scalars["loss_box_l1"]),
            "loss_box_giou": self._average(scalars["loss_box_giou"]),
            "loss_attn_kl": self._average(scalars["loss_attn_kl"]),
            "loss_coverage": self._average(scalars["loss_coverage"]),
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
            "pbd_delta_norm_active": self._average(scalars["pbd_delta_norm"]),
            "pbd_active_positions": scalars["pbd_active_positions"]["sum"],
            "pbd_to_hidden_ratio": self._average(scalars["pbd_to_hidden_ratio"]),
            "coarse_iou_mean": self._average(scalars["coarse_iou_mean"]),
            "coarse_recall_03": self._average(scalars["coarse_recall_03"]),
            "coarse_recall_05": self._average(scalars["coarse_recall_05"]),
            "matched_slots": self._average(scalars["matched_slots"]),
            "unmatched_slots": self._average(scalars["unmatched_slots"]),
            "slot_usage_entropy": self._average(scalars["slot_usage_entropy"]),
            "box_anchor_count": scalars["box_anchor_count"]["sum"],
            "unique_slot_count": self._average(scalars["unique_slot_count"]),
            "duplicate_slot_rate": self._average(scalars["duplicate_slot_rate"]),
            "relation_grad_norm": relation_grad_norm,
            "gate_grad_norm": gate_grad_norm,
            "image_gate_grad_norm": image_gate_grad_norm,
            "slot_gate_grad_norm": slot_gate_grad_norm,
            "pbd_grad_norm": pbd_grad_norm,
            "coarse_box_grad_norm": coarse_box_grad_norm,
            "coord_bridge_grad_norm": coord_bridge_grad_norm,
            "grad_relation": relation_grad_norm,
            "grad_coarse_box": coarse_box_grad_norm,
            "grad_pbd": pbd_grad_norm,
            "grad_coord_bridge": coord_bridge_grad_norm,
            "update_ratio_relation": self._average(
                scalars["relation_relative_update_norm"]
            ),
            "update_ratio_pbd": self._average(
                scalars["pbd_relative_update_norm"]
            ),
            "update_ratio_coord_bridge": self._average(
                scalars["coord_bridge_relative_update_norm"]
            ),
            "tasks": {},
        }
        for group in ("relation", "image_gate", "slot_gate", "pbd", "coarse_box", "coord_bridge"):
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
                "scale_w_l5": (
                    values["detail_weight_l5_sum"] / values["detail_weight_count"]
                    if values["detail_weight_count"] else None
                ),
                "scale_w_l15": (
                    values["detail_weight_l15_sum"] / values["detail_weight_count"]
                    if values["detail_weight_count"] else None
                ),
                "scale_w_l26": (
                    values["detail_weight_l26_sum"] / values["detail_weight_count"]
                    if values["detail_weight_count"] else None
                ),
                "scale_entropy": (
                    values["scale_entropy_sum"] / values["scale_entropy_count"]
                    if values["scale_entropy_count"] else None
                ),
                "scale_batch_std_l5": (
                    values["scale_batch_std_l5_sum"] / values["scale_batch_std_l5_count"]
                    if values["scale_batch_std_l5_count"] else None
                ),
                "scale_batch_std_l15": (
                    values["scale_batch_std_l15_sum"] / values["scale_batch_std_l15_count"]
                    if values["scale_batch_std_l15_count"] else None
                ),
                "scale_batch_std_l26": (
                    values["scale_batch_std_l26_sum"] / values["scale_batch_std_l26_count"]
                    if values["scale_batch_std_l26_count"] else None
                ),
                "coord_prior_lambda": (
                    values["coord_prior_lambda_sum"] / values["coord_prior_lambda_count"]
                    if values["coord_prior_lambda_count"] else None
                ),
                "soft_gate_beta": (
                    values["soft_gate_beta_sum"] / values["soft_gate_beta_count"]
                    if values["soft_gate_beta_count"] else None
                ),
            }
        if self.is_world_process_zero():
            written = self._ui5_excel.update_train(step, metrics)
            logger.info(
                "[UI5Excel] step=%s written=%s path=%s",
                step,
                written,
                self._ui5_excel.path,
            )
        self._reset_ui5_window()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def log(self, logs, start_time=None):
        if self._ui5_enabled and "grad_norm" in logs:
            self._add_ui5_scalar("grad_norm", logs["grad_norm"])
        step = int(self.state.global_step)
        tracker_metrics = {}
        if (
            self._cpt_enabled
            and step > 0
            and step % self._cpt_metrics_interval == 0
            and step != self._cpt_last_written_step
        ):
            # Every rank must participate because this routine performs a
            # distributed all-gather; only rank 0 writes the JSONL.
            tracker_metrics = self._write_cpt_metrics(step, logs)
            self._cpt_last_written_step = step
        combined_logs = dict(logs)
        combined_logs.update(tracker_metrics)
        result = super().log(combined_logs, start_time)
        if self._ui5_enabled and (
            step in {1, 20} or (step > 0 and step % 100 == 0)
        ):
            self._capture_ui5_parameter_updates()
            if step == 20:
                required_groups = {"relation", "image_gate", "slot_gate"}
                if self._ui5_scalar["pbd_active_positions"]["sum"] > 0:
                    required_groups.add("pbd")
                stage = str(getattr(self.model.config, "tc_msed_stage", "v4"))
                if stage in {"m2", "m3", "m4", "m5"}:
                    required_groups.add("coarse_box")
                if stage in {"m3", "m4", "m5"}:
                    required_groups.add("coord_bridge")
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
            if step >= 500 and step % 100 == 0:
                stage = str(getattr(self.model.config, "tc_msed_stage", "v4"))
                required_groups = {"relation", "image_gate", "slot_gate"}
                if self._ui5_scalar["pbd_active_positions"]["sum"] > 0:
                    required_groups.add("pbd")
                if stage in {"m2", "m3", "m4", "m5"}:
                    required_groups.add("coarse_box")
                if stage in {"m3", "m4", "m5"}:
                    required_groups.add("coord_bridge")
                inactive = []
                for group in sorted(required_groups):
                    ratio_state = self._ui5_scalar[
                        f"{group}_relative_update_norm"
                    ]
                    ratio = (
                        ratio_state["sum"] / ratio_state["count"]
                        if ratio_state["count"]
                        else 0.0
                    )
                    if ratio < 1.0e-6:
                        inactive.append((group, ratio))
                if inactive:
                    raise RuntimeError(
                        "TC-MSED parameter update ratio stayed below 1e-6 "
                        f"through optimizer step {step}: {inactive}"
                    )
        if self._ui5_enabled and (
            step > 0
            and step % 100 == 0
            and step != self._ui5_last_flushed_step
        ):
            self._flush_ui5_excel(step, logs)
            self._ui5_last_flushed_step = step
            self._save_ui5_window_state()
        elif self._ui5_enabled and "train_runtime" in logs:
            # Persist a partial 100-step window at a clean segment/final stop.
            self._save_ui5_window_state()
        return result
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        from torch.utils.data import DataLoader

        logical_workers = max(1, int(self.args.dataloader_num_workers))
        if (
            self._cpt_enabled
            and self.train_dataset._saved_num_workers is not None
            and self.train_dataset._saved_num_workers != logical_workers
        ):
            raise RuntimeError(
                "CPT dataloader worker count changed across resume: "
                f"saved={self.train_dataset._saved_num_workers}, current={logical_workers}"
            )
        if isinstance(self.train_dataset, StreamPackedDatasetMTP):
            self.train_dataset._configured_num_workers = logical_workers
        
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
    base_seed: int = 42
) -> StreamPackedDatasetMTP:
    """Build StreamPackedDatasetMTP."""
    ds_collections = json.loads(open(data_args.meta_path).read())
    
    datasets = []
    dataset_weights = []
    cpt_sampling_tasks = []
    
    for ds_name, meta in ds_collections.items():
        meta = resolve_recipe_entry_paths(meta, data_args.meta_path)
        repeat_time = meta.get('repeat_time', 1)
        try:
            ds = LazySupervisedDatasetMTP(
                ds_name, meta, processor,
                block_size=model_args.block_size,
                repeat_time=repeat_time,
                target_fps=data_args.target_fps,
                max_frames=data_args.max_frames,
                video_total_pixels=data_args.video_total_pixels,
                balance_ui_defects=data_args.balance_ui_defects,
                ui_records_per_class=data_args.ui_records_per_class,
                ui_negative_to_positive_ratio=data_args.ui_negative_to_positive_ratio,
            )
            
            if len(ds) == 0:
                logger.warning(f'Dataset {ds_name} is empty, skipping.')
                continue
            
            datasets.append(ds)
            
            weight = resolve_dataset_sampling_weight(meta, len(ds))
            dataset_weights.append(weight)
            cpt_sampling_tasks.append(
                {
                    "name": ds.cpt_task or ds_name,
                    "rows": len(ds),
                    "mean_total_supervised_tokens": meta.get(
                        "mean_total_supervised_tokens"
                    ),
                }
            )
            
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
    cpt_sampling_config = None
    if _env_flag("LOCANY_CPT_MODE", default=False):
        task_names = [dataset.cpt_task for dataset in datasets]
        duplicates = sorted(
            task for task in set(task_names) if task_names.count(task) > 1
        )
        missing = sorted(set(CPT_TASKS).difference(task_names))
        unexpected = sorted(set(task_names).difference(CPT_TASKS))
        if duplicates or missing or unexpected:
            raise ValueError(
                "CPT recipe must contain exactly one dataset for each canonical task: "
                f"duplicates={duplicates}, missing={missing}, unexpected={unexpected}"
            )

        def optional_float(name):
            value = os.environ.get(name)
            return None if value is None or not value.strip() else float(value)

        cpt_sampling_config = resolve_cpt_sampling(
            cpt_sampling_tasks,
            mode=os.environ.get("CPT_SAMPLING_MODE", "sample_equal"),
            size_alpha=optional_float("CPT_SIZE_ALPHA"),
            token_beta=optional_float("CPT_TOKEN_BETA"),
            min_task_prob=float(os.environ.get("CPT_MIN_TASK_PROB", "0")),
            max_task_prob=float(os.environ.get("CPT_MAX_TASK_PROB", "1")),
        )
        probabilities = {
            task["name"]: float(task["probability"])
            for task in cpt_sampling_config["tasks"]
        }
        dataset_weights = [probabilities[dataset.cpt_task] for dataset in datasets]
        logger.warning(
            "[CPT sampling] %s",
            json.dumps(cpt_sampling_config, ensure_ascii=False, sort_keys=True),
        )

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
        cpt_sampling_config=cpt_sampling_config,
    )


@record
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
        if last_checkpoint is not None and _env_flag("LOCANY_CPT_MODE", default=False):
            checkpoint_name = osp.basename(osp.normpath(last_checkpoint))
            checkpoint_step = int(checkpoint_name.rsplit("checkpoint-", 1)[1])
            reconcile_cpt_checkpoint_completion(
                Path(last_checkpoint),
                global_step=checkpoint_step,
                world_size=(dist.get_world_size() if dist.is_initialized() else 1),
                output_dir=Path(training_args.output_dir),
            )
            
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
            tc_msed_stage=model_args.tc_msed_stage,
            relation_box_l1_loss_weight=model_args.relation_box_l1_loss_weight,
            relation_box_giou_loss_weight=model_args.relation_box_giou_loss_weight,
            relation_coverage_loss_weight=model_args.relation_coverage_loss_weight,
            relation_coord_prior_sigma=model_args.relation_coord_prior_sigma,
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
            relation_gate_mode=("soft" if model_args.tc_msed_stage in {"m4", "m5"} else "observe"),
            relation_focal_beta=model_args.relation_focal_beta,
            relation_focal_gamma=model_args.relation_focal_gamma,
            tc_msed_stage=model_args.tc_msed_stage,
            relation_task_scale_router=model_args.tc_msed_stage in {"m1", "m2", "m3", "m4", "m5"},
            relation_set_localizer=model_args.tc_msed_stage in {"m2", "m3", "m4", "m5"},
            relation_dynamic_slot_pbd=model_args.tc_msed_stage in {"m3", "m4", "m5"},
            relation_coordinate_bridge=model_args.tc_msed_stage in {"m3", "m4", "m5"},
            relation_soft_gate=model_args.tc_msed_stage in {"m4", "m5"},
            relation_overlap_adapter=model_args.tc_msed_stage == "m5",
            relation_box_l1_loss_weight=model_args.relation_box_l1_loss_weight,
            relation_box_giou_loss_weight=model_args.relation_box_giou_loss_weight,
            relation_coverage_loss_weight=model_args.relation_coverage_loss_weight,
            relation_coord_prior_sigma=model_args.relation_coord_prior_sigma)
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
    # The fused loss emits detached per-token CE only for CPT diagnostics.
    # Keeping the flag off by default preserves the SFT forward contract and
    # avoids the diagnostic buffer allocation in non-CPT jobs.
    model._cpt_observability_enabled = _env_flag(
        "LOCANY_CPT_MODE", default=False
    )
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
    train_dataset = build_stream_packed_dataset_mtp(model_args, data_args, processor, base_seed=training_args.seed)
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
            SaveCheckpointCallback(
                interval_hours=model_args.save_every_n_hours,
                stop_after_save=_env_flag(
                    "LOCANY_STOP_AFTER_PERIODIC_SAVE", default=False
                ),
            )
        )
    my_callbacks.append(MemoryLoggerCallback())
    my_callbacks.append(DataloaderStateCallback(train_dataset))
    my_callbacks.append(CheckpointCompletionCallback())
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
        
        if checkpoint is not None:
            training_args.ignore_data_skip = True
            logger.info("Enabled ignore_data_skip=True for stateful dataloader resume.")
            
            rank = get_rank()
            dataloader_state_path = os.path.join(checkpoint, f"dataloader_state_rank{rank}.pt")
            
            if os.path.exists(dataloader_state_path):
                try:
                    dataloader_state = torch.load(dataloader_state_path, weights_only=False)
                    train_dataset.load_state_dict(dataloader_state)
                    logger.info(f"Rank {rank}: Loaded dataloader state from {dataloader_state_path}")
                except Exception as e:
                    if _env_flag("LOCANY_CPT_MODE", default=False):
                        raise RuntimeError(
                            f"Rank {rank}: CPT resume requires valid dataloader state: "
                            f"{dataloader_state_path}"
                        ) from e
                    logger.warning(f"Rank {rank}: Failed to load dataloader state: {e}")
                    traceback.print_exc()
            else:
                if _env_flag("LOCANY_CPT_MODE", default=False):
                    raise FileNotFoundError(
                        f"Rank {rank}: CPT resume dataloader state is missing: "
                        f"{dataloader_state_path}"
                    )
                logger.warning(f"Rank {rank}: No dataloader state found at {dataloader_state_path}")

            if _env_flag("LOCANY_CPT_MODE", default=False):
                checkpoint_name = osp.basename(osp.normpath(str(checkpoint)))
                checkpoint_step = (
                    int(checkpoint_name.rsplit("checkpoint-", 1)[1])
                    if "checkpoint-" in checkpoint_name
                    else None
                )
                if (
                    checkpoint_step is not None
                    and trainer._cpt_last_written_step > checkpoint_step
                ):
                    raise RuntimeError(
                        "CPT metrics history is ahead of the resume checkpoint; "
                        f"metrics_step={trainer._cpt_last_written_step}, "
                        f"checkpoint_step={checkpoint_step}. Use a new output directory "
                        "or resume the latest complete checkpoint."
                    )

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        segment_mode = _env_flag("LOCANY_SEGMENT_MODE", default=False)
        final_segment = trainer.state.global_step >= training_args.max_steps
        if not segment_mode or final_segment:
            trainer.save_model()
        elif get_rank() == 0:
            logger.info(
                "Segment mode stopped at step %s; skipped duplicate model export to output root",
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
    training_complete = (not training_args.do_train) or (
        'trainer' in locals() and trainer.state.global_step >= training_args.max_steps
    )
    if not segment_mode or training_complete:
        with open(osp.join(training_args.output_dir, 'done.txt'), 'w') as f:
            f.write('done: ' + time.ctime())
    elif get_rank() == 0:
        logger.info("Segment is resumable but not final; done.txt was not written")


if __name__ == '__main__':
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except BaseException:
                logger.exception("Failed to destroy distributed process group during shutdown")
