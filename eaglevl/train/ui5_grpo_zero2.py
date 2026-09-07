"""Order ZeRO-2 gradient buckets independently of each rank's autograd graph.

ZeRO-2 fills IPG buckets as leaf hooks arrive. Equal slot counts and zero
dependencies ensure participation, but do not order those hooks. Mixed AR,
MTP replay and padding graphs can therefore send different collectives.

Queue readiness and pass gradients to the ORIGINAL ZeRO reducer in a fixed
parameter order. It still owns communication, partitioning, accumulation,
overflow, clipping and optimizer state. No extra gradient copies or graphs.
"""
from eaglevl.train.ui5_grpo_core import digest


class OrderedZero2Reduction:
    VERSION = "ordered-zero2-ipg-v1"

    def __init__(self, optimizer, named_parameters):
        if not getattr(optimizer, "partition_gradients", False):
            raise ValueError("ordered reduction requires ZeRO-2 gradient partitioning")
        if (getattr(optimizer, "overlap_comm", True)
                or getattr(optimizer, "cpu_offload", True)
                or getattr(optimizer, "use_grad_accum_attribute", True)
                or not getattr(optimizer, "contiguous_gradients", False)):
            raise ValueError("ordered ZeRO-2 requires contiguous gradients, no overlap/offload/grad_accum attribute")
        if getattr(optimizer, "_ui5_ordered_reduction", None) is not None:
            raise ValueError("ordered ZeRO-2 reducer already installed")
        self.optimizer = optimizer
        self._reduce = optimizer.reduce_ready_partitions_and_remove_grads
        self._epilogue = optimizer.overlapping_partition_gradients_reduce_epilogue
        if not callable(self._reduce) or not callable(self._epilogue):
            raise ValueError("unsupported ZeRO-2 gradient reduction API")
        names = {id(p): name for name, p in named_parameters if p.requires_grad}
        # Reverse order usually follows decoder backprop, releasing ready
        # gradients promptly. Both ranks derive it from the same optimizer.
        self.parameters = [(i, p) for i, group in enumerate(optimizer.bit16_groups)
                           for p in group if p.requires_grad][::-1]
        self.indices = {id(p): j for j, (_, p) in enumerate(self.parameters)}
        if not names or set(names) != set(self.indices) or len(self.indices) != len(self.parameters):
            raise ValueError("ZeRO-2 parameter inventory differs from the actor's trainable parameters")
        inventory = [dict(name=names[id(p)], group=i, shape=list(p.shape),
                          numel=p.numel(), dtype=str(p.dtype)) for i, p in self.parameters]
        self.plan = dict(version=self.VERSION, order="reverse_optimizer_parameters",
                         reduce_bucket_size=optimizer.reduce_bucket_size,
                         reduce_scatter=optimizer.reduce_scatter,
                         communication_dtype=str(optimizer.communication_data_type),
                         parameter_count=len(inventory), parameters=inventory)
        self.plan["identity"] = digest(self.plan)
        self.active = False
        self.closed = True
        self.cursor = 0
        self.step = self.slot = None
        self.pending, self.seen = set(), set()
        optimizer.reduce_ready_partitions_and_remove_grads = self._ready
        optimizer.overlapping_partition_gradients_reduce_epilogue = self._finish_reduction
        optimizer._ui5_ordered_reduction = self

    def begin_slot(self, step, slot):
        if self.active or not self.closed:
            raise RuntimeError("previous ZeRO-2 slot did not finish")
        self.step, self.slot = step, slot
        self.cursor = 0
        self.pending, self.seen = set(), set()
        self.pending_elements = self.peak_pending_elements = 0
        self.active, self.closed = True, False

    def _ready(self, parameter, group_index):
        if not self.active:
            raise RuntimeError("ZeRO-2 gradient hook outside an aligned GRPO slot")
        index = self.indices.get(id(parameter))
        if index is None or self.parameters[index][0] != group_index or index in self.seen:
            raise RuntimeError("unknown or repeated parameter gradient in ZeRO-2 slot")
        if self.optimizer.get_gradient_for_reduction(parameter) is None:
            raise RuntimeError("ZeRO-2 ready hook has no gradient")
        if not getattr(parameter, "ds_grad_is_ready", True):
            raise RuntimeError("transient partial gradients are unsupported in a GRPO completion slot")
        self.pending.add(index)
        self.seen.add(index)
        self.pending_elements += parameter.numel()
        self.peak_pending_elements = max(self.peak_pending_elements, self.pending_elements)
        while self.cursor in self.pending:
            group, ready = self.parameters[self.cursor]
            self._reduce(ready, group)
            self.pending.remove(self.cursor)
            self.pending_elements -= ready.numel()
            self.cursor += 1

    def _finish_reduction(self):
        if not self.active or self.cursor != len(self.parameters) or self.pending:
            missing = [self.plan["parameters"][i]["name"]
                       for i in range(len(self.parameters)) if i not in self.seen]
            raise RuntimeError(f"incomplete ZeRO-2 slot before reduction epilogue: "
                               f"step={self.step} slot={self.slot} missing={missing[:12]}")
        # The final bucket and partitioned accumulation are native DeepSpeed.
        self._epilogue()
        self.active = False

    def end_slot(self):
        if self.active or self.closed:
            raise RuntimeError("ZeRO-2 slot requires exactly one reduction epilogue")
        self.closed = True
        return dict(reduced_parameters=self.cursor, peak_pending_elements=self.peak_pending_elements)
