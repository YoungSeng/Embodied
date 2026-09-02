"""Small optimizer compatibility helpers used by custom trainers."""


UI_RELATION_PARAMETER_MARKERS = (
    "relation_pyramid.",
    "relation_pbd.",
)


def is_ui_relation_parameter(name: str) -> bool:
    """Return whether a parameter belongs to the M32-only relation modules."""

    return any(marker in str(name) for marker in UI_RELATION_PARAMETER_MARKERS)


def two_learning_rate_parameter_groups(
    named_parameters,
    *,
    decay_parameter_names,
    inherited_learning_rate: float,
    ui_relation_learning_rate: float,
    weight_decay: float,
):
    """Build decay/no-decay groups while preserving exactly two LR families."""

    buckets = {
        ("cpt_inherited", True): [],
        ("cpt_inherited", False): [],
        ("ui_relation", True): [],
        ("ui_relation", False): [],
    }
    decay_parameter_names = set(decay_parameter_names)
    for name, parameter in named_parameters:
        if not getattr(parameter, "requires_grad", False):
            continue
        family = "ui_relation" if is_ui_relation_parameter(name) else "cpt_inherited"
        buckets[(family, name in decay_parameter_names)].append(parameter)

    learning_rates = {
        "cpt_inherited": float(inherited_learning_rate),
        "ui_relation": float(ui_relation_learning_rate),
    }
    groups = []
    for (family, apply_decay), parameters in buckets.items():
        if not parameters:
            continue
        groups.append(
            {
                "params": parameters,
                "lr": learning_rates[family],
                "weight_decay": float(weight_decay) if apply_decay else 0.0,
                "ui5_lr_group": family,
            }
        )
    return groups


def optimizer_learning_rates(optimizer):
    """Read current scheduler-adjusted LRs from a real or wrapped optimizer."""

    current = optimizer
    for _ in range(4):
        groups = getattr(current, "param_groups", None)
        if groups is not None:
            result = {}
            for group in groups:
                family = group.get("ui5_lr_group") if isinstance(group, dict) else None
                if family in {"cpt_inherited", "ui_relation"}:
                    result[family] = float(group["lr"])
            return result
        current = getattr(current, "optimizer", None)
        if current is None:
            break
    return {}


def optimizer_parameters(optimizer):
    """Return parameters from a PyTorch optimizer or Accelerate DummyOptim.

    PyTorch optimizers expose ``param_groups``.  When an optimizer is declared
    in the DeepSpeed config, Accelerate instead gives Trainer a ``DummyOptim``
    whose original parameter groups are stored in ``params``.
    """
    groups = getattr(optimizer, "param_groups", None)
    if groups is None:
        groups = getattr(optimizer, "params", None)
        if groups is None:
            raise TypeError(
                f"Unsupported optimizer {type(optimizer).__name__}: "
                "expected 'param_groups' or 'params'"
            )

        # Preserve one-shot iterators because DeepSpeed still needs to consume
        # DummyOptim.params after Trainer creates the optimizer and scheduler.
        if not isinstance(groups, (list, tuple)):
            groups = list(groups)
            optimizer.params = groups

    parameters = []
    for group in groups:
        if not isinstance(group, dict):
            parameters.append(group)
            continue

        group_parameters = group.get("params", ())
        if not isinstance(group_parameters, (list, tuple)):
            group_parameters = list(group_parameters)
            group["params"] = group_parameters
        parameters.extend(group_parameters)
    return parameters
