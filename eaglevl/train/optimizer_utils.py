"""Small optimizer compatibility helpers used by custom trainers."""


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
