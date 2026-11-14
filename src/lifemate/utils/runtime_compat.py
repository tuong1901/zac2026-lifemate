from __future__ import annotations

import re


def torchao_compat_action(version: str | None) -> str:
    """Return whether the optional TorchAO install is safe for the installed PEFT."""
    if version is None:
        return "not_installed"
    match = re.match(r"^(\d+)\.(\d+)", version)
    if not match:
        raise ValueError(f"Could not parse installed torchao version: {version!r}")
    major, minor = map(int, match.groups())
    return "remove" if (major, minor) < (0, 16) else "keep"


def fp16_dtype_argument(model_class, dtype):
    """Select the current Transformers `dtype` option, with old-version fallback."""
    import inspect

    parameters = inspect.signature(model_class.from_pretrained).parameters
    name = "dtype" if "dtype" in parameters else "torch_dtype"
    return {name: dtype}
