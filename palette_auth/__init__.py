"""palette-auth: Self-contained in-pixel image authentication and recovery."""
from __future__ import annotations

from typing import Any

from .core import (
    VerificationResult,
    inspect,
    render_recovery,
    render_tamper_map,
    sign_image,
    verify_image,
)
from .crypto import (
    generate_keypair,
    load_private_key,
    load_public_key,
    save_private_key,
    save_public_key,
    sign,
    verify,
)

__version__ = "1.2.0"

_AI_EXPORTS = {
    "DualStreamForensicNet": (".ai_detector", "DualStreamForensicNet"),
    "TamperCNN": (".ai_detector", "TamperCNN"),
    "neural_recover_image": (".neural_recovery", "neural_recover_image"),
}


def __getattr__(name: str) -> Any:
    if name in _AI_EXPORTS:
        module_name, attr_name = _AI_EXPORTS[name]
        try:
            import importlib
            mod = importlib.import_module(module_name, package=__package__)
            val = getattr(mod, attr_name)
            globals()[name] = val
            return val
        except ImportError as err:
            raise ImportError(
                f"Optional AI dependency required for '{name}'. "
                f"Install with: pip install 'palette-auth[ai]'"
            ) from err
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "DualStreamForensicNet",
    "TamperCNN",
    "VerificationResult",
    "__version__",
    "generate_keypair",
    "inspect",
    "load_private_key",
    "load_public_key",
    "neural_recover_image",
    "render_recovery",
    "render_tamper_map",
    "save_private_key",
    "save_public_key",
    "sign",
    "sign_image",
    "verify",
    "verify_image",
]
