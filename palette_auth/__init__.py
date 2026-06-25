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

