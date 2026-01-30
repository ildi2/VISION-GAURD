# chimeric_identity/adapters/__init__.py
# ============================================================================
# CHIMERIC ADAPTERS - Non-Invasive Integration Layer
# ============================================================================
#
# Purpose:
#   Provide read-only wrappers around existing face/gait/source_auth engines.
#   Normalize heterogeneous outputs into unified evidence schemas.
#
# Design Principle:
#   - ZERO modifications to existing subsystems
#   - READ-ONLY access (no state changes in adapted engines)
#   - NORMALIZATION: Convert subsystem outputs → chimeric evidence types
#   - ERROR ISOLATION: Adapter failures don't crash main engines
#
# Adapters:
#   - FaceAdapter: IdentityEngine → FaceEvidence
#   - GaitAdapter: GaitEngine → GaitEvidence
#   - SourceAuthAdapter: SourceAuthEngine → SourceAuthEvidence

__all__ = [
    "FaceAdapter",
    "GaitAdapter",
    "SourceAuthAdapter",
]

# Import adapters for direct access
from .face_adapter import FaceAdapter
from .gait_adapter import GaitAdapter
from .source_auth_adapter import SourceAuthAdapter
