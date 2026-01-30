# chimeric_identity/__init__.py
# ============================================================================
# CHIMERIC BIOMETRIC IDENTITY FUSION MODULE
# ============================================================================
#
# SIMPLIFIED ARCHITECTURE (v2.0):
# ===============================
#
#   Face Engine (unchanged)    Gait Engine (unchanged)
#         │                          │
#         │ face_id="p_0007"        │ gait_id="marildo"
#         │ conf=0.85               │ conf=0.60
#         │                          │
#         └──────────┬───────────────┘
#                    │
#            Identity Registry
#            ┌──────────────────┐
#            │ "marildo cani"   │
#            │ face: p_0007     │  ← Maps different IDs
#            │ gait: marildo    │    to same person!
#            └──────────────────┘
#                    │
#            Simple Fusion
#            ┌──────────────────┐
#            │ 0.75×0.85 = 0.64 │  face contribution
#            │ 0.25×0.60 = 0.15 │  gait contribution
#            │ ─────────────────│
#            │ Total: 0.79      │  combined confidence
#            └──────────────────┘
#
# KEY FILES:
# ==========
#   simple_fusion.py     - Core fusion logic (weighted combination)
#   simple_runner.py     - Camera runner with visualization
#   identity_registry.py - Maps face_id ↔ gait_id for same person
#   identity_matcher.py  - Smart matching for enrollment sync
#   check_galleries.py   - Gallery diagnostics
#
# USAGE:
# ======
#   # Run chimeric fusion
#   $env:GAITGUARD_FACE_KEY="your-key"
#   python -m chimeric_identity.simple_runner
#
#   # Sync galleries
#   python -m chimeric_identity.identity_matcher sync
#
# Version: 2.0.0 (Simplified Inheritance-Based Architecture)

__version__ = "2.0.0"
__author__ = "GaitGuard Team"

# ============================================================================
# CORE API
# ============================================================================

# Simple fusion engine
from chimeric_identity.simple_fusion import (
    SimpleFusionEngine,
    FusionResult,
    FusionState,
    FusionWeights,
    FaceInput,
    GaitInput,
    get_fusion_engine,
    format_fusion_result,
    create_face_input_from_decision,
    create_gait_input_from_signal,
)

# Identity registry (maps face_id ↔ gait_id)
from chimeric_identity.identity_registry import (
    IdentityRegistry,
    PersonRecord,
    get_identity_registry,
)

# Simple runner
from chimeric_identity.simple_runner import (
    SimpleChimericRunner,
    SimpleRunnerConfig,
    run_chimeric,
)

# Enrollment integration
from chimeric_identity.enrollment_integration import (
    ChimericEnrollment,
    sync_existing_galleries,
)

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Version
    "__version__",
    
    # Fusion Engine
    "SimpleFusionEngine",
    "FusionResult",
    "FusionState", 
    "FusionWeights",
    "FaceInput",
    "GaitInput",
    "get_fusion_engine",
    "format_fusion_result",
    "create_face_input_from_decision",
    "create_gait_input_from_signal",
    
    # Identity Registry
    "IdentityRegistry",
    "PersonRecord",
    "get_identity_registry",
    
    # Runner
    "SimpleChimericRunner",
    "SimpleRunnerConfig",
    "run_chimeric",
    
    # Enrollment
    "ChimericEnrollment",
    "sync_existing_galleries",
]
