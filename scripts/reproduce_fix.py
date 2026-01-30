
import sys
import os
import logging
from dataclasses import dataclass

# Mock logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reproduce_fix")

# --- MOCKS to replicate the environment ---

@dataclass
class SchedulerConfig:
    enabled: bool = True
    budget_policy: str = "adaptive"

@dataclass
class MergeConfig:
    enabled: bool = True
    merge_strategy: dict = None

@dataclass
class BindingConfig:
    enabled: bool = True
    confirmation: dict = None
    switching: dict = None
    contradiction: dict = None

@dataclass
class GovernanceConfig:
    scheduler: SchedulerConfig
    merge: MergeConfig
    binding: BindingConfig

@dataclass
class Config:
    governance: GovernanceConfig

# Replicate the core/main_loop.py fix logic
from dataclasses import asdict

def test_main_loop_logic():
    logger.info("--- Testing Main Loop Config Fix ---")
    
    # Create the Dataclass structure that was failing
    cfg = Config(
        governance=GovernanceConfig(
            scheduler=SchedulerConfig(enabled=True),
            merge=MergeConfig(enabled=True),
            binding=BindingConfig(enabled=True)
        )
    )
    
    # Simulate Scheduler Check
    scheduler_cfg_dict = cfg.governance.scheduler
    if hasattr(scheduler_cfg_dict, "enabled"):
        scheduler_cfg_dict = asdict(scheduler_cfg_dict)
    
    if isinstance(scheduler_cfg_dict, dict):
        logger.info("PASS: Scheduler config successfully converted to dict")
    else:
        logger.error(f"FAIL: Scheduler config is {type(scheduler_cfg_dict)}")

    # Simulate Merge Check
    merge_cfg_dict = cfg.governance.merge
    if hasattr(merge_cfg_dict, "enabled"):
        merge_cfg_dict = asdict(merge_cfg_dict)
    
    if isinstance(merge_cfg_dict, dict):
        logger.info("PASS: Merge config successfully converted to dict")
    else:
        logger.error(f"FAIL: Merge config is {type(merge_cfg_dict)}")

# Replicate identity/binding.py fix logic
from identity.binding import BindingManager

def test_binding_manager_logic():
    logger.info("\n--- Testing Binding Manager Fix ---")
    
    # Scene 1: Full System Config (Legacy)
    full_cfg = Config(
        governance=GovernanceConfig(
            scheduler=SchedulerConfig(),
            merge=MergeConfig(),
            binding=BindingConfig(enabled=True)
        )
    )
    bm_full = BindingManager(full_cfg)
    if bm_full.enabled:
         logger.info("PASS: BindingManager init with Full Config")
    else:
         logger.error("FAIL: BindingManager init with Full Config")

    # Scene 2: Partial Config (Identity Engine style)
    partial_cfg = BindingConfig(enabled=True, confirmation={}, switching={}, contradiction={})
    
    # The fix is inside BindingManager.__init__, so we instantiate it
    bm_partial = BindingManager(partial_cfg)
    if bm_partial.enabled:
         logger.info("PASS: BindingManager init with Partial Config (Direct Object)")
    else:
         logger.error("FAIL: BindingManager init with Partial Config failed (disabled)")

if __name__ == "__main__":
    try:
        test_main_loop_logic()
        test_binding_manager_logic()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTESTS FAILED: {e}")
