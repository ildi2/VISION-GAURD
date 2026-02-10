#!/usr/bin/env python3

import argparse
import logging
import os
import shutil
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
ROLLOUT_DIR = PROJECT_ROOT / "scripts" / "rollout"
BACKUP_DIR = PROJECT_ROOT / "config" / "backups"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"

STAGES = {
    1: {
        "name": "Shadow Mode",
        "file": "stage1_shadow.yaml",
        "duration": "1 week",
        "risk": "Zero (observe-only)",
        "description": "Enable shadow mode for observation and threshold tuning"
    },
    2: {
        "name": "Strict Guards",
        "file": "stage2_strict.yaml",
        "duration": "1 week",
        "risk": "Low (conservative thresholds)",
        "description": "Enable real mode with strict guards"
    },
    3: {
        "name": "Relaxed Guards",
        "file": "stage3_relaxed.yaml",
        "duration": "1 week",
        "risk": "Medium (production thresholds)",
        "description": "Relax to production thresholds"
    },
    4: {
        "name": "Grace Reattachment",
        "file": "stage4_grace.yaml",
        "duration": "1 week",
        "risk": "Medium (new feature)",
        "description": "Enable grace reattachment"
    },
    5: {
        "name": "Full Production",
        "file": "stage5_production.yaml",
        "duration": "Ongoing",
        "risk": "Low (validated)",
        "description": "Final production configuration"
    }
}


def load_config(config_path: Path) -> Dict[str, Any]:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        log.error(f"Failed to load config from {config_path}: {e}")
        return {}


def save_config(config: Dict[str, Any], config_path: Path) -> bool:
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        log.info(f"Saved config to {config_path}")
        return True
    except Exception as e:
        log.error(f"Failed to save config to {config_path}: {e}")
        return False


def create_backup(config_path: Path) -> Optional[Path]:
    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"default_{timestamp}.yaml"
        shutil.copy2(config_path, backup_path)
        log.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        log.error(f"Failed to create backup: {e}")
        return None


def validate_vision_identity_config(config: Dict[str, Any]) -> bool:
    if 'vision_identity' not in config:
        log.error("Missing 'vision_identity' section in config")
        return False
    
    vision_identity = config['vision_identity']
    
    if 'mode' not in vision_identity:
        log.error("Missing 'mode' in vision_identity config")
        return False
    
    valid_modes = ['classic', 'shadow_continuity', 'continuity']
    if vision_identity['mode'] not in valid_modes:
        log.error(f"Invalid mode: {vision_identity['mode']} (must be one of {valid_modes})")
        return False
    
    if 'continuity' not in vision_identity:
        log.error("Missing 'continuity' section in vision_identity config")
        return False
    
    continuity = vision_identity['continuity']
    
    required_fields = [
        'min_track_age_frames',
        'appearance_distance_threshold',
        'max_bbox_displacement_fraction',
        'grace_window_seconds'
    ]
    
    for field in required_fields:
        if field not in continuity:
            log.error(f"Missing required field: {field}")
            return False
    
    log.info("Configuration validation passed")
    return True


def merge_configs(base_config: Dict[str, Any], stage_config: Dict[str, Any]) -> Dict[str, Any]:
    result = base_config.copy()
    
    if 'vision_identity' in stage_config:
        if 'vision_identity' not in result:
            result['vision_identity'] = {}
        
        result['vision_identity'].update(stage_config['vision_identity'])
    
    return result


def deploy_stage(stage_num: int, force: bool = False) -> bool:
    if stage_num not in STAGES:
        log.error(f"Invalid stage number: {stage_num} (must be 1-5)")
        return False
    
    stage = STAGES[stage_num]
    stage_file = ROLLOUT_DIR / stage['file']
    
    if not stage_file.exists():
        log.error(f"Stage file not found: {stage_file}")
        return False
    
    log.info(f"\n{'='*60}")
    log.info(f"STAGE {stage_num}: {stage['name']}")
    log.info(f"{'='*60}")
    log.info(f"Description: {stage['description']}")
    log.info(f"Duration: {stage['duration']}")
    log.info(f"Risk Level: {stage['risk']}")
    log.info(f"Config File: {stage['file']}")
    log.info(f"{'='*60}\n")
    
    if not force:
        response = input(f"Deploy Stage {stage_num}? (yes/no): ")
        if response.lower() != 'yes':
            log.info("Deployment cancelled")
            return False
    
    log.info("Loading configurations...")
    base_config = load_config(DEFAULT_CONFIG)
    stage_config = load_config(stage_file)
    
    if not base_config or not stage_config:
        log.error("Failed to load configurations")
        return False
    
    log.info("Merging configurations...")
    merged_config = merge_configs(base_config, stage_config)
    
    log.info("Validating configuration...")
    if not validate_vision_identity_config(merged_config):
        log.error("Configuration validation failed")
        return False
    
    log.info("Creating backup...")
    backup_path = create_backup(DEFAULT_CONFIG)
    if not backup_path:
        log.error("Failed to create backup - aborting deployment")
        return False
    
    log.info("Deploying configuration...")
    if not save_config(merged_config, DEFAULT_CONFIG):
        log.error("Deployment failed - restoring backup")
        shutil.copy2(backup_path, DEFAULT_CONFIG)
        return False
    
    log.info(f"\n{'='*60}")
    log.info(f"✅ STAGE {stage_num} DEPLOYED SUCCESSFULLY")
    log.info(f"{'='*60}")
    log.info(f"Backup created: {backup_path}")
    log.info(f"Mode: {merged_config['vision_identity']['mode']}")
    log.info(f"\nNext steps:")
    log.info(f"1. Restart GaitGuard pipeline")
    log.info(f"2. Monitor metrics for {stage['duration']}")
    log.info(f"3. Validate success criteria")
    log.info(f"4. Proceed to Stage {stage_num + 1} if successful")
    log.info(f"{'='*60}\n")
    
    return True


def rollback_to_backup(backup_path: Optional[str] = None) -> bool:
    if backup_path:
        backup_file = Path(backup_path)
    else:
        if not BACKUP_DIR.exists():
            log.error("No backups found")
            return False
        
        backups = sorted(BACKUP_DIR.glob("default_*.yaml"), reverse=True)
        if not backups:
            log.error("No backups found")
            return False
        
        backup_file = backups[0]
    
    if not backup_file.exists():
        log.error(f"Backup file not found: {backup_file}")
        return False
    
    log.info(f"\n{'='*60}")
    log.info(f"ROLLBACK TO: {backup_file.name}")
    log.info(f"{'='*60}\n")
    
    response = input("Confirm rollback? (yes/no): ")
    if response.lower() != 'yes':
        log.info("Rollback cancelled")
        return False
    
    log.info("Creating safety backup...")
    create_backup(DEFAULT_CONFIG)
    
    log.info("Restoring configuration...")
    try:
        shutil.copy2(backup_file, DEFAULT_CONFIG)
        log.info(f"\n{'='*60}")
        log.info(f"✅ ROLLBACK SUCCESSFUL")
        log.info(f"{'='*60}")
        log.info(f"Restored from: {backup_file}")
        log.info(f"\nNext steps:")
        log.info(f"1. Restart GaitGuard pipeline")
        log.info(f"2. Verify system stable")
        log.info(f"3. Review rollback cause")
        log.info(f"{'='*60}\n")
        return True
    except Exception as e:
        log.error(f"Rollback failed: {e}")
        return False


def emergency_disable() -> bool:
    log.warning(f"\n{'='*60}")
    log.warning(f"⚠️  EMERGENCY DISABLE - CLASSIC MODE")
    log.warning(f"{'='*60}\n")
    
    config = load_config(DEFAULT_CONFIG)
    if not config:
        log.error("Failed to load configuration")
        return False
    
    log.info("Creating backup...")
    create_backup(DEFAULT_CONFIG)
    
    if 'vision_identity' not in config:
        config['vision_identity'] = {}
    
    config['vision_identity']['mode'] = 'classic'
    
    log.info("Disabling continuity mode...")
    if not save_config(config, DEFAULT_CONFIG):
        log.error("Failed to save configuration")
        return False
    
    log.info(f"\n{'='*60}")
    log.info(f"✅ CONTINUITY MODE DISABLED")
    log.info(f"{'='*60}")
    log.info(f"Mode: classic (continuity disabled)")
    log.info(f"\nNext steps:")
    log.info(f"1. Restart GaitGuard pipeline immediately")
    log.info(f"2. Verify system stable")
    log.info(f"3. Investigate incident")
    log.info(f"4. Review logs for root cause")
    log.info(f"{'='*60}\n")
    
    return True


def show_status() -> None:
    config = load_config(DEFAULT_CONFIG)
    
    if not config or 'vision_identity' not in config:
        log.info("Vision Identity mode not configured")
        return
    
    vision_identity = config['vision_identity']
    mode = vision_identity.get('mode', 'unknown')
    
    log.info(f"\n{'='*60}")
    log.info(f"ROLLOUT STATUS")
    log.info(f"{'='*60}")
    log.info(f"Current Mode: {mode}")
    
    if mode == 'classic':
        log.info(f"Stage: N/A (continuity disabled)")
    elif mode == 'shadow_continuity':
        log.info(f"Stage: 1 (Shadow Mode)")
    elif mode == 'continuity':
        continuity = vision_identity.get('continuity', {})
        
        min_age = continuity.get('min_track_age_frames', 0)
        appearance_thresh = continuity.get('appearance_distance_threshold', 0)
        grace_window = continuity.get('grace_window_seconds', 0)
        
        if min_age == 15 and appearance_thresh == 0.30:
            log.info(f"Stage: 2 (Strict Guards)")
        elif grace_window == 0.0:
            log.info(f"Stage: 3 (Relaxed Guards)")
        elif grace_window > 0.0:
            log.info(f"Stage: 4/5 (Grace Enabled / Production)")
        else:
            log.info(f"Stage: Unknown (custom configuration)")
        
        log.info(f"\nKey Thresholds:")
        log.info(f"  - Track Age: {min_age} frames")
        log.info(f"  - Appearance: {appearance_thresh} distance")
        log.info(f"  - Grace Window: {grace_window} seconds")
    
    if BACKUP_DIR.exists():
        backups = sorted(BACKUP_DIR.glob("default_*.yaml"), reverse=True)
        if backups:
            log.info(f"\nAvailable Backups: {len(backups)}")
            for i, backup in enumerate(backups[:5]):
                log.info(f"  {i+1}. {backup.name}")
    
    log.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 9 Rollout Manager - Vision Identity Continuity Mode"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    deploy_parser = subparsers.add_parser('deploy', help='Deploy specific stage')
    deploy_parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3, 4, 5],
                               help='Stage number to deploy (1-5)')
    deploy_parser.add_argument('--force', action='store_true',
                              help='Skip confirmation prompt')
    
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to previous config')
    rollback_parser.add_argument('--backup', type=str,
                                help='Specific backup file to restore')
    
    subparsers.add_parser('emergency-disable', help='Emergency disable (classic mode)')
    
    subparsers.add_parser('status', help='Show current rollout status')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'deploy':
        success = deploy_stage(args.stage, args.force)
        sys.exit(0 if success else 1)
    
    elif args.command == 'rollback':
        success = rollback_to_backup(args.backup)
        sys.exit(0 if success else 1)
    
    elif args.command == 'emergency-disable':
        success = emergency_disable()
        sys.exit(0 if success else 1)
    
    elif args.command == 'status':
        show_status()
        sys.exit(0)


if __name__ == '__main__':
    main()
