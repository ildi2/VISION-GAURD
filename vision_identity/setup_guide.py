

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def step_1_check_environment():
    print("\n" + "="*70)
    print("STEP 1: ENVIRONMENT CHECK")
    print("="*70)
    
    print(f"Python: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    
    packages = ['cv2', 'torch', 'numpy', 'insightface']
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {pkg}: {version}")
        except ImportError:
            print(f"  ✗ {pkg}: NOT INSTALLED")
    
    try:
        import torch
        print(f"\nCUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  Device: {torch.cuda.get_device_name(0)}")
    except:
        pass
    
    print("\n✓ Environment check complete")


def step_2_setup_face_key():
    print("\n" + "="*70)
    print("STEP 2: FACE GALLERY ENCRYPTION KEY")
    print("="*70)
    
    key_var = "GAITGUARD_FACE_KEY"
    current_key = os.environ.get(key_var, "")
    
    if current_key:
        print(f"Current key is set (length: {len(current_key)} chars)")
        if len(current_key) not in [32, 48, 64]:
            print("  ⚠️  Key length should be 32, 48, or 64 hex chars!")
    else:
        print("⚠️  No encryption key is set!")
    
    print("""
To set a new key (PowerShell):
  $env:GAITGUARD_FACE_KEY = "your-32-character-hex-key-here-12345678"

IMPORTANT: The key MUST be 32, 48, or 64 hex characters (16/24/32 bytes).
Example 32-char key: a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6

⚠️  If you lost your key, you need to delete the old gallery and start fresh:
  Remove-Item "data\\face_gallery.enc"
""")
    
    return bool(current_key and len(current_key) in [32, 48, 64])


def step_3_check_galleries():
    print("\n" + "="*70)
    print("STEP 3: GALLERY STATUS")
    print("="*70)
    
    data_dir = project_root / "data"
    
    face_path = data_dir / "face_gallery.enc"
    print(f"\nFace Gallery: {face_path}")
    print(f"  Exists: {face_path.exists()}")
    if face_path.exists():
        print(f"  Size: {face_path.stat().st_size:,} bytes")
    
    gait_path = data_dir / "gait_gallery.pkl"
    print(f"\nGait Gallery: {gait_path}")
    print(f"  Exists: {gait_path.exists()}")
    if gait_path.exists():
        size = gait_path.stat().st_size
        print(f"  Size: {size:,} bytes")
        if size < 50:
            print("  ⚠️  Gallery appears EMPTY!")
        else:
            try:
                import pickle
                data = pickle.load(open(gait_path, 'rb'))
                identities = list(data.get('identities', {}).keys())
                print(f"  Identities: {identities if identities else 'None'}")
            except Exception as e:
                print(f"  Error reading: {e}")
    
    registry_path = data_dir / "identity_registry.json"
    print(f"\nIdentity Registry: {registry_path}")
    print(f"  Exists: {registry_path.exists()}")
    if registry_path.exists():
        import json
        data = json.load(open(registry_path))
        persons = data.get('persons', {})
        print(f"  Registered persons: {len(persons)}")
        for pid, info in persons.items():
            print(f"    - {info.get('display_name', pid)}: face={info.get('face_gallery_id')}, gait={info.get('gait_gallery_id')}")


def step_4_enrollment_instructions():
    print("\n" + "="*70)
    print("STEP 4: ENROLLMENT INSTRUCTIONS")
    print("="*70)
    
    print("""
To enroll a person in the Vision Identity system, you need:
1. Face enrollment (from camera or image)
2. Gait enrollment (from video file)
3. Registry sync (to link them)

─────────────────────────────────────────────────────────────────────
FACE ENROLLMENT (from camera):
─────────────────────────────────────────────────────────────────────
$env:GAITGUARD_FACE_KEY = "your-32-hex-char-key-here-12345678"
python -m identity.enrollment_cli enroll --name "John" --camera 0

─────────────────────────────────────────────────────────────────────
GAIT ENROLLMENT (from video):
─────────────────────────────────────────────────────────────────────
python -m motion_analysis.enrollment_cli enroll --name "john" --source "path/to/video.mp4"

─────────────────────────────────────────────────────────────────────
IDENTITY REGISTRY SYNC (after both enrollments):
─────────────────────────────────────────────────────────────────────
python -m vision_identity.enrollment_integration sync

IMPORTANT: Use MATCHING names (case-insensitive) for face and gait!
- Face: "John" 
- Gait: "john"
The sync will match them automatically.

─────────────────────────────────────────────────────────────────────
RUN VISION IDENTITY FUSION:
─────────────────────────────────────────────────────────────────────
$env:GAITGUARD_FACE_KEY = "your-32-hex-char-key-here-12345678"
python -m vision_identity.simple_runner
""")


def step_5_quick_test():
    print("\n" + "="*70)
    print("STEP 5: QUICK TEST")
    print("="*70)
    
    print("Testing component imports...")
    
    tests = [
        ("Face config", "from face.config import default_face_config"),
        ("Gait config", "from motion_analysis.gait.config import default_gait_config"),
        ("Face gallery", "from identity.face_gallery import FaceGallery"),
        ("Gait gallery", "from motion_analysis.gait.gait_gallery import GaitGallery"),
        ("Identity registry", "from vision_identity.identity_registry import IdentityRegistry"),
        ("Simple fusion", "from vision_identity.simple_fusion import SimpleFusionEngine"),
        ("Simple runner", "from vision_identity.simple_runner import SimpleVisionRunner"),
    ]
    
    all_ok = True
    for name, code in tests:
        try:
            exec(code)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            all_ok = False
    
    if all_ok:
        print("\n✓ All components import successfully!")
    else:
        print("\n⚠️  Some components failed to import. Check errors above.")
    
    return all_ok


def main():
    print("="*70)
    print("VISION IDENTITY SYSTEM - SETUP GUIDE")
    print("="*70)
    
    step_1_check_environment()
    key_ok = step_2_setup_face_key()
    step_3_check_galleries()
    step_4_enrollment_instructions()
    step_5_quick_test()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if not key_ok:
        print("""
⚠️  NEXT STEPS:
1. Set the encryption key: $env:GAITGUARD_FACE_KEY = "your-32-hex-chars"
2. If you lost the old key, delete face_gallery.enc and re-enroll
3. Enroll face: python -m identity.enrollment_cli enroll --name "Name" --camera 0
4. Enroll gait: python -m motion_analysis.enrollment_cli enroll --name "name" --source "video.mp4"
5. Sync registry: python -m vision_identity.enrollment_integration sync
6. Run fusion: python -m vision_identity.simple_runner
""")
    else:
        print("""
✓ Key is set! You can proceed with:
1. Check galleries: python -m vision_identity.check_galleries
2. Enroll face: python -m identity.enrollment_cli enroll --name "Name" --camera 0
3. Enroll gait: python -m motion_analysis.enrollment_cli enroll --name "name" --source "video.mp4"
4. Sync registry: python -m vision_identity.enrollment_integration sync
5. Run fusion: python -m vision_identity.simple_runner
""")


if __name__ == "__main__":
    main()
