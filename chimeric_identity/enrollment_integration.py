# chimeric_identity/enrollment_integration.py
# ============================================================================
# CHIMERIC ENROLLMENT - Register Person in Both Galleries
# ============================================================================
#
# Purpose:
#   Enroll a person in BOTH face gallery AND gait gallery, then register
#   the mapping in the identity registry so chimeric fusion knows they're
#   the same person.
#
# This is the KEY to making chimeric fusion work:
#   1. Enroll face → get face_gallery_id (e.g., "p_0001")
#   2. Enroll gait → get gait_gallery_id (e.g., "Marildo")
#   3. Register both IDs in identity_registry.json
#
# After enrollment, chimeric fusion can look up:
#   "Is face_id=p_0001 the same person as gait_id=Marildo?" → YES

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple, List
from pathlib import Path

from chimeric_identity.identity_registry import (
    IdentityRegistry,
    get_identity_registry,
    PersonRecord
)

logger = logging.getLogger(__name__)


class ChimericEnrollment:
    """
    Enrollment manager for chimeric identity system.
    
    Handles enrollment into both face and gait galleries, then creates
    the identity mapping so fusion knows they're the same person.
    
    Usage:
        enrollment = ChimericEnrollment()
        
        # Enroll a new person
        person_id = enrollment.enroll_person(
            display_name="Marildo",
            face_frames=captured_face_frames,
            gait_sequence=captured_gait_sequence
        )
        
        # Now chimeric fusion will recognize this person via face OR gait
        # and combine confidences when both are present!
    """
    
    def __init__(
        self,
        registry: Optional[IdentityRegistry] = None
    ):
        """
        Initialize chimeric enrollment.
        
        Args:
            registry: Identity registry (uses global singleton if None)
        """
        self.registry = registry or get_identity_registry()
        
        # Initialize face gallery (lazy)
        self._face_gallery = None
        
        # Initialize gait gallery (lazy)
        self._gait_gallery = None
        
        logger.info("[CHIMERIC-ENROLLMENT] Initialized")
    
    @property
    def face_gallery(self):
        """Lazy load face gallery."""
        if self._face_gallery is None:
            try:
                from identity.face_gallery import FaceGallery
                from face.config import default_face_config
                
                face_cfg = default_face_config()
                self._face_gallery = FaceGallery(face_cfg.gallery)
                logger.info("[CHIMERIC-ENROLLMENT] Face gallery loaded")
            except Exception as e:
                logger.error(f"[CHIMERIC-ENROLLMENT] Face gallery load failed: {e}")
        
        return self._face_gallery
    
    @property
    def gait_gallery(self):
        """Lazy load gait gallery."""
        if self._gait_gallery is None:
            try:
                from gait_subsystem.gait.gait_gallery import GaitGallery
                from gait_subsystem.gait.config import default_gait_config
                
                gait_cfg = default_gait_config()
                self._gait_gallery = GaitGallery(gait_cfg)
                logger.info("[CHIMERIC-ENROLLMENT] Gait gallery loaded")
            except Exception as e:
                logger.error(f"[CHIMERIC-ENROLLMENT] Gait gallery load failed: {e}")
        
        return self._gait_gallery
    
    # ========================================================================
    # MAIN ENROLLMENT METHODS
    # ========================================================================
    
    def enroll_person(
        self,
        display_name: str,
        face_embedding: Optional[any] = None,
        gait_embedding: Optional[any] = None,
        category: str = "resident"
    ) -> Optional[PersonRecord]:
        """
        Enroll a person in both galleries and register in identity mapping.
        
        This is the MAIN enrollment method that:
        1. Adds to face gallery → gets face_id
        2. Adds to gait gallery → gets gait_id
        3. Registers both in identity registry
        
        Args:
            display_name: Human-readable name (e.g., "Marildo")
            face_embedding: Face embedding(s) from FaceRoute
            gait_embedding: Gait embedding from GaitExtractor
            category: Person category
        
        Returns:
            PersonRecord if successful, None otherwise
        """
        face_id = None
        gait_id = None
        
        # Step 1: Enroll in face gallery
        if face_embedding is not None and self.face_gallery:
            face_id = self._enroll_face(display_name, face_embedding, category)
            if face_id:
                logger.info(f"[CHIMERIC-ENROLLMENT] Face enrolled: {face_id}")
        
        # Step 2: Enroll in gait gallery
        if gait_embedding is not None and self.gait_gallery:
            gait_id = self._enroll_gait(display_name, gait_embedding, category)
            if gait_id:
                logger.info(f"[CHIMERIC-ENROLLMENT] Gait enrolled: {gait_id}")
        
        # Step 3: Register in identity registry
        if face_id or gait_id:
            person = self.registry.register_person(
                display_name=display_name,
                face_id=face_id,
                gait_id=gait_id,
                category=category
            )
            logger.info(
                f"[CHIMERIC-ENROLLMENT] Registered: {display_name} "
                f"(face={face_id}, gait={gait_id})"
            )
            return person
        
        logger.warning(f"[CHIMERIC-ENROLLMENT] No enrollment for {display_name}")
        return None
    
    def enroll_face_only(
        self,
        display_name: str,
        face_embedding: any,
        category: str = "resident"
    ) -> Optional[str]:
        """
        Enroll only face (gait can be added later).
        
        Returns face_gallery_id
        """
        face_id = None
        
        if self.face_gallery:
            face_id = self._enroll_face(display_name, face_embedding, category)
        
        if face_id:
            self.registry.register_person(
                display_name=display_name,
                face_id=face_id,
                category=category
            )
        
        return face_id
    
    def enroll_gait_only(
        self,
        display_name: str,
        gait_embedding: any,
        category: str = "resident"
    ) -> Optional[str]:
        """
        Enroll only gait (face can be added later).
        
        Returns gait_gallery_id
        """
        gait_id = None
        
        if self.gait_gallery:
            gait_id = self._enroll_gait(display_name, gait_embedding, category)
        
        if gait_id:
            self.registry.register_person(
                display_name=display_name,
                gait_id=gait_id,
                category=category
            )
        
        return gait_id
    
    def add_gait_to_existing(
        self,
        display_name: str,
        gait_embedding: any
    ) -> bool:
        """
        Add gait enrollment to existing face-only person.
        
        Use this when someone was enrolled with face only and
        now we want to add gait recognition.
        """
        person = self.registry.lookup_by_name(display_name)
        if person is None:
            logger.warning(f"[CHIMERIC-ENROLLMENT] Person not found: {display_name}")
            return False
        
        gait_id = self._enroll_gait(display_name, gait_embedding, person.category)
        if gait_id:
            self.registry.register_person(
                display_name=display_name,
                gait_id=gait_id
            )
            return True
        
        return False
    
    # ========================================================================
    # INTERNAL METHODS
    # ========================================================================
    
    def _enroll_face(
        self,
        display_name: str,
        face_embedding: any,
        category: str
    ) -> Optional[str]:
        """
        Enroll in face gallery.
        
        The face gallery uses person_id like "p_0001".
        We need to generate this ID or use what gallery assigns.
        """
        try:
            import numpy as np
            
            # Get or generate person_id
            # Check if person already exists
            existing = self.face_gallery.get_by_name(display_name) if hasattr(self.face_gallery, 'get_by_name') else None
            
            if existing:
                # Update existing
                person_id = existing.person_id
            else:
                # Generate new ID
                person_id = self._generate_face_id()
            
            # Add template to gallery
            from identity.face_gallery import FaceTemplate, PersonEntry
            
            # Handle single or multiple embeddings
            if isinstance(face_embedding, list):
                embeddings = face_embedding
            else:
                embeddings = [face_embedding]
            
            templates = []
            for emb in embeddings:
                template = FaceTemplate(
                    embedding=np.asarray(emb, dtype=np.float32),
                    metadata={"source": "chimeric_enrollment"}
                )
                templates.append(template)
            
            # Check if person exists in gallery
            person_entry = self.face_gallery.get_person(person_id)
            
            if person_entry:
                # Add templates to existing person
                for template in templates:
                    self.face_gallery.add_template(person_id, template)
            else:
                # Create new person
                person_entry = PersonEntry(
                    person_id=person_id,
                    category=category,
                    name=display_name,
                    templates=templates
                )
                self.face_gallery.add_person(person_entry)
            
            # Save gallery
            self.face_gallery.save()
            
            return person_id
        
        except Exception as e:
            logger.error(f"[CHIMERIC-ENROLLMENT] Face enroll failed: {e}")
            return None
    
    def _enroll_gait(
        self,
        display_name: str,
        gait_embedding: any,
        category: str
    ) -> Optional[str]:
        """
        Enroll in gait gallery.
        
        The gait gallery uses identity_id which is typically the name.
        """
        try:
            import numpy as np
            
            # Gait gallery uses display_name as identity_id
            identity_id = display_name
            
            # Add embedding to gallery
            self.gait_gallery.add_gait_embedding(
                identity_id=identity_id,
                new_embedding=np.asarray(gait_embedding, dtype=np.float32),
                category=category,
                confirmed=True
            )
            
            # Save gallery
            self.gait_gallery.save_gallery()
            
            return identity_id
        
        except Exception as e:
            logger.error(f"[CHIMERIC-ENROLLMENT] Gait enroll failed: {e}")
            return None
    
    def _generate_face_id(self) -> str:
        """Generate a unique face gallery ID."""
        # Count existing persons
        try:
            all_persons = self.face_gallery.list_all()
            count = len(all_persons) + 1
            return f"p_{count:04d}"
        except Exception:
            # Fallback to timestamp-based ID
            return f"p_{int(time.time())}"
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def list_enrolled(self) -> List[PersonRecord]:
        """List all enrolled persons."""
        return self.registry.list_all()
    
    def get_enrollment_status(self, display_name: str) -> dict:
        """
        Get enrollment status for a person.
        
        Returns dict with:
        - has_face: bool
        - has_gait: bool
        - face_id: str or None
        - gait_id: str or None
        """
        person = self.registry.lookup_by_name(display_name)
        if person is None:
            return {
                "enrolled": False,
                "has_face": False,
                "has_gait": False,
                "face_id": None,
                "gait_id": None
            }
        
        return {
            "enrolled": True,
            "has_face": person.has_face(),
            "has_gait": person.has_gait(),
            "has_both": person.has_both(),
            "face_id": person.face_gallery_id,
            "gait_id": person.gait_gallery_id
        }
    
    def delete_person(self, display_name: str) -> bool:
        """
        Delete person from all galleries and registry.
        
        WARNING: This removes ALL data for the person!
        """
        person = self.registry.lookup_by_name(display_name)
        if person is None:
            return False
        
        # Remove from face gallery
        if person.face_gallery_id and self.face_gallery:
            try:
                self.face_gallery.remove_person(person.face_gallery_id)
                self.face_gallery.save()
            except Exception as e:
                logger.warning(f"[CHIMERIC-ENROLLMENT] Face gallery remove failed: {e}")
        
        # Remove from gait gallery  
        if person.gait_gallery_id and self.gait_gallery:
            try:
                if hasattr(self.gait_gallery, '_identities'):
                    self.gait_gallery._identities.pop(person.gait_gallery_id, None)
                    self.gait_gallery.save_gallery()
            except Exception as e:
                logger.warning(f"[CHIMERIC-ENROLLMENT] Gait gallery remove failed: {e}")
        
        # Remove from registry
        self.registry.remove_person(display_name)
        
        logger.info(f"[CHIMERIC-ENROLLMENT] Deleted: {display_name}")
        return True


# ============================================================================
# CLI INTEGRATION
# ============================================================================

def sync_existing_galleries():
    """
    Sync existing face and gait galleries with identity registry.
    
    Use this to populate identity_registry.json from existing galleries
    when names match between face and gait entries.
    """
    enrollment = ChimericEnrollment()
    registry = enrollment.registry
    
    synced_count = 0
    
    # Get all face entries
    face_persons = {}
    if enrollment.face_gallery:
        try:
            all_face = enrollment.face_gallery.list_all()
            for person in all_face:
                if person.name:
                    face_persons[person.name.lower()] = person.person_id
        except Exception as e:
            logger.warning(f"Could not list face gallery: {e}")
    
    # Get all gait entries
    gait_persons = {}
    if enrollment.gait_gallery:
        try:
            if hasattr(enrollment.gait_gallery, '_identities'):
                for gait_id in enrollment.gait_gallery._identities:
                    gait_persons[gait_id.lower()] = gait_id
        except Exception as e:
            logger.warning(f"Could not list gait gallery: {e}")
    
    # Find matches (case-insensitive)
    for name_lower, face_id in face_persons.items():
        if name_lower in gait_persons:
            gait_id = gait_persons[name_lower]
            
            # Get proper display name from face gallery
            person_entry = enrollment.face_gallery.get_person(face_id)
            display_name = person_entry.name if person_entry and person_entry.name else name_lower.title()
            
            # Register
            registry.register_person(
                display_name=display_name,
                face_id=face_id,
                gait_id=gait_id,
                category=person_entry.category if person_entry else "resident"
            )
            synced_count += 1
            print(f"Synced: {display_name} (face={face_id}, gait={gait_id})")
    
    print(f"\nSynced {synced_count} persons with matching face and gait entries")
    return synced_count


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python enrollment_integration.py sync    # Sync existing galleries")
        print("  python enrollment_integration.py list    # List enrolled persons")
        print("  python enrollment_integration.py status <name>  # Check enrollment status")
        sys.exit(0)
    
    cmd = sys.argv[1]
    
    if cmd == "sync":
        sync_existing_galleries()
    
    elif cmd == "list":
        enrollment = ChimericEnrollment()
        persons = enrollment.list_enrolled()
        
        if not persons:
            print("No enrolled persons")
        else:
            print(f"Enrolled persons ({len(persons)}):")
            for p in persons:
                status = "✓ Both" if p.has_both() else ("👤 Face" if p.has_face() else "🚶 Gait")
                print(f"  {p.display_name}: {status} (face={p.face_gallery_id}, gait={p.gait_gallery_id})")
    
    elif cmd == "status" and len(sys.argv) >= 3:
        name = sys.argv[2]
        enrollment = ChimericEnrollment()
        status = enrollment.get_enrollment_status(name)
        
        print(f"Enrollment status for '{name}':")
        for key, value in status.items():
            print(f"  {key}: {value}")
    
    else:
        print(f"Unknown command: {cmd}")
