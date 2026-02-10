
from __future__ import annotations

import logging
import time
from typing import Optional, Tuple, List
from pathlib import Path

from vision_identity.identity_registry import (
    IdentityRegistry,
    get_identity_registry,
    PersonRecord
)

logger = logging.getLogger(__name__)


class VisionEnrollment:
    
    def __init__(
        self,
        registry: Optional[IdentityRegistry] = None
    ):
        self.registry = registry or get_identity_registry()
        
        self._face_gallery = None
        
        self._gait_gallery = None
        
        logger.info("[VISION-ENROLLMENT] Initialized")
    
    @property
    def face_gallery(self):
        if self._face_gallery is None:
            try:
                from identity.face_gallery import FaceGallery
                from face.config import default_face_config
                
                face_cfg = default_face_config()
                self._face_gallery = FaceGallery(face_cfg.gallery)
                logger.info("[VISION-ENROLLMENT] Face gallery loaded")
            except Exception as e:
                logger.error(f"[VISION-ENROLLMENT] Face gallery load failed: {e}")
        
        return self._face_gallery
    
    @property
    def gait_gallery(self):
        if self._gait_gallery is None:
            try:
                from motion_analysis.gait.gait_gallery import GaitGallery
                from motion_analysis.gait.config import default_gait_config
                
                gait_cfg = default_gait_config()
                self._gait_gallery = GaitGallery(gait_cfg)
                logger.info("[VISION-ENROLLMENT] Gait gallery loaded")
            except Exception as e:
                logger.error(f"[VISION-ENROLLMENT] Gait gallery load failed: {e}")
        
        return self._gait_gallery
    
    
    def enroll_person(
        self,
        display_name: str,
        face_embedding: Optional[any] = None,
        gait_embedding: Optional[any] = None,
        category: str = "resident"
    ) -> Optional[PersonRecord]:
        face_id = None
        gait_id = None
        
        if face_embedding is not None and self.face_gallery:
            face_id = self._enroll_face(display_name, face_embedding, category)
            if face_id:
                logger.info(f"[VISION-ENROLLMENT] Face enrolled: {face_id}")
        
        if gait_embedding is not None and self.gait_gallery:
            gait_id = self._enroll_gait(display_name, gait_embedding, category)
            if gait_id:
                logger.info(f"[VISION-ENROLLMENT] Gait enrolled: {gait_id}")
        
        if face_id or gait_id:
            person = self.registry.register_person(
                display_name=display_name,
                face_id=face_id,
                gait_id=gait_id,
                category=category
            )
            logger.info(
                f"[VISION-ENROLLMENT] Registered: {display_name} "
                f"(face={face_id}, gait={gait_id})"
            )
            return person
        
        logger.warning(f"[VISION-ENROLLMENT] No enrollment for {display_name}")
        return None
    
    def enroll_face_only(
        self,
        display_name: str,
        face_embedding: any,
        category: str = "resident"
    ) -> Optional[str]:
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
        person = self.registry.lookup_by_name(display_name)
        if person is None:
            logger.warning(f"[VISION-ENROLLMENT] Person not found: {display_name}")
            return False
        
        gait_id = self._enroll_gait(display_name, gait_embedding, person.category)
        if gait_id:
            self.registry.register_person(
                display_name=display_name,
                gait_id=gait_id
            )
            return True
        
        return False
    
    
    def _enroll_face(
        self,
        display_name: str,
        face_embedding: any,
        category: str
    ) -> Optional[str]:
        try:
            import numpy as np
            
            existing = self.face_gallery.get_by_name(display_name) if hasattr(self.face_gallery, 'get_by_name') else None
            
            if existing:
                person_id = existing.person_id
            else:
                person_id = self._generate_face_id()
            
            from identity.face_gallery import FaceTemplate, PersonEntry
            
            if isinstance(face_embedding, list):
                embeddings = face_embedding
            else:
                embeddings = [face_embedding]
            
            templates = []
            for emb in embeddings:
                template = FaceTemplate(
                    embedding=np.asarray(emb, dtype=np.float32),
                    metadata={"source": "vision_enrollment"}
                )
                templates.append(template)
            
            person_entry = self.face_gallery.get_person(person_id)
            
            if person_entry:
                for template in templates:
                    self.face_gallery.add_template(person_id, template)
            else:
                person_entry = PersonEntry(
                    person_id=person_id,
                    category=category,
                    name=display_name,
                    templates=templates
                )
                self.face_gallery.add_person(person_entry)
            
            self.face_gallery.save()
            
            return person_id
        
        except Exception as e:
            logger.error(f"[VISION-ENROLLMENT] Face enroll failed: {e}")
            return None
    
    def _enroll_gait(
        self,
        display_name: str,
        gait_embedding: any,
        category: str
    ) -> Optional[str]:
        try:
            import numpy as np
            
            identity_id = display_name
            
            self.gait_gallery.add_gait_embedding(
                identity_id=identity_id,
                new_embedding=np.asarray(gait_embedding, dtype=np.float32),
                category=category,
                confirmed=True
            )
            
            self.gait_gallery.save_gallery()
            
            return identity_id
        
        except Exception as e:
            logger.error(f"[VISION-ENROLLMENT] Gait enroll failed: {e}")
            return None
    
    def _generate_face_id(self) -> str:
        try:
            all_persons = self.face_gallery.list_all()
            count = len(all_persons) + 1
            return f"p_{count:04d}"
        except Exception:
            return f"p_{int(time.time())}"
    
    
    def list_enrolled(self) -> List[PersonRecord]:
        return self.registry.list_all()
    
    def get_enrollment_status(self, display_name: str) -> dict:
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
        person = self.registry.lookup_by_name(display_name)
        if person is None:
            return False
        
        if person.face_gallery_id and self.face_gallery:
            try:
                self.face_gallery.remove_person(person.face_gallery_id)
                self.face_gallery.save()
            except Exception as e:
                logger.warning(f"[VISION-ENROLLMENT] Face gallery remove failed: {e}")
        
        if person.gait_gallery_id and self.gait_gallery:
            try:
                if hasattr(self.gait_gallery, '_identities'):
                    self.gait_gallery._identities.pop(person.gait_gallery_id, None)
                    self.gait_gallery.save_gallery()
            except Exception as e:
                logger.warning(f"[VISION-ENROLLMENT] Gait gallery remove failed: {e}")
        
        self.registry.remove_person(display_name)
        
        logger.info(f"[VISION-ENROLLMENT] Deleted: {display_name}")
        return True


def sync_existing_galleries():
    enrollment = VisionEnrollment()
    registry = enrollment.registry
    
    synced_count = 0
    
    face_persons = {}
    if enrollment.face_gallery:
        try:
            all_face = enrollment.face_gallery.list_all()
            for person in all_face:
                if person.name:
                    face_persons[person.name.lower()] = person.person_id
        except Exception as e:
            logger.warning(f"Could not list face gallery: {e}")
    
    gait_persons = {}
    if enrollment.gait_gallery:
        try:
            if hasattr(enrollment.gait_gallery, '_identities'):
                for gait_id in enrollment.gait_gallery._identities:
                    gait_persons[gait_id.lower()] = gait_id
        except Exception as e:
            logger.warning(f"Could not list gait gallery: {e}")
    
    for name_lower, face_id in face_persons.items():
        if name_lower in gait_persons:
            gait_id = gait_persons[name_lower]
            
            person_entry = enrollment.face_gallery.get_person(face_id)
            display_name = person_entry.name if person_entry and person_entry.name else name_lower.title()
            
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
        enrollment = VisionEnrollment()
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
        enrollment = VisionEnrollment()
        status = enrollment.get_enrollment_status(name)
        
        print(f"Enrollment status for '{name}':")
        for key, value in status.items():
            print(f"  {key}: {value}")
    
    else:
        print(f"Unknown command: {cmd}")
