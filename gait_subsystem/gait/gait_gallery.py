"""
gait/gait_gallery.py - FAISS GALLERY & MATCHING LOGIC
"""
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
import logging
import pickle
from dataclasses import dataclass
from gait_subsystem.gait.config import GaitConfig
import faiss

logger = logging.getLogger(__name__)

@dataclass
class GaitIdentityData:
    """
    Data structure representing a unique identity in the gallery.
    
    Attributes:
        identity_id: The unique string ID (e.g., "Francesco").
        faiss_id: The integer ID used internally by FAISS.
        ema_embedding: The representative embedding vector (averaged over time).
        raw_embeddings: List of recent raw embeddings (optional history).
        num_updates: How many times this identity has been updated/seen.
        category: User category (e.g., "resident", "guest").
    """
    identity_id: str
    faiss_id: int
    ema_embedding: np.ndarray       
    raw_embeddings: List[np.ndarray] 
    num_updates: int = 0
    category: str = "resident"
    # [DEEP ROBUST] Anthropometric Stats (Median [LegRatio, WidthRatio])
    anthro_mean: Optional[np.ndarray] = None

@dataclass
class PersonSummary:
    """DTO for listing persons via CLI."""
    person_id: str
    name: str
    category: str
    num_templates: int

class FaissIndexWrapper:
    """
    Wrapper around the FAISS library to handle vector indexing and searching.
    
    Technical Note:
    We use 'IndexFlatIP' (Inner Product). 
    Since the embeddings are L2-normalized (length = 1), the Inner Product 
    is mathematically equivalent to Cosine Similarity.
    
    Distance = 1.0 - CosineSimilarity.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.current_faiss_id_counter = 0

    def add(self, emb, fid):
        """Adds a normalized vector to the FAISS index with a specific ID."""
        if emb.ndim == 1:
            emb = emb[np.newaxis, :]
        
        # Copy to avoid modifying the original array reference
        emb_copy = emb.astype('float32').copy()
        
        # Ensure L2 Normalization (Critical for Cosine Similarity via IP)
        faiss.normalize_L2(emb_copy)
        self.index.add_with_ids(emb_copy, np.array([fid], dtype=np.int64))
        
    def search(self, query, k=1):
        """Searches for the k-nearest neighbors."""
        if query.ndim == 1:
            query = query[np.newaxis, :]
            
        query_copy = query.astype('float32').copy()
        faiss.normalize_L2(query_copy)
        
        # Returns (similarities, indices)
        return self.index.search(query_copy, k)

class GaitGallery:
    """
    Manages the database of known gait identities.
    Handles loading/saving, updating embeddings via EMA (Exponential Moving Average),
    and searching for matches with strict thresholding.
    """
    def __init__(self, config: GaitConfig):
        self.config = config
        self._identities: Dict[str, GaitIdentityData] = {}
        self.faiss_index = FaissIndexWrapper(config.gallery.dim)
        
        # [DEEP ROBUST] Template ID Mapping
        # Maps unique FAISS ID (int) -> Person ID (str)
        # This solves the aggregation issue where identical IDs collapsed results.
        self.template_id_map: Dict[int, str] = {}
        
        if config.gallery.gallery_path.exists():
            self.load_gallery()

    def save_gallery(self):
        """Persists the gallery state to disk using Pickle."""
        try:
            state = {
                "identities": self._identities, 
                "cnt": self.faiss_index.current_faiss_id_counter
            }
            with open(self.config.gallery.gallery_path, "wb") as f:
                pickle.dump(state, f)
            logger.debug("Gallery saved successfully.")
        except Exception as e:
            logger.error(f"Error saving gallery: {e}")

    def load_gallery(self):
        """Loads the gallery state and reconstructs the FAISS index."""
        try:
            with open(self.config.gallery.gallery_path, "rb") as f:
                state = pickle.load(f)
                self._identities = state["identities"]
                self.faiss_index.current_faiss_id_counter = state["cnt"]
            
            # Rebuild FAISS index from stored embeddings
            self.faiss_index.index.reset()
            self.template_id_map.clear()
            
            # Global Template ID Counter (Monotonic)
            # We rebuild it based on load order to ensure consistency
            tid_counter = 0
            
            for d in self._identities.values():
                # [DEEP ROBUST] Multi-Prototype Indexing
                # Index all raw embeddings if available to capture view variance
                templates = getattr(d, "raw_embeddings", [])
                if not templates: templates = [d.ema_embedding] # Fallback
                
                for vec in templates:
                     # Add to FAISS with UNIQUE Template ID
                     self.faiss_index.add(vec, tid_counter)
                     
                     # Map TID -> PID
                     self.template_id_map[tid_counter] = d.identity_id
                     
                     tid_counter += 1
            
            # Sync the wrapper's counter to continue from here
            self.faiss_index.current_faiss_id_counter = tid_counter
                     
            logger.info(f"Gallery loaded: {len(self._identities)} identities ({tid_counter} templates).")
        except Exception as e:
            logger.warning(f"No existing gallery found or file corrupted ({e}). Starting fresh.")

    def _normalize_numpy(self, x):
        """Helper to normalize a numpy array to unit length (L2 norm)."""
        norm = np.linalg.norm(x)
        if norm > 1e-6:
            return x / norm
        return x

    def add_gait_embedding(self, identity_id, new_embedding, category="resident", confirmed=True, anthro_stats: Optional[np.ndarray] = None):
        """
        Adds or Updates an identity.
        
        If the identity exists and 'confirmed' is True, it updates the stored embedding
        using Exponential Moving Average (EMA). This allows the system to adapt
        to slight changes in a person's gait over time.
        """
        if identity_id not in self._identities:
            # New Identity
            # Normalize before storing
            new_embedding = self._normalize_numpy(new_embedding)
            
            self._identities[identity_id] = GaitIdentityData(
                identity_id=identity_id, 
                faiss_id=0, # Legacy field, unused now (we use dynamic TIDs)
                ema_embedding=new_embedding, 
                raw_embeddings=[new_embedding], 
                num_updates=1, 
                category=category,
                anthro_mean=anthro_stats # Initialize with first sample
            )
            # Add to index immediately with New TID
            tid = self.faiss_index.current_faiss_id_counter
            self.faiss_index.current_faiss_id_counter += 1
            
            self.faiss_index.add(new_embedding, tid)
            self.template_id_map[tid] = identity_id # Update Map
            
            logger.info(f"Added new identity: {identity_id}")
            
        else:
            # Update Existing Identity [MULTI-PROTOTYPE]
            d = self._identities[identity_id]
            d.category = category
            
            # Ensure raw_embeddings list exists
            if getattr(d, "raw_embeddings", None) is None:
                d.raw_embeddings = [d.ema_embedding]
                
            # Append new template
            d.raw_embeddings.append(new_embedding)
            d.num_updates += 1
            
            # Limit history to keep gallery efficient (e.g. 15 templates max)
            # This allows Front, Back, Side, Diagonal, etc.
            if len(d.raw_embeddings) > 15:
                d.raw_embeddings.pop(0) # Remove oldest
            
            # Update EMA (still useful for "Central Tendency" analysis)
            alpha = self.config.gallery.ema_alpha
            d.ema_embedding = self._normalize_numpy((1 - alpha) * d.ema_embedding + alpha * new_embedding)

            # [DEEP ROBUST ANTHRO] Update Geometric Stats
            if anthro_stats is not None:
                if d.anthro_mean is None:
                    d.anthro_mean = anthro_stats
                else:
                    # Slow updates to body shape (weight new sample 20%)
                    d.anthro_mean = 0.8 * d.anthro_mean + 0.2 * anthro_stats

            # [DEEP ROBUST] Incremental Update (Priority 5)
            # Instead of Rebuild (save/load), just add the new vector.
            # This relies on the index and map being consistent.
            
            tid = self.faiss_index.current_faiss_id_counter
            self.faiss_index.current_faiss_id_counter += 1
            
            self.faiss_index.add(new_embedding, tid)
            self.template_id_map[tid] = identity_id
            
            logger.info(f"Updated identity {identity_id} (Template #{tid})")
            
            # Persist the Data (so next start rebuilds correctly)
            self.save_gallery()
            return

        self.save_gallery()

    def search(self, query_embedding: np.ndarray, anthro_query: Optional[np.ndarray] = None) -> Tuple[Optional[str], Optional[float], Dict[str, Any]]:
        """
        Searches the gallery with Aggregation Strategy (Max Sim per Identity).
        Returns None as match_id if no strict match, but always returns details.
        """
        total_vectors = self.faiss_index.index.ntotal
        if total_vectors == 0:
            return None, 0.0, {}

        # Search Deep (Retrieve enough to find 2nd person even if 1st has many templates)
        k_search = min(50, total_vectors)
        sims, fids = self.faiss_index.search(query_embedding, k=k_search)
        
        sims_row = sims[0]
        fids_row = fids[0]

        if fids_row[0] == -1:
            return None, 0.0, {}

        # [DEBUG DEEP ROBUST] Diagnose '2nd: None'
        # Print the raw retrieved IDs and their mappings
        debug_pids = [self.template_id_map.get(f, 'UNK') for f in fids_row if f != -1]
        logger.debug(f"🔍 DEBUG SEARCH: FIDs={fids_row[fids_row != -1]} PIDs={debug_pids}")


        # --- CANDIDATE AGGREGATION ---
        # Map: pid -> (best_sim, best_dist)
        candidates = {} 
        
        for i in range(len(fids_row)):
            fid = fids_row[i]
            if fid == -1: continue
            
            similarity = sims_row[i]
            distance = 1.0 - similarity
            
            # Retrieve String ID via Template ID Map
            # [DEEP ROBUST] O(1) Lookup
            pid = self.template_id_map.get(fid, "Unknown")
            
            if pid == "Unknown": continue
            
            if pid not in candidates:
                candidates[pid] = (similarity, distance)
            else:
                # Keep Max Similarity (Best Template Match)
                if similarity > candidates[pid][0]:
                    candidates[pid] = (similarity, distance)

        # [DEEP ROBUST ANTHRO] Apply Geometric Penalty
        # If we have query stats AND the candidate has stats, penalize mismatch.
        if anthro_query is not None:
            penalty_weight = getattr(self.config.robust, "anthro_penalty_weight", 0.5)
            
            # We iterate a copy to modify scores
            for pid in list(candidates.keys()):
                 ident = self._identities.get(pid)
                 if ident and ident.anthro_mean is not None:
                     # Calculate L1 Distance (Absolute difference in ratios)
                     # Vector: [LegRatio, WidthRatio]
                     dist_vec = np.abs(ident.anthro_mean - anthro_query)
                     
                     # Weight Leg Ratio more (index 0) as it's more stable than width in side view
                     # e.g. 0.7 * Leg + 0.3 * Width
                     geo_dist = 0.7 * dist_vec[0] + 0.3 * dist_vec[1]
                     
                     # Log debug info if significant
                     if geo_dist > 0.05: # 5% difference in body shape
                         original_sim, dist = candidates[pid]
                         
                         # Apply Penalty: New Sim = Old Sim - (Weight * GeoDist)
                         # e.g. 0.88 - (0.5 * 0.2) = 0.78
                         # This drops "Castle" below confirmation threshold if shapes differ.
                         new_sim = max(0.0, original_sim - (penalty_weight * geo_dist))
                         
                         candidates[pid] = (new_sim, dist)
                         # logger.debug(f"📐 Anthro Penalty {pid}: {original_sim:.2f}->{new_sim:.2f} (Dist {geo_dist:.2f})")

        # Sort by Similarity Descending (Using Adjusted Scores)
        ranked = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
        
        if not ranked:
            return None, 0.0, {}

        # [DEEP ROBUST] TOP-K VOTING (Soft Voting)
        # Instead of taking ranked[0], we analyze the top K templates involved.
        # Since 'candidates' currently stores only MAX sim per PID, we need to respect that.
        # But wait - 'candidates' is already aggregated per PID.
        
        # To do true Template Voting, we would need the raw scores of ALL templates, not just max per pid.
        
        best_pid, (best_sim, best_dist) = ranked[0]
        
        details = {
            "best_pid": best_pid,
            "best_sim": best_sim,
            "best_dist": best_dist,
            "margin": 0.0
        }

        if len(ranked) > 1:
            second_pid, (second_sim, _) = ranked[1]
            margin = best_sim - second_sim
            details["second_pid"] = second_pid
            details["second_sim"] = second_sim
            details["margin"] = margin
        
        # Inject Anthropometric Debug Data
        if anthro_query is not None:
            details["anthro_vec"] = anthro_query.tolist()
            if best_pid in self._identities and self._identities[best_pid].anthro_mean is not None:
                 details["match_anthro_dist"] = float(np.sum(np.abs(self._identities[best_pid].anthro_mean - anthro_query)))

        # --- DECISION LOGIC ---
        limit = self.config.thresholds.max_match_distance
        
        # 1. Validation Check
        if best_dist > limit:
            # Too far. Return None as match, but pass details for logging.
            logger.debug(f"ℹ️ Reject Weak: {best_pid} (Sim {best_sim:.4f})")
            details["status"] = "REJECT_WEAK"
            return None, best_sim, details

        # 2. Margin Check
        if margin < self.config.thresholds.min_match_margin:
            logger.debug(f"⚠️ Low Margin ({margin:.3f}) - Passing to Engine")
            details["status"] = "LOW_MARGIN"
            return best_pid, best_sim, details
            
        return best_pid, best_sim, details

    def list_persons(self) -> List[PersonSummary]:
        return [PersonSummary(k, k, v.category, v.num_updates) for k, v in self._identities.items()]

    def delete_person(self, pid: str) -> bool:
        if pid in self._identities:
            self.faiss_index.index.remove_ids(np.array([self._identities[pid].faiss_id], dtype=np.int64))
            del self._identities[pid]
            self.save_gallery()
            return True
        return False

    def get_category(self, pid: str) -> str:
        """Returns the category (e.g., 'resident', 'visitor') of an enrolled identity."""
        if pid in self._identities:
            return self._identities[pid].category
        return "unknown"