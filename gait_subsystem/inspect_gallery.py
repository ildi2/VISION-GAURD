
#inspect_gallery.py  
from __future__ import annotations
import numpy as np
import sys
from gait.config import default_gait_config
from gait.gait_gallery import GaitGallery

def main():
    # Carica la configurazione e la galleria
    cfg = default_gait_config()
    gallery = GaitGallery(cfg)
    
    # Recupera tutte le identità
    identities = list(gallery._identities.values())
    
    if not identities:
        print("❌ La galleria è vuota. Fai prima l'enrollment.")
        return

    print(f"\n📊 MATRICE DI SIMILARITÀ ({len(identities)} persone)")
    print("Valori vicini a 1.0 = Identici")
    print("Valori vicini a 0.0 = Diversi")
    print(f"Soglia Match Sicuro: {cfg.thresholds.max_match_distance} (Distanza) -> Similarità > {1.0 - cfg.thresholds.max_match_distance:.2f}")
    print("-" * 80)

    # Intestazione tabella
    ids = [d.identity_id for d in identities]
    print(f"{'ID':<15} |", end="")
    for pid in ids:
        print(f" {pid[:6]:<7}", end="")
    print("\n" + "-" * (15 + 8 * len(ids) + 2))

    # Calcolo matrice
    for i, p1 in enumerate(identities):
        print(f"{p1.identity_id:<15} |", end="")
        
        emb1 = p1.ema_embedding
        # Normalizza per sicurezza (dovrebbe già esserlo)
        emb1 = emb1 / np.linalg.norm(emb1)

        for j, p2 in enumerate(identities):
            emb2 = p2.ema_embedding
            emb2 = emb2 / np.linalg.norm(emb2)

            # Cosine Similarity = dot product tra vettori normalizzati
            similarity = np.dot(emb1, emb2)
            
            # Formattazione condizionale per leggibilità
            val_str = f"{similarity:.3f}"
            if i == j:
                print(f"\033[94m {val_str:<7}\033[0m", end="") # Blu (stessa persona)
            elif similarity > (1.0 - cfg.thresholds.max_match_distance):
                print(f"\033[91m {val_str:<7}\033[0m", end="") # Rosso (Falso positivo pericolo!)
            elif similarity > (1.0 - cfg.thresholds.max_weak_match_distance):
                print(f"\033[93m {val_str:<7}\033[0m", end="") # Giallo (Vicini)
            else:
                print(f"\033[92m {val_str:<7}\033[0m", end="") # Verde (Ben distinti)
        
        print(f"") # Nuova riga

    print("-" * 80)
    print("LEGENDA:")
    print("\033[94mBLU\033[0m   = Stessa persona (deve essere 1.000)")
    print("\033[92mVERDE\033[0m = Ben distinti (Ottimo)")
    print("\033[93mGIALLO\033[0m= Simili ma distinguibili (Warning)")
    print("\033[91mROSSO\033[0m  = Troppo simili, rischio confusione (Conflict)")

if __name__ == "__main__":
    main()