import os
import numpy as np

def check_shards(directory="wikipedia_fr"):
    total_tokens = 0
    num_shards = 0
    
    print(f"Vérification des shards dans {directory}...")
    
    # Parcourir tous les fichiers .npy
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            path = os.path.join(directory, filename)
            shard = np.load(path)
            total_tokens += len(shard)
            num_shards += 1
            print(f"Shard {filename}: {len(shard)} tokens")
    
    print(f"\nRésumé:")
    print(f"Nombre de shards: {num_shards}")
    print(f"Nombre total de tokens: {total_tokens:,}")
    print(f"Taille moyenne des shards: {total_tokens/num_shards:,.0f} tokens")

if __name__ == "__main__":
    check_shards() 