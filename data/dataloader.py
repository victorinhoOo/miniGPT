import os
import numpy as np
import torch

def load_tokens(filename):
    """Charge un fichier de tokens depuis un fichier .npy
    
    Args:
        filename: Chemin vers le fichier .npy contenant les tokens
    
    Returns:
        torch.Tensor: Tenseur contenant les tokens
    """
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    return torch.tensor(npt, dtype=torch.long)

class DataLoaderLite:
    """Chargeur de données optimisé pour l'entraînement du modèle GPT.
    
    Cette classe gère le chargement et le découpage des données en batches.
    Elle supporte également la distribution des données sur plusieurs processus
    pour l'entraînement distribué.
    
    Args:
        B: Taille du batch
        T: Longueur de séquence
        process_rank: Rang du processus actuel (pour DDP)
        num_processes: Nombre total de processus (pour DDP)
        split: 'train' ou 'val'
        num_workers: Nombre de workers pour le chargement des données
    """
    def __init__(self, B, T, process_rank, num_processes, split, num_workers=4):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # Récupération des shards
        data_root = "wikipedia_fr"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        
        # Affichage du nombre de shards si processus principal
        if process_rank == 0:
            print(f"found {len(shards)} shards for split {split}")
        
        self.reset()

    def reset(self):
        """Réinitialise le chargeur à son état initial."""
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """Retourne le prochain batch de données.
        
        Returns:
            tuple: (x, y) où x sont les entrées et y les cibles
                  Chaque élément a la forme (B, T)
        """
        B, T = self.B, self.T
        
        # Extraction du batch courant
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = buf[:-1].view(B, T)  # Entrées
        y = buf[1:].view(B, T)   # Cibles (décalées de 1)
        
        # Avance la position
        self.current_position += B * T * self.num_processes
        
        # Si on atteint la fin du shard, passe au suivant
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
            
        return x, y
