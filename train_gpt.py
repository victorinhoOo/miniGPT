from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# --------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Mécanisme d'attention qui permet au modèle de traiter les séquences de texte.
    
    Cette classe implémente l'attention "causale", elle peut être
    comparé à la lecture d'une phrase :
    
    - Quand nous lisons le mot "je", nous ne pouvons voir que ce mot
    - Pour le mot "mange", nous pouvons voir "je mange"
    - Pour "une", nous voyons "je mange une"
    - Et ainsi de suite...
    
    Cette contrainte force le modèle à prédire chaque mot uniquement en se basant sur
    les mots précédents, comme un humain qui lit ou écrit de gauche à droite.
    
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # projections pour les clés, requêtes et valeurs pour toutes les têtes d'attention, regroupées en un seul tenseur
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # projection de sortie pour combiner les résultats de toutes les têtes
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # création du masque triangulaire causal qui empêche de voir les tokens futurs
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                           .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # B = taille du batch (nombre de séquences traitées en parallèle)
        # T = longueur de la séquence (nombre de tokens)
        # C = dimension des embeddings
        B, T, C = x.size()

        # Calcul des vecteurs query (q), key (k) et value (v) pour chaque tête d'attention
        # Ces vecteurs permettent de déterminer quels mots sont importants pour chaque position
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Réorganisation des dimensions pour traiter séparément chaque tête d'attention
        # Format final : [batch, nb_têtes, longueur_seq, dimension_par_tête]
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Calcul des scores d'attention
        # 1. Multiplication matricielle entre q et k pour obtenir les scores bruts
        # 2. Mise à l'échelle pour éviter que les gradients ne soient trop grands
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Application du masque causal : on remplace par -inf les positions futures 
        # (cela permet de ne pas integrer les mots futurs dans le calcul d'attention)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # Normalisation par softmax pour obtenir des poids d'attention entre 0 et 1
        att = F.softmax(att, dim=-1)

        # Calcul de la sortie en multipliant les poids d'attention avec les valeurs
        y = att @ v
        # Réorganisation des dimensions pour obtenir le format attendu
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Projection finale pour combiner les résultats de toutes les têtes
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    """Perceptron multicouche (Multi-Layer Perceptron).
    
    Cette classe implémente une couche de transformation non-linéaire qui permet 
    au modèle d'apprendre des relations complexes dans les données.
    
    Le processus se déroule en trois étapes :
    1. Projection vers une dimension plus grande (x4) pour capturer plus d'informations
    2. Application d'une fonction d'activation non-linéaire (GELU)
    3. Projection retour vers la dimension d'origine
    
    Args:
        config: Configuration contenant n_embd (dimension des embeddings)
    """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
    
    def forward(self, x):
        """Applique la transformation MLP sur l'entrée.
        
        Args:
            x: Tensor d'entrée de forme (batch_size, seq_length, n_embd)
            
        Returns:
            Tensor de même forme que l'entrée après transformation non-linéaire
        """
        x = self.c_fc(x)    # Expansion de la dimension
        x = self.gelu(x)    # Application de la non-linéarité
        x = self.c_proj(x)  # Retour à la dimension d'origine
        return x


class Block(nn.Module):
    """Bloc de base du transformer GPT.
    
    Cette classe représente un bloc fondamental de l'architecture GPT,
    combinant l'attention et le traitement MLP. Chaque bloc effectue
    deux opérations principales avec des connexions résiduelles :
    
    1. Attention avec normalisation :
       - Normalise l'entrée (LayerNorm)
       - Applique l'attention
       - Ajoute le résultat à l'entrée originale 
    
    2. MLP avec normalisation :
       - Normalise l'entrée (LayerNorm)
       - Applique la transformation MLP
       - Ajoute le résultat à l'entrée (connexion résiduelle)
    
    Les connexions résiduelles (x + ...) permettent au réseau d'être
    plus profond en évitant le problème de la disparition du gradient.
    
    Args:
        config: Configuration contenant n_embd (dimension des embeddings)
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)      # Première couche de normalisation
        self.attn = CausalSelfAttention(config)      # Mécanisme d'attention
        self.ln_2 = nn.LayerNorm(config.n_embd)      # Deuxième couche de normalisation
        self.mlp = MLP(config)                       # Perceptron multicouche

    def forward(self, x):
        """Applique les transformations du bloc sur l'entrée.
        
        Args:
            x: Tensor d'entrée de forme (batch_size, seq_length, n_embd)
            
        Returns:
            Tensor de même forme après application de l'attention et du MLP
        """
        x = x + self.attn(self.ln_1(x))  # Attention avec connexion résiduelle
        x = x + self.mlp(self.ln_2(x))   # MLP avec connexion résiduelle
        return x

@dataclass
class GPTConfig:
    """Configuration du modèle GPT.
    
    Cette classe définit les hyperparamètres principaux du modèle :
    
    Attributes:
        block_size: Longueur maximale des séquences que le modèle peut traiter
        vocab_size: Taille du vocabulaire (nombre de tokens différents)
        n_layer: Nombre de blocs transformer empilés
        n_head: Nombre de têtes d'attention par bloc
        n_embd: Dimension des vecteurs d'embedding (représentation des tokens)
    """
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class GPT(nn.Module):
    """Implémentation simplifiée du modèle GPT (Generative Pre-trained Transformer).
    
    Cette classe assemble tous les composants du transformer pour créer le modèle complet.
    Elle contient :
    
    1. Une couche d'embedding pour les tokens (wte):
       - Transforme chaque token en un vecteur de dimension n_embd
    
    2. Une couche d'embedding pour les positions (wpe):
       - Permet au modèle de savoir où se trouve chaque token dans la séquence
    
    3. Une pile de blocs transformer (h):
       - Chaque bloc contient de l'attention et un MLP
       - Le nombre de blocs est défini par n_layer
    
    4. Une couche de normalisation finale (ln_f):
       - Stabilise les activations avant la prédiction
    
    5. Une tête de langage (lm_head):
       - Convertit les embeddings en probabilités sur le vocabulaire
    
    Args:
        config: Instance de GPTConfig contenant les hyperparamètres du modèle
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # Token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # Position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Blocs transformer
            ln_f = nn.LayerNorm(config.n_embd),  # Normalisation finale
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # Projection vers le vocabulaire

