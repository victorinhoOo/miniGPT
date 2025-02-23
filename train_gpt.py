from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
import time
import os
import tiktoken

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
        self.c_proj.MINIGPT_SCALE_INIT = 1
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
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Application du masque causal : on remplace par -inf les positions futures 
        # (cela permet de ne pas integrer les mots futurs dans le calcul d'attention)
        #att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # Normalisation par softmax pour obtenir des poids d'attention entre 0 et 1
        #att = F.softmax(att, dim=-1)

        # Calcul de la sortie en multipliant les poids d'attention avec les valeurs
        #y = att @ v

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

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
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.MINIGPT_SCALE_INIT = 1
    
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

        # Partage des poids entre l'embedding et la couche de sortie (weight tying)
        # Cette technique réduit le nombre de paramètres et améliore les performances
        self.transformer.wte.weight = self.lm_head.weight

        # Initialisation des poids 
        self.apply(self._init_weights)


    def _init_weights(self, module):

        """Initialise les poids des modules pour que ceux si suivent une distribution normale
        cela permet d'éviter les problèmes d'explosion ou de disparition des gradients."""

        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'MINIGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emd = self.transformer.wpe(pos)
        tok_emd = self.transformer.wte(idx)
        x = tok_emd + pos_emd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Charge les poids d'un modèle GPT-2 pré-entraîné depuis Huggingface
        
        Cette méthode permet de récupérer un modèle GPT-2 déjà entraîné, plutôt que de
        partir de zéro. 
        """
        # On vérifie que le type de modèle demandé existe bien
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Chargement des poids depuis le modele pre-entraine : %s" % model_type)

        # Configuration du modèle selon sa taille
        # Plus le modèle est grand, plus il a de paramètres et plus il est puissant
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M paramètres - Version de base
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M paramètres - Version moyenne
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M paramètres - Grande version
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M paramètres - Version extra large
        }[model_type]
        
        # Paramètres fixes pour tous les modèles GPT-2
        config_args['vocab_size'] = 50257  # Taille du vocabulaire (nombre de mots que connaît le modèle)
        config_args['block_size'] = 1024   # Longueur maximale du texte que le modèle peut traiter

        # Création de notre modèle vide
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        # On retire certains éléments techniques qui ne sont pas des paramètres à copier
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # Chargement du modèle pré-entraîné depuis Huggingface
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        # On retire certains éléments techniques qui ne sont pas des paramètres à copier
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        
        # Liste des paramètres qui nécessitent une transposition
        # (comme retourner une matrice pour qu'elle soit dans le bon sens)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # Vérification que les deux modèles ont le même nombre de paramètres
        assert len(sd_keys_hf) == len(sd_keys), f"Nombre de paramètres différent : {len(sd_keys_hf)} != {len(sd_keys)}"

        # Copie des paramètres du modèle pré-entraîné vers notre modèle
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # Certains paramètres doivent être transposés avant la copie
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Copie simple pour les autres paramètres
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print(f"using fused AdamW: {use_fused}")

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        
        return optimizer

# --------------------------------------------------------------------------------------------------

import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, num_workers=4):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "wikipedia_fr"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

prompts_test = [
    "Bonjour, comment allez-vous",
    "La France est un pays",
    "L'intelligence artificielle permet de",
    "Le but de ce projet est de"
]

# Paramètres ajustés pour A100
B = 32                    
T = 1024                  
total_batch_size = 524288 # 2**19 

# Paramètres d'apprentissage optimisés
max_lr = 3e-4            
min_lr = max_lr * 0.1
warmup_steps = 300       
max_steps = 6000         

# Activer la compilation torch pour de meilleures performances
use_compile = True       # Activation de torch.compile()

# Optimisations CUDA
torch.backends.cuda.matmul.allow_tf32 = True  # Permet TF32 sur A100
torch.backends.cudnn.benchmark = True         # Optimise les convolutions


# Optimise le DataLoader
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
train_loader = DataLoaderLite(
    B=B, 
    T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="train",
    num_workers=4 
)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

def get_lr(it):
    # 1) réchauffement linéaire pendant warmup_iters étapes
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) si it > lr_decay_iters, retourne le taux d'apprentissage minimum
    if it > max_steps:
        return min_lr
    # 3) entre les deux, utilise une décroissance en cosinus jusqu'au taux minimum
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # le coefficient commence à 1 et descend à 0
    return min_lr + coeff * (max_lr - min_lr)

# optimisation !
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# crée le répertoire de logs où nous écrirons les points de contrôle et les logs
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # ouverture en écriture pour vider le fichier
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # écriture optionnelle des checkpoints du modèle
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)

    # effectue une étape d'optimisation
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # nous devons mettre à l'échelle la perte pour tenir compte de l'accumulation du gradient,
        # car les gradients s'ajoutent simplement à chaque backward() successif.
        # l'addition des gradients correspond à une SOMME dans l'objectif, mais
        # au lieu d'une SOMME nous voulons une MOYENNE. On met à l'échelle la perte ici pour que le résultat soit correct
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # détermine et définit le taux d'apprentissage pour cette itération
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # attend que le GPU termine son travail
    t1 = time.time()
    dt = t1 - t0 # différence de temps en secondes
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()