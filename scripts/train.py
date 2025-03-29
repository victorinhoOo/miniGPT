import os
import sys
# Ajouter le chemin racine du projet au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import math
import torch
import tiktoken
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model.gpt import GPT, GPTConfig
from data.dataloader import DataLoaderLite

# Configuration d'entraînement
B = 32                    # Taille du lot par GPU
T = 1024                  # Longueur de contexte pour les séquences
total_batch_size = 524288 # Taille totale du lot (2**19)

# Paramètres d'apprentissage
max_lr = 3e-4            # Taux d'apprentissage initial
min_lr = max_lr * 0.1    # Taux d'apprentissage final
warmup_steps = 300       # Période d'échauffement pour le taux d'apprentissage
max_steps = 6000         # Nombre maximal d'itérations

# Optimisations des performances
use_compile = True       # Utilisation de torch.compile pour accélérer l'exécution
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# Configuration du mode distribué
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialisation du tokenizer
enc = tiktoken.get_encoding("gpt2")

# Création des chargeurs de données
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
train_loader = DataLoaderLite(
    B=B, T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="train",
    num_workers=4
)
val_loader = DataLoaderLite(
    B=B, T=T,
    process_rank=ddp_rank,
    num_processes=ddp_world_size,
    split="val"
)

def get_lr(it):
    """Calcule le taux d'apprentissage selon une planification cosinus."""
    if it < warmup_steps:
        return max_lr * it / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def main():
    # Initialisation du modèle
    model = GPT(GPTConfig())
    model.to(device)
    if use_compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    # Configuration de l'optimiseur
    raw_model = model.module if ddp else model
    optimizer = raw_model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=max_lr,
        device=device
    )

    # Préparation du dossier de journalisation
    log_dir = "log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")

    # Boucle principale d'entraînement
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # Phase de validation périodique
        if step % 250 == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                    val_loss_accum += loss.detach() / val_loss_steps
            
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            
            if master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

                # Sauvegarde périodique du modèle
                if step > 0 and (step % 5000 == 0 or last_step):
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': step,
                        'val_loss': val_loss_accum.item()
                    }
                    torch.save(checkpoint, os.path.join(log_dir, f"model_{step:05d}.pt"))

        # Phase d'entraînement
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss = model(x, y)
                loss = loss / grad_accum_steps
            
            loss_accum += loss.detach()
            loss.backward()

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # Optimisation et mise à jour des poids
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()

        # Mesure de performance et journalisation
        dt = time.time() - t0
        tokens_processed = B * T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    if ddp:
        destroy_process_group()

if __name__ == '__main__':
    main()
