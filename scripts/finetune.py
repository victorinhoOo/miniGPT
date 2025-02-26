import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from datasets import load_dataset
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from model.gpt import GPT
import wandb
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Configuration optimisée pour OpenAssistant sur A100
config = {
    'batch_size': 4,                    
    'gradient_accumulation_steps': 8,    
    'learning_rate': 1e-6,              # Learning rate réduit
    'num_epochs': 2,                    # Réduit car dataset plus grand
    'max_length': 512,                  
    'warmup_steps': 2000,               # Augmenté pour dataset plus grand
    'weight_decay': 0.01,               # Weight decay réduit
    'checkpoint_dir': 'checkpoints/oasst',
    'log_interval': 5,
    'max_grad_norm': 0.5,
    'model_name': 'gpt2-xl'
}

# Ajouter la configuration mémoire PyTorch
torch.cuda.set_per_process_memory_fraction(0.95)  # Utiliser 95% de la VRAM
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Éviter la fragmentation

# Optimisations A100
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

class OpenAssistantDataset(Dataset):
    def __init__(self, split="train"):
        print("Loading OpenAssistant dataset...")
        full_dataset = load_dataset("OpenAssistant/oasst1")
        
        # Split train/val (90/10)
        dataset_dict = full_dataset['train'].train_test_split(test_size=0.1, seed=42)
        self.dataset = dataset_dict['train' if split == "train" else 'test']
        print(f"Dataset size: {len(self.dataset)}")
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Format simple
        role = "Assistant" if item['role'] == 'assistant' else "User"
        text = item['text'].strip()
        chat = f"{role}: {text}\n{self.tokenizer.eos_token}"
        
        # Tokenisation avec padding
        encoding = self.tokenizer(
            chat,
            max_length=config['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'val_loss': val_loss,
        'config': model.config,
    }
    
    if is_best:
        path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
    else:
        path = os.path.join(config['checkpoint_dir'], f'model_epoch_{epoch}.pt')
    
    torch.save(checkpoint, path)

def train():
    try:
        # Chargement du modèle XL
        print(f"Loading {config['model_name']} (1.3B parameters)...")
        model = GPT.from_pretrained(config['model_name'])
        
        # Initialisation de wandb pour le suivi
        wandb.init(
            project="gpt2-xl-chatbot",
            config=config,
            notes="Finetuning GPT-2 XL (1.3B) on OpenAssistant"
        )
        
        # Monitoring de la VRAM
        print(f"GPU Memory before loading model: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        model.to(device)
        print(f"GPU Memory after loading model: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        
        # Création des dossiers
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        # Chargement du dataset
        train_dataset = OpenAssistantDataset("train")
        val_dataset = OpenAssistantDataset("validation")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Optimiseur et scheduler
        optimizer = model.configure_optimizers(
            weight_decay=config['weight_decay'],
            learning_rate=config['learning_rate'],
            device=device
        )
        
        num_training_steps = len(train_loader) * config['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        scaler = GradScaler()
        
        for epoch in range(config['num_epochs']):
            model.train()
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]}')
            
            for step, batch in enumerate(progress_bar):
                # Déplacer les données sur GPU
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass avec mixed precision
                with autocast():
                    logits, loss = model(input_ids, targets=labels)
                    loss = loss / config['gradient_accumulation_steps']
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (step + 1) % config['gradient_accumulation_steps'] == 0:
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                
                # Logging
                if step % config['log_interval'] == 0:
                    wandb.log({
                        'train_loss': loss.item(),
                        'epoch': epoch,
                        'step': step,
                        'learning_rate': scheduler.get_last_lr()[0]
                    })
                    
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Validation avec mixed precision
            model.eval()
            val_loss = 0
            
            with torch.no_grad(), autocast():
                for batch in tqdm(val_loader, desc='Validation'):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    logits, loss = model(input_ids, targets=labels)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader)
            wandb.log({'val_loss': val_loss, 'epoch': epoch})
            
            # Sauvegardes
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                val_loss=val_loss,
                is_best=is_best
            )
            
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            val_loss=val_loss,
            is_best=False
        )
    
    finally:
        wandb.finish()

if __name__ == '__main__':
    train() 