import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tiktoken
from torch.nn import functional as F
from model.gpt import GPT

def generate(
    model,
    prompt,
    num_samples=5,
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    device='cuda'
):
    """Génère du texte à partir d'un prompt en utilisant GPT-2."""
    model.eval()
    model.to(device)
    
    # Encode le prompt en tokens
    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_samples, 1)  # (num_samples, seq_len)
    tokens = tokens.to(device)
    
    # Génère token par token
    with torch.no_grad():
        while tokens.size(1) < max_new_tokens:
            # Forward pass
            outputs = model(tokens)  # (B, T, vocab_size)
            if isinstance(outputs, tuple):
                logits = outputs[0]  # Si le modèle retourne un tuple, prend le premier élément
            else:
                logits = outputs
                
            # Prend le dernier token et applique la température
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            
            # Échantillonne le prochain token
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Affiche le texte généré
            for i in range(num_samples):
                generated_text = enc.decode(tokens[i].tolist())
                print(f"\nSample {i+1}:", generated_text, end='', flush=True)
            
            print('\n' + '-'*50)
            
            # Arrête si on génère un EOT
            if (next_token == enc.eot_token).any():
                break
    
    return [enc.decode(tokens[i].tolist()) for i in range(num_samples)]

def main():
    print("Chargement du modèle GPT-2 XL (1.3B paramètres)...")
    model = GPT.from_pretrained('gpt2-xl')  # Version 1.3B
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Un seul prompt
    prompt = "Python is a programming language"
    
    print(f"\nPrompt: {prompt}")
    print("-"*50)
    generate(
        model=model,
        prompt=prompt,
        num_samples=3,        
        max_new_tokens=150,   # Un peu plus long car modèle plus puissant
        temperature=0.7,      # Un peu plus bas pour plus de cohérence
        top_k=50,
        device=device
    )

if __name__ == '__main__':
    main()
