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
    """Génère du texte avec le modèle en partant d'une amorce textuelle."""
    model.eval()
    model.to(device)
    
    # Convertit le texte en tokens pour le modèle
    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(enc.encode(prompt), dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_samples, 1)  # Dimension: (num_samples, seq_len)
    tokens = tokens.to(device)
    
    # Processus de génération séquentielle
    with torch.no_grad():
        while tokens.size(1) < max_new_tokens:
            # Passage avant dans le modèle
            outputs = model(tokens)  # Dimension: (B, T, vocab_size)
            if isinstance(outputs, tuple):
                logits = outputs[0]  # Extraction des logits si format tuple
            else:
                logits = outputs
                
            # Application de la température sur le dernier token
            logits = logits[:, -1, :] / temperature
            
            # Échantillonnage top-k pour diversifier la génération
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            
            # Sélection du token suivant selon les probabilités
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Affichage des résultats intermédiaires
            for i in range(num_samples):
                generated_text = enc.decode(tokens[i].tolist())
                print(f"\nSample {i+1}:", generated_text, end='', flush=True)
            
            print('\n' + '-'*50)
            
            # Fin de génération si token de fin détecté
            if (next_token == enc.eot_token).any():
                break
    
    return [enc.decode(tokens[i].tolist()) for i in range(num_samples)]

def main():
    print("Chargement du modèle ...")
    model = GPT.from_pretrained('gpt2-medium')  # Modèle de taille moyenne
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Texte d'entrée pour la génération
    prompt = "Python is a programming language"
    
    print(f"\nPrompt: {prompt}")
    print("-"*50)
    generate(
        model=model,
        prompt=prompt,
        num_samples=3,        
        max_new_tokens=150,   # Longueur de génération adaptée
        temperature=0.7,      # Contrôle de la diversité textuelle
        top_k=50,
        device=device
    )

if __name__ == '__main__':
    main()
