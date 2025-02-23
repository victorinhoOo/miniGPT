import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import tiktoken
from model.gpt import GPT, GPTConfig

def load_model(checkpoint_path):
    """Charge le modèle depuis un checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model_config = checkpoint['config']
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model'])
    return model

def generate(
    model,
    prompt,
    max_new_tokens=150,
    temperature=0.8,
    top_k=200,
    device='cuda'
):
    """Génère du texte à partir d'un prompt.
    
    Args:
        model: Le modèle GPT
        prompt: Texte d'amorce
        max_new_tokens: Nombre maximum de tokens à générer
        temperature: Contrôle la "créativité" (0.0 = déterministe, 1.0 = aléatoire)
        top_k: Nombre de tokens parmi lesquels choisir à chaque étape
        device: Dispositif de calcul ('cuda' ou 'cpu')
    """
    model.eval()
    model.to(device)
    
    # Encode le prompt en tokens
    enc = tiktoken.get_encoding("gpt2")
    tokens = torch.tensor(enc.encode(prompt)).unsqueeze(0).to(device)
    
    # Génère token par token
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Si la séquence est trop longue, on garde les derniers tokens
            if tokens.size(1) > model.config.block_size:
                tokens = tokens[:, -model.config.block_size:]
            
            # Prédit le prochain token
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / temperature
            
            # Applique le top-k sampling
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Échantillonne le prochain token
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Décode et affiche le token généré
            generated_text = enc.decode(tokens[0].tolist())
            print(generated_text, end='', flush=True)
            
            # Arrête si on génère un caractère de fin
            if next_token.item() == enc.eot_token:
                break
    
    print('\n' + '-'*50 + '\n')
    return generated_text

def main():
    # Configuration
    checkpoint_path = "log/model_06000.pt"  # Dernier checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Charge le modèle
    model = load_model(checkpoint_path)
    
    # Prompts de test
    prompts = [
        "La France est un pays",
        "L'intelligence artificielle permet de",
        "Le but de ce projet est de",
        "Dans un avenir proche, les robots",
        "La programmation en Python"
    ]
    
    # Génère du texte pour chaque prompt
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-"*50)
        generate(
            model=model,
            prompt=prompt,
            max_new_tokens=150,
            temperature=0.8,
            top_k=200,
            device=device
        )

if __name__ == '__main__':
    main()
