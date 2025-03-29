import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import GPT2Tokenizer
from model.gpt import GPT

def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=150,
    temperature=0.7,
    top_k=40,
    device='cuda'
):
    """Génère une réponse conversationnelle à partir d'une entrée utilisateur."""
    model.eval()
    
    # Formatage du prompt pour le dialogue
    chat_prompt = f"User: {prompt}\nAssistant:"
    
    tokens = torch.tensor(tokenizer.encode(chat_prompt)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Prédiction des probabilités du token suivant
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / temperature
            
            # Filtrage par probabilité pour les top-k tokens
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Décodage et affichage incrémental
            text = tokenizer.decode(next_token[0])
            print(text, end='', flush=True)
            
            # Détection de fin de réponse
            if "User:" in text:
                break
    
    print('\n' + '-'*50)

def main():
    print("Chargement du modèle en cours...")
    
    model = GPT.from_pretrained('gpt2-medium')
    model.to('cuda')
    model.eval()
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\nInterface de dialogue initialisée. Tapez 'exit' pour quitter.")
    print("-" * 50)
    
    while True:
        user_input = input("\nVous: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break
            
        print("\nAssistant: ", end="", flush=True)
        generate(
            model=model,
            tokenizer=tokenizer,
            prompt=user_input
        )

if __name__ == '__main__':
    main() 