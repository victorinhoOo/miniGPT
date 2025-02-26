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
    """Génère une réponse en utilisant le modèle finetuné."""
    model.eval()
    
    # Format comme dans le finetuning
    chat_prompt = f"User: {prompt}\nAssistant:"
    
    tokens = torch.tensor(tokenizer.encode(chat_prompt)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits, _ = model(tokens)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Affichage progressif
            text = tokenizer.decode(next_token[0])
            print(text, end='', flush=True)
            
            # Arrêt si on détecte un nouveau tour de dialogue
            if "User:" in text:
                break
    
    print('\n' + '-'*50)

def main():
    print("Loading GPT-2 XL finetuned on OpenAssistant...")
    
    # Charger le modèle de base
    model = GPT.from_pretrained('gpt2-xl')
    
    # Charger le checkpoint finetuné
    checkpoint = torch.load('checkpoints/oasst/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to('cuda')
    model.eval()
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("\nChat initialized. Type 'exit' to quit.")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
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