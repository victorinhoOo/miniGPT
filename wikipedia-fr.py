import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from huggingface_hub import login

def tokenize(doc):
    """Tokenize un article Wikipedia et retourne un numpy array uint16"""
    tokens = [eot]  # Token de début
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens, dtype=np.uint16)
    return tokens_np

def write_datafile(filename, tokens_np):
    """Sauvegarde un shard en .npy"""
    np.save(filename, tokens_np)

def process_wikipedia():
    # Configuration
    local_dir = "wikipedia_fr"
    shard_size = int(1e7)  # 10M tokens par shard

    # Authentification Hugging Face
    login("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")

    # Créer le dossier local pour stocker les données tokenisées
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Télécharger Wikipedia FR
    print("Téléchargement de Wikipedia FR...")
    try:
        dataset = load_dataset(
            "wikipedia",
            "20220301.fr",
            split="train",
            trust_remote_code=True,
            cache_dir="./wiki_cache",
            verification_mode="no_checks"
        )
        print(f"Début du téléchargement...")
        print(f"Premier article : {dataset[0]['title']}")
        print(f"Téléchargement réussi! Nombre d'articles : {len(dataset)}")
        print(f"Taille moyenne des articles : {sum(len(doc['text']) for doc in dataset.select(range(100))) / 100:.0f} caractères")
    except Exception as e:
        print(f"Erreur lors du téléchargement : {e}")
        print(f"Type d'erreur : {type(e)}")
        raise

    # Tokeniser et sauvegarder en shards
    print("Tokenisation et écriture des shards...")
    nprocs = max(1, os.cpu_count() // 2)

    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize, dataset, chunksize=16):
            if token_count + len(tokens) < shard_size:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"wiki_fr_{split}_{shard_index:06d}.npy")
                remainder = shard_size - token_count
                if progress_bar:
                    progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                token_count = len(tokens) - remainder
                all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
                all_tokens_np[:token_count] = tokens[remainder:]

        if token_count > 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"wiki_fr_{split}_{shard_index:06d}.npy")
            write_datafile(filename, all_tokens_np[:token_count])

    print(f"{shard_index+1} shards sauvegardés dans {local_dir}")

# Initialiser le tokenizer GPT-2 (global pour être accessible par tokenize())
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

if __name__ == '__main__':
    process_wikipedia() 