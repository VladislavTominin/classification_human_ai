import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from transformers import CamembertModel, CamembertTokenizer
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from human_ai.code.dataset import get_train_dataset, get_train_dataset

root_path = Path('/shared/experiments/lymph_nodes/zones/zones_az/debug/my_project/camembert_embeddings')


def text_transform(data, mode, device='cuda'):
    save_path = root_path / mode
    save_path.mkdir(exist_ok=True)

    tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-large")
    camembert = CamembertModel.from_pretrained("camembert/camembert-large")
    camembert.to(device)
    camembert.eval()

    for sample in tqdm(data):
        text, idx = sample['text'], sample['id']
        tokenized_sentence = tokenizer.tokenize(text)
        encoded_sentence = tokenizer.encode(tokenized_sentence)
        encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0).to(device)
        embedding = camembert(encoded_sentence).last_hidden_state.squeeze().detach().cpu()
        print(embedding.shape)
        torch.save(embedding, save_path / f'tensor{idx}.pt')
        torch.cuda.empty_cache()


data = get_train_dataset()
text_transform(data, 'train')