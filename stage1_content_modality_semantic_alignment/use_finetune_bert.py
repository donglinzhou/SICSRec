import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

category_name = 'Bili_Cartoon'
strategy = "stage1"
embedding_size = 64

data_directory = f'./Downstream_datasets/{category_name}/'
modal = 'llm'
batch_size = 1024
path = f"./finetune_{modal}/{category_name}/{embedding_size}/{strategy}/bert/"
saved_name = f'{category_name}_text_{modal}_{strategy}.pth'

meta_file_path = os.path.join(data_directory, f'{category_name}_item_sort.csv')
column_names = ['item_id', 'chinese_title', 'english_title']

df = pd.read_csv(meta_file_path, header=None, names=column_names)
df.sort_values(by='item_id', inplace=True)
df.reset_index(drop=True, inplace=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)


tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
model = AutoModel.from_pretrained(path)
model.to(device)

information_embedding = []

batch_num = (df.shape[0] + batch_size - 1) // batch_size

for i in tqdm(range(batch_num), total=batch_num, ncols=70, leave=False, unit='b'):
    batch = df.loc[i * batch_size:(i + 1) * batch_size - 1, 'chinese_title'].tolist()

    inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=256)
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    information_embedding.extend(cls_embeddings.cpu().tolist())

embedding_tensor = torch.tensor(information_embedding)
torch.save(embedding_tensor, os.path.join(data_directory, saved_name))

print(f'Embedding vectors successfully saved')
