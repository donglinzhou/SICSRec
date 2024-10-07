import os
import torch
from torch import nn, optim
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor, SwinForImageClassification

class CustomDataset(Dataset):
    def __init__(self, image_folder, item_pairs_cn_path, item_pairs_id_path):
        self.image_folder = image_folder

        with open(item_pairs_cn_path, 'r', encoding='utf-8') as f:
            self.text_pairs = [line.strip().split(',') for line in f.readlines()]

        with open(item_pairs_id_path, 'r') as f:
            self.image_pairs = [line.strip().split(',') for line in f.readlines()]

    def __len__(self):
        return len(self.text_pairs)

    def __getitem__(self, idx):

        if idx >= len(self.image_pairs):
            raise IndexError(f"Index {idx} out of range for image_pairs.")

        text_pair = self.text_pairs[idx]
        image_pair = [os.path.join(self.image_folder, img_id + ".jpg") for img_id in self.image_pairs[idx]]
        images = [Image.open(image_path).convert("RGB") for image_path in image_pair]

        return text_pair, images, idx


def collate_fn(batch):
    text_pairs, images_pairs, indices = zip(*batch)

    target_texts = [pair[0] for pair in text_pairs]
    pos_texts = [pair[1] for pair in text_pairs]

    target_images = [pair[0] for pair in images_pairs]
    pos_images = [pair[1] for pair in images_pairs]

    target_images_tensors = feature_extractor(images=target_images, return_tensors="pt").pixel_values
    pos_images_tensors = feature_extractor(images=pos_images, return_tensors="pt").pixel_values

    return (target_texts, pos_texts), (target_images_tensors, pos_images_tensors), indices


class TextAdapter(nn.Module):
    def __init__(self):
        super(TextAdapter, self).__init__()
        self.bert_mlp = nn.Sequential(
            nn.Linear(bert_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, bert_outputs_1, bert_outputs_2):
        text_embeddings_1 = self.bert_mlp(bert_outputs_1)
        text_embeddings_2 = self.bert_mlp(bert_outputs_2)

        return text_embeddings_1, text_embeddings_2

class ImageAdapter(nn.Module):
    def __init__(self):
        super(ImageAdapter, self).__init__()

        self.swin_mlp = nn.Sequential(
            nn.Linear(swin_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, swin_outputs_1, swin_outputs_2):
        swin_embeddings_1 = self.swin_mlp(swin_outputs_1)
        swin_embeddings_2 = self.swin_mlp(swin_outputs_2)

        return swin_embeddings_1, swin_embeddings_2

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, target_emb, pos_emb):
        batch_size = target_emb.size(0)

        all_emb = torch.cat([pos_emb, target_emb], dim=0)
        similarity_matrix = torch.matmul(target_emb, all_emb.T) / self.temperature  # (batch_size, 2 * batch_size)

        labels = torch.arange(batch_size).to(target_emb.device)

        loss = self.criterion(similarity_matrix, labels)
        return loss


class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(CLIPLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text_emb, image_emb):
        batch_size = text_emb.size(0)

        similarity_matrix = torch.matmul(text_emb, image_emb.T) / self.temperature

        labels = torch.arange(batch_size).to(text_emb.device)

        loss = self.criterion(similarity_matrix, labels) + self.criterion(similarity_matrix.T, labels)
        return loss


def save_finetuned_model(bert_model, bert_tokenizer, swin_model, feature_extractor):
    bert_model.save_pretrained(bert_save_path)
    bert_tokenizer.save_pretrained(bert_save_path)
    print(f"微调后的 BERT 模型和 tokenizer 已保存到 {bert_save_path}！")

    swin_model.save_pretrained(swin_save_path)
    feature_extractor.save_pretrained(swin_save_path)
    print(f"Swin 模型的 preprocessor_config.json 已保存到 {swin_save_path}！")
    print(f"微调后的 Swin 模型已保存到 {swin_save_path}！")


ImageFile.LOAD_TRUNCATED_IMAGES = True

bert_output_dim = 768
swin_output_dim = 1000
embedding_dim = 64
batch_size = 128
num_epochs = 10
lr = 1e-5
category = 'Bili_Cartoon'

image_folder = f"./Downstream_datasets/{category}/{category}_cover1/"
item_pairs_cn_path = f'./Downstream_datasets/{category}/item_pairs_cn.txt'
item_pairs_id_path = f'./Downstream_datasets/{category}/item_pairs_id.txt'

strategy = "stage1_ab_wo_t2i"
bert_save_path = f"./finetune_llm/{category}/{embedding_dim}/{strategy}/bert/"
swin_save_path = f"./finetune_llm/{category}//{embedding_dim}/{strategy}/swin/"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = CustomDataset(image_folder, item_pairs_cn_path, item_pairs_id_path)

bert_path = './pretrained_models/bert-base-cased'
bert_tokenizer = AutoTokenizer.from_pretrained(bert_path, use_fast=True)
bert_model = AutoModel.from_pretrained(bert_path)

model_folder = "./pretrained_models/swin_base"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_folder)
swin_model = SwinForImageClassification.from_pretrained(model_folder)

for param in bert_model.parameters():
    param.requires_grad = False

for param in swin_model.parameters():
    param.requires_grad = False

for param in bert_model.encoder.layer[-1].parameters():
    param.requires_grad = True

for param in bert_model.pooler.parameters():
    param.requires_grad = True

for param in swin_model.swin.encoder.layers[-1].parameters():
    param.requires_grad = True

for param in swin_model.classifier.parameters():
    param.requires_grad = True

# for name, param in swin_model.named_parameters():
#     print(name, param.requires_grad)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


bert_model = bert_model.to(device)
swin_model = swin_model.to(device)
textAdapter = TextAdapter().to(device)
imageAdapter = ImageAdapter().to(device)
optimizer = optim.SGD(filter(lambda p: p.requires_grad, list(bert_model.parameters()) + list(swin_model.parameters())),
                       lr=lr)


criterion_infonce = InfoNCELoss().to(device)
criterion_clip = CLIPLoss().to(device)
os.makedirs(bert_save_path, exist_ok=True)
os.makedirs(swin_save_path, exist_ok=True)
for epoch in range(num_epochs):
    bert_model.train()
    swin_model.train()
    running_loss = 0.0

    for text_pairs, images_batch, indices in dataloader:
        target_texts, pos_texts = text_pairs
        target_images, pos_images = images_batch

        target_inputs = bert_tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        pos_inputs = bert_tokenizer(pos_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        target_text_emb = bert_model(**target_inputs).pooler_output  # (batch_size, hidden_size)
        pos_text_emb = bert_model(**pos_inputs).pooler_output  # (batch_size, hidden_size)

        target_images = target_images.to(device)
        pos_images = pos_images.to(device)

        target_image_emb = swin_model(target_images).logits  # (batch_size, hidden_size)
        pos_image_emb = swin_model(pos_images).logits   # (batch_size, hidden_size)

        target_text_emb, pos_text_emb = textAdapter(target_text_emb, pos_text_emb)
        target_image_emb, pos_image_emb = imageAdapter(target_image_emb, pos_image_emb)

        loss_text = criterion_infonce(target_text_emb, pos_text_emb)
        loss_image = criterion_infonce(target_image_emb, pos_image_emb)
        loss_clip = criterion_clip(target_text_emb, target_image_emb)

        # 总损失
        total_loss = loss_text + loss_image + loss_clip
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += total_loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")

save_finetuned_model(bert_model, bert_tokenizer, swin_model, feature_extractor)


# ----------------- 微调效果验证 -----------------
#
# def calculate_cosine_similarity(emb1, emb2):
#     emb1 = emb1.detach().cpu().numpy()
#     emb2 = emb2.detach().cpu().numpy()
#     return cosine_similarity(emb1, emb2)
#
# # 加载微调前模型
# bert_model_before = AutoModel.from_pretrained('./pretrained_models/bert-base-cased').to(device)
# swin_model_before = SwinForImageClassification.from_pretrained('./pretrained_models/swin_base').to(device)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
# text_similarity_before, text_similarity_after,loss_clip_before,loss_clip_after =0,0,0,0
# image_similarity_before, image_similarity_after = 0, 0
# times = 0
# # 验证一个 batch
# for text_pairs, images_batch, indices in dataloader:
#     target_texts, pos_texts = text_pairs
#     target_images, pos_images = images_batch
#     target_inputs = bert_tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True).to(device)
#     pos_inputs = bert_tokenizer(pos_texts, return_tensors="pt", padding=True, truncation=True).to(device)
#     times += 1
#     # 文本对处理：分别处理目标样本和正样本
#     with torch.no_grad():
#         # 微调前：获取文本嵌入
#         target_text_emb_p = bert_model_before(**target_inputs).pooler_output  # (batch_size, hidden_size)
#         pos_text_emb_p = bert_model_before(**pos_inputs).pooler_output  # (batch_size, hidden_size)
#         text_similarity_before += calculate_cosine_similarity(target_text_emb_p, pos_text_emb_p).mean()
#
#         # 微调后：获取文本嵌入
#         target_text_emb_n = bert_model(**target_inputs).pooler_output  # (batch_size, hidden_size)
#         pos_text_emb_n = bert_model(**pos_inputs).pooler_output  # (batch_size, hidden_size)
#         text_similarity_after += calculate_cosine_similarity(target_text_emb_n, pos_text_emb_n).mean()
#
#     # 图像对处理：分别处理目标样本和正样本
#     target_images = target_images.to(device)
#     pos_images = pos_images.to(device)
#
#     with torch.no_grad():
#         # 微调前：获取图像嵌入
#         target_image_emb_p = swin_model_before(target_images).logits  # (batch_size, hidden_size)
#         pos_image_emb_p = swin_model_before(pos_images).logits  # (batch_size, hidden_size)
#         image_similarity_before += calculate_cosine_similarity(target_image_emb_p, pos_image_emb_p).mean()
#
#         # 微调后：获取图像嵌入
#         target_image_emb_n = swin_model(target_images).logits  # (batch_size, hidden_size)
#         pos_image_emb_n = swin_model(pos_images).logits  # (batch_size, hidden_size)
#         image_similarity_after += calculate_cosine_similarity(target_image_emb_n, pos_image_emb_n).mean()
#
#     # MLP 和 CLIP 损失计算
#     with torch.no_grad():
#         # 微调前的嵌入适配与损失计算
#         target_text_emb_adapted_p, pos_text_emb_adapted_p = textAdapter(target_text_emb_p, pos_text_emb_p)
#         target_image_emb_adapted_p, pos_image_emb_adapted_p = imageAdapter(target_image_emb_p, pos_image_emb_p)
#         loss_clip_before += criterion_clip(target_text_emb_adapted_p, target_image_emb_adapted_p).item()
#
#         # 微调后的嵌入适配与损失计算
#         target_text_emb_adapted_n, pos_text_emb_adapted_n = textAdapter(target_text_emb_n, pos_text_emb_n)
#         target_image_emb_adapted_n, pos_image_emb_adapted_n = imageAdapter(target_image_emb_n, pos_image_emb_n)
#         loss_clip_after += criterion_clip(target_text_emb_adapted_n, target_image_emb_adapted_n).item()
#
# print("微调前的文本相似度: ", text_similarity_before / times)
# print("微调后的文本相似度: ", text_similarity_after / times)
# print("微调前的图片相似度: ", image_similarity_before / times)
# print("微调后的图片相似度: ", image_similarity_after / times)
# print("微调前的CLIP loss: ", loss_clip_before  / times)
# print("微调后的CLIP loss: ", loss_clip_after  / times)