from transformers import AutoFeatureExtractor, SwinForImageClassification
from PIL import Image, ImageFile
import torch
import os
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomImageDataset(Dataset):
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")], key=lambda x: int(os.path.splitext(x)[0]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.image_folder, filename)
        image = Image.open(image_path).convert("RGB")
        return image, idx

def collate_fn(batch):
    images, indices = zip(*batch)
    images_tensor = feature_extractor(images=list(images), return_tensors="pt")
    return images_tensor, indices

dataset_name = "Bili_Cartoon"
embedding_size = 64
strategy = "stage1"

modal = "llm"
image_folder = f"./Downstream_datasets/{dataset_name}/{dataset_name}_cover1"
model_folder = f"./finetune_{modal}/{dataset_name}/{embedding_size}/{strategy}/swin/"
data_directory = f'./Downstream_datasets/{dataset_name}/'
saved_name = f'{dataset_name}_image_{modal}_{strategy}.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = AutoFeatureExtractor.from_pretrained(model_folder)
model = SwinForImageClassification.from_pretrained(model_folder).to(device)

dataset = CustomImageDataset(image_folder)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)
embeddings = []

for images_batch, indices in dataloader:
    images_batch = {k: v.to(device) for k, v in images_batch.items()}

    with torch.no_grad():
        outputs = model(**images_batch)
        for idx, tensor_image in zip(indices, outputs.logits):
            embeddings.append((idx, tensor_image.detach().cpu()))

embeddings.sort(key=lambda x: x[0])
embeddings_tensor = torch.stack([t[1] for t in embeddings])
torch.save(embeddings_tensor, os.path.join(data_directory, saved_name))

print("Embeddings saved successfully.")
