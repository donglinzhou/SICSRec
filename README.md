
The source code for our "Self-Supervised Representation Learning with ID-Content
Modality Alignment for Sequential Recommendation"

# 1. Requirements

We conduct experiments on a Tesla V100 PCIe GPU with 32GB memory, and our code is based on the following packages:

- torch == 2.2.1
- transformers == 4.44.2
- recbole == 1.2.0 
- optuna == 3.6.1
- pandas == 2.0.3  
- numpy == 1.20.3  

## 2. Dataset

1. We use four public Bili_datasets from [NinRec]([NineRec: A Benchmark Dataset Suite for Evaluating Transferable Recommendation | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/10461053)) and download it from [Downstream_datasets.tar.gz - Google 云端硬盘](https://drive.google.com/file/d/15RlthgPczrFbP4U7l6QflSImK5wSGP5K/view).
2. If you want to encode image embeddings using an image encoder, please download the image data from the previous link and place it into `./stage1_content_modality_semantic_alignment/Downstream_dataset/category/category_cover1/`, where `category` is the name of the dataset, such as `Bili_Cartoon`. We have also provided embedding weights.

3.  Text and image embeddings weights are `category_text_llm_stage1.pth` and `category_image_llm_stage1.pth`, respectively.
## 3. Pretrained model

You need to download [ChatGLM-4 9B](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat/file/view/master?fileName=LICENSE&status=0) and place it into `./stage1_content_modality_semantic_alignment/glm-4-chat`.

Moreover, you need to download the pre-trained models [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) and [microsoft/swin-base-patch4-window7-224](https://huggingface.co/microsoft/swin-base-patch4-window7-224) and place them into `./stage1_content_modality_semantic_alignment/pretrained_models`.

## 4. Function 
- `./stage1_content_modality_semantic_alignment/llm_driven_sample.py`: to obtain the LLM-driven sample dataset.

- `./stage1_content_modality_semantic_alignment/sft_content_encoder.py`: to fine-tune the text encoder and image encoder jointly.

- `./stage1_content_modality_semantic_alignment/use_finetune_bert.py`: to encode each item's text embedding using the text encoder.

- `./stage1_content_modality_semantic_alignment/use_finetune_swinbase.py`: to encode each item's image embedding using the image encoder.

- `./stage2_sequence_preference_learning/SICSRec_main1.py`: to start the training of SICSRec. We have provided pre-trained weights of the ID-encoder in `./stage2_sequence_preference_learning/saved`.

