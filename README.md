
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

1. we use four public Bili_datasets from [NinRec]([NineRec: A Benchmark Dataset Suite for Evaluating Transferable Recommendation | IEEE Journals & Magazine | IEEE Xplore](https://ieeexplore.ieee.org/document/10461053)) paper and download it from [Downstream_datasets.tar.gz - Google 云端硬盘](https://drive.google.com/file/d/15RlthgPczrFbP4U7l6QflSImK5wSGP5K/view).

2. If you want to encode image embeddings by using image encoder,  please down image data from previous link amd put it into `./stage1_content_modality_semantic_alignment/Downstream_dataset/category/category_cover1/`, which `category` is the name of dataset, such as Bili_Cartoon. We also have provided embedding weights.
3. Text and image embeddings weight are ’category_text_llm_stage1.pth‘ and ’category_image_llm_stage1.pth‘, respectively. 
## 3. Pretrained model

you need to download [ChatGLM-4 9B]([glm-4-9b-chat · 模型库 (modelscope.cn)](https://modelscope.cn/models/ZhipuAI/glm-4-9b-chat/file/view/master?fileName=LICENSE&status=0)) and put it into `./stage1_content_modality_semantic_alignment/glm-4-chat.

Also, you need to download pre-trained [google-bert/bert-base-uncased · Hugging Face](https://huggingface.co/google-bert/bert-base-uncased)) and [microsoft/swin-base-patch4-window7-224 · Hugging Face](https://huggingface.co/microsoft/swin-base-patch4-window7-224) and put them into `./stage1_content_modality_semantic_alignment/pretrained_models.

## 4. Function 
`./stage1_content_modality_semantic_alignment/llm_driven_sample.py
: to obtains the llm-driven sample dataset.

`./stage1_content_modality_semantic_alignment/sft_content_encoder.py
: to fine tune text encoder and image encoder jointly

`./stage1_content_modality_semantic_alignment/use_finetune_bert.py
: encode each item text embedding by using text encoder

`./stage1_content_modality_semantic_alignment/use_finetune_swinbase.py
: encode each item image embedding by using image encoder

`./stage2_sequence_preference_learning/SICSRec_main1.py
: to start the training of SICSRec. We have provide pre-trained-weight of ID-encoder in `./stage2_sequence_preference_learning/saved

# SICSRec