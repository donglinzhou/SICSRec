import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from recbole.data.dataset import SequentialDataset
import os
import numpy as np

class IDDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.plm_text_size = config['plm_text_size']
        self.plm_image_size = config['plm_image_size']
        self.plm_suffix = config['plm_suffix']


        # plm_embedding_weight_text, plm_embedding_weight_image = self.load_plm_embedding()
        # self.plm_embedding_text = self.weight2emb(plm_embedding_weight_text, self.plm_text_size)
        # self.plm_embedding_image = self.weight2emb(plm_embedding_weight_image, self.plm_image_size)

    def load_plm_embedding(self, image='_image_llm_', text='_text_llm_'):
        # image 和 text 标识改一下
        feat_path_text = osp.join(self.config['data_path'], f'{self.dataset_name}{text}.{self.plm_suffix}')
        tensor_text = torch.load(feat_path_text, map_location=self.config['device']) #(4724,768)
        padding_embedding = torch.zeros(1, self.plm_text_size).to(self.config['device'])
        tensor_text = torch.cat((padding_embedding, tensor_text), dim=0)

        feat_path_image = osp.join(self.config['data_path'], f'{self.dataset_name}{image}.{self.plm_suffix}')
        tensor_image = torch.load(feat_path_image, map_location=self.config['device']) #(4724,1000)
        padding_embedding = torch.zeros(1, self.plm_image_size).to(self.config['device'])
        tensor_image = torch.cat((padding_embedding, tensor_image), dim=0)

        return tensor_text, tensor_image

    def weight2emb(self, weight, emb_size):
        # weight = torch.tensor(weight).to(self.config['device'])

        plm_embedding = nn.Embedding(self.item_num, emb_size, padding_idx=0)
        plm_embedding.weight.data = weight
        plm_embedding.weight.requires_grad = True

        return plm_embedding
class UniSRecDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.plm_text_size = config['plm_text_size']
        self.plm_image_size = config['plm_image_size']
        self.plm_suffix = config['plm_suffix']
        self.stage = config['stage']
        plm_embedding_weight_text, plm_embedding_weight_image = self.load_plm_embedding()
        self.plm_embedding_text = self.weight2emb(plm_embedding_weight_text, self.plm_text_size)
        self.plm_embedding_image = self.weight2emb(plm_embedding_weight_image, self.plm_image_size)
        self.field2token_id = self.field2token_id
    def load_plm_embedding(self, image='_image_llm_', text='_text_llm_'):
        # image 和 text 标识改一下
        feat_path_text = osp.join(self.config['data_path'], f'{self.dataset_name}{text}{self.stage}.{self.plm_suffix}')
        tensor_text = torch.load(feat_path_text, map_location=self.config['device']) #(4724,768)
        padding_embedding = torch.zeros(1, self.plm_text_size).to(self.config['device'])
        tensor_text = torch.cat((padding_embedding, tensor_text), dim=0)
        mapped_feat_text = torch.zeros((self.item_num, self.plm_text_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat_text[i] = tensor_text[int(token)]

        feat_path_image = osp.join(self.config['data_path'], f'{self.dataset_name}{image}{self.stage}.{self.plm_suffix}')
        tensor_image = torch.load(feat_path_image, map_location=self.config['device']) #(4724,1000)
        padding_embedding = torch.zeros(1, self.plm_image_size).to(self.config['device'])
        tensor_image = torch.cat((padding_embedding, tensor_image), dim=0)
        mapped_feat_image = torch.zeros((self.item_num, self.plm_image_size))
        for i, token in enumerate(self.field2id_token['item_id']):
            if token == '[PAD]': continue
            mapped_feat_image[i] = tensor_image[int(token)]

        return mapped_feat_text, mapped_feat_image

    def weight2emb(self, weight, emb_size):
        plm_embedding = nn.Embedding(self.item_num, emb_size, padding_idx=0)
        plm_embedding.weight.data = weight
        plm_embedding.weight.requires_grad = True

        return plm_embedding
