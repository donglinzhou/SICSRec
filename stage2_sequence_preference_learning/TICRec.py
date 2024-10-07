import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import TransformerEncoder, TransformerDecoder
from recbole.model.abstract_recommender import SequentialRecommender


class MLPAdapter(nn.Module):
    def __init__(self, input_hidden_size, hidden_size):
        super(MLPAdapter, self).__init__()
        self.text_hidden_size = input_hidden_size
        self.hidden_size = hidden_size

        # Define the layers
        self.fc1 = nn.Linear(self.text_hidden_size, self.text_hidden_size // 2)
        self.fc2 = nn.Linear(self.text_hidden_size // 2, self.hidden_size)

    def forward(self, x):
        # Apply the first fully connected layer and a ReLU activation function
        x = F.relu(self.fc1(x))
        # Apply the second fully connected layer
        # x = self.fc1(x)
        x = self.fc2(x)
        return x
class Projection(nn.Module):
    def __init__(self, input_hidden_size, hidden_size):
        super(Projection, self).__init__()
        self.text_hidden_size = input_hidden_size
        self.hidden_size = hidden_size

        # Define the layers
        self.fc1 = nn.Linear(self.text_hidden_size,  self.hidden_size)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return x


class ID_SeqEncoder(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        loss = self.loss_fct(logits, pos_items)
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores


class Content_SeqEncoder(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.device = config['device']
        self.temperature = config['temperature']
        self.plm_text_embedding = copy.deepcopy(dataset.plm_embedding_text)
        self.plm_image_embedding = copy.deepcopy(dataset.plm_embedding_image)

        self.text_hidden_size = config['text_size']
        self.image_hidden_size = config['image_size']
        self.alpha = config['alpha']
        self.lamba = config['lamba']
        self.lora_rank = config['lora_rank']

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.loss_fct = nn.CrossEntropyLoss()

        self.text_adapter = MLPAdapter(self.text_hidden_size, self.hidden_size)
        self.image_adapter = MLPAdapter(self.image_hidden_size, self.hidden_size)

        self.content_LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.content_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.content_trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            add_lora=False
        )

        self.item_dropout = nn.Dropout(config['item_dropout'])
        self.image_mlp = nn.Linear(self.hidden_size, self.hidden_size)
        self.text_mlp = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, item_seq, item_seq_len, uni_modality_emb):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        extended_attention_mask = self.get_attention_mask(item_seq)

        # # content embedding sequence
        uni_modality_emb = uni_modality_emb + position_embedding
        uni_modality_emb = self.content_LayerNorm(uni_modality_emb)
        uni_modality_emb = self.content_dropout(uni_modality_emb)
        uni_modality_trm_output = self.content_trm_encoder(uni_modality_emb,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers=True)
        uni_modality_output = uni_modality_trm_output[-1]
        uni_modality_pref = self.gather_indexes(uni_modality_output, item_seq_len - 1)

        return uni_modality_pref  # [B H]

    def seq_item_alignment_task(self, seq_output, same_pos_id, pos_items_emb):
        pos_items_emb = F.normalize(pos_items_emb, dim=1)
        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device),
                                 neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        text_emb = self.text_adapter(self.plm_text_embedding(item_seq))
        image_emb = self.image_adapter(self.plm_image_embedding(item_seq))

        combined_text_image_emb = text_emb + image_emb
        norm_combined_emb = F.normalize(combined_text_image_emb, p=2, dim=-1)
        seq_output = self.forward(item_seq, item_seq_len, norm_combined_emb)
        seq_output = F.normalize(seq_output, dim=1)

        pos_items = interaction[self.POS_ITEM_ID]
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))
        pos_items_text_emb = self.text_adapter(self.plm_text_embedding(pos_items))
        pos_items_image_emb = self.image_adapter(self.plm_image_embedding(pos_items))
        combined_pos_item_emb = pos_items_text_emb + pos_items_image_emb
        norm_combined_pos_item_emb = F.normalize(combined_pos_item_emb, p=2, dim=-1)

        ID2Mo_loss = self.seq_item_alignment_task(seq_output, same_pos_id, norm_combined_pos_item_emb)

        text_embs = self.text_adapter(self.plm_text_embedding.weight)
        image_embs = self.image_adapter(self.plm_image_embedding.weight)
        item_embs = text_embs + image_embs
        item_embs = F.normalize(item_embs, p=2, dim=-1)
        logits = torch.matmul(seq_output, item_embs.transpose(0, 1)) / self.temperature
        loss = self.loss_fct(logits, pos_items)

        # Calculate L2 regularization loss
        l2_regularization = torch.tensor(0.0).to(logits.device)
        for param in self.plm_text_embedding.parameters():
            l2_regularization += torch.norm(param, p=2)
        for param in self.plm_image_embedding.parameters():
            l2_regularization += torch.norm(param, p=2)

        return loss + self.lamba * l2_regularization + self.alpha * ID2Mo_loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        text_emb = self.text_adapter(self.plm_text_embedding(item_seq))
        image_emb = self.image_adapter(self.plm_image_embedding(item_seq))

        combined_text_image_emb = text_emb + image_emb
        norm_combined_emb = F.normalize(combined_text_image_emb, p=2, dim=-1)
        seq_output = self.forward(item_seq, item_seq_len, norm_combined_emb)
        seq_output = F.normalize(seq_output, dim=1)

        text_embs = self.text_adapter(self.plm_text_embedding.weight)
        image_embs = self.image_adapter(self.plm_image_embedding.weight)
        item_embs = text_embs + image_embs
        item_embs = F.normalize(item_embs, p=2, dim=-1)

        scores = torch.matmul(seq_output,  item_embs.transpose(0, 1))

        return scores


class SICSRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.device = config['device']
        self.temperature = config['temperature']

        self.plm_text_embedding = copy.deepcopy(dataset.plm_embedding_text)
        self.plm_image_embedding = copy.deepcopy(dataset.plm_embedding_image)

        self.text_hidden_size = config['text_size']
        self.image_hidden_size = config['image_size']
        self.train_stage = config['train_stage']

        self.alpha = config['alpha']
        self.lamba = config['lamba']
        self.lora_rank = config['lora_rank']

        # x ID-based model
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            add_lora=True,
            lora_rank=self.lora_rank
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.loss_fct = nn.CrossEntropyLoss()

        self.text_adapter = MLPAdapter(self.text_hidden_size, self.hidden_size)
        self.image_adapter = MLPAdapter(self.image_hidden_size, self.hidden_size)

        self.aggregation_layer = nn.Linear(self.hidden_size * 3, self.hidden_size)

        self.content_LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.content_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.content_trm_decoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
            add_lora=False
        )
        self.query_LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.query_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.value_LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.value_dropout = nn.Dropout(self.hidden_dropout_prob)

        self.id_content_trm_decoder = TransformerDecoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )

        self.item_dropout = nn.Dropout(config['item_dropout'])

    def image_text_alignment_task(self, text_features, image_features):
        batch_size, seq_length, hidden_size = text_features.size()

        text_features = text_features.view(batch_size * seq_length, hidden_size)
        image_features = image_features.view(batch_size * seq_length, hidden_size)
        similarity_matrix = torch.matmul(text_features, image_features.t()) / self.temperature

        pos_samples = torch.diag(similarity_matrix)

        mask = torch.eye(batch_size * seq_length, device=self.device).bool()
        neg_samples = similarity_matrix.masked_fill(mask, float('-inf'))

        logits = torch.cat([pos_samples.view(batch_size * seq_length, 1), neg_samples], dim=1)
        targets = torch.zeros(batch_size * seq_length).long().to(self.device)
        loss = self.loss_fct(logits, targets)

        return loss

    def forward(self, item_seq, item_emb, item_seq_len, uni_modality_emb):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        input_emb = item_emb + position_embedding
        # id embedding sequence
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        id_output = trm_output[-1]
        id_output_pref = self.gather_indexes(id_output, item_seq_len - 1)

        # # content embedding sequence
        uni_modality_emb = uni_modality_emb + position_embedding
        uni_modality_emb = self.content_LayerNorm(uni_modality_emb)
        uni_modality_emb = self.content_dropout(uni_modality_emb)
        uni_modality_trm_output = self.content_trm_decoder(uni_modality_emb,
                                                           extended_attention_mask,
                                                           output_all_encoded_layers=True)
        uni_modality_output = uni_modality_trm_output[-1]
        uni_modality_pref = self.gather_indexes(uni_modality_output, item_seq_len - 1)

        id_output = self.query_LayerNorm(id_output)
        id_output = self.query_dropout(id_output)
        uni_modality_output = self.value_LayerNorm(uni_modality_output)
        uni_modality_output = self.value_dropout(uni_modality_output)
        id_content_output = self.id_content_trm_decoder(id_output, uni_modality_output,
                                                        extended_attention_mask,
                                                        output_all_encoded_layers=True
                                                        )
        id_content_output = id_content_output[-1]
        id_content_pref = self.gather_indexes(id_content_output, item_seq_len - 1)
        output = self.aggregation_layer(
            torch.cat([id_output_pref, uni_modality_pref, id_content_pref], dim=-1)
        )
        return output # [B H]

    def seq_item_alignment_task(self, seq_output, same_pos_id, pos_items_emb):
        pos_items_emb = F.normalize(pos_items_emb, dim=1)
        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device),
                                 neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        id_emb = self.item_embedding(item_seq)

        text_emb = self.text_adapter(self.plm_text_embedding(item_seq))
        image_emb = self.image_adapter(self.plm_image_embedding(item_seq))

        combined_text_image_emb = text_emb + image_emb
        norm_combined_emb = F.normalize(combined_text_image_emb, p=2, dim=-1)
        seq_output = self.forward(item_seq, id_emb, item_seq_len, norm_combined_emb)
        seq_output = F.normalize(seq_output, dim=1)

        pos_items = interaction[self.POS_ITEM_ID]
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))
        pos_items_text_emb = self.text_adapter(self.plm_text_embedding(pos_items))
        pos_items_image_emb = self.image_adapter(self.plm_image_embedding(pos_items))
        combined_pos_item_emb = pos_items_text_emb + pos_items_image_emb
        norm_combined_pos_item_emb = F.normalize(combined_pos_item_emb, p=2, dim=-1)
        ID2Mo_loss = self.seq_item_alignment_task(seq_output, same_pos_id, norm_combined_pos_item_emb)

        test_item_emb = self.item_embedding.weight
        test_item_emb = self.item_dropout(test_item_emb)
        test_item_emb = F.normalize(test_item_emb, dim=1)
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
        loss = self.loss_fct(logits, pos_items)

        # Calculate L2 regularization loss
        l2_regularization = torch.tensor(0.0).to(logits.device)
        for param in self.item_embedding.parameters():
            l2_regularization += torch.norm(param, p=2)
        for param in self.plm_text_embedding.parameters():
            l2_regularization += torch.norm(param, p=2)
        for param in self.plm_image_embedding.parameters():
            l2_regularization += torch.norm(param, p=2)

        return loss + self.lamba * l2_regularization + self.alpha * ID2Mo_loss

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        id_emb = self.item_embedding(item_seq)
        text_emb = self.text_adapter(self.plm_text_embedding(item_seq))
        image_emb = self.image_adapter(self.plm_image_embedding(item_seq))

        combined_text_image_emb = text_emb + image_emb
        norm_combined_emb = F.normalize(combined_text_image_emb, p=2, dim=-1)
        seq_output = self.forward(item_seq, id_emb, item_seq_len, norm_combined_emb)
        seq_output = F.normalize(seq_output, dim=1)

        itemID_scores = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        return itemID_scores