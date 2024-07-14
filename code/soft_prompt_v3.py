# encoding: utf-8

import warnings
warnings.filterwarnings("ignore")


from dgl.nn import GraphConv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch import nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import numpy as np


class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    @staticmethod
    def transpose(x):
        return x.transpose(-2, -1)

    @staticmethod
    def normalize(*xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

    def info_nce(self, query, positive_key, labels, temperature=0.1, reduction='mean'):
        query, positive_key = self.normalize(query, positive_key)
        logits = query @ self.transpose(positive_key).contiguous()
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels, reduction=reduction)
        return loss

    def forward(self, query, positive_key, labels):
        return self.info_nce(query, positive_key, labels,
                             temperature=self.temperature,
                             reduction=self.reduction)


class GraphEncoder(nn.Module):
    def __init__(self, in_feat, n_layers, hidden_size, out_size):
        super(GraphEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = F.relu
        # first layer
        self.layers.append(GraphConv(in_feat, hidden_size, bias=True))
        # middle layer
        for _ in range(1, n_layers-1):
            self.layers.append(GraphConv(hidden_size, hidden_size, bias=True))

        # last layer (no activation function)
        self.layers.append(GraphConv(hidden_size, out_size, bias=True))
        self.skip_proj = nn.Linear(hidden_size, out_size)

    def forward(self, graph, node_feats):
        h = node_feats
        num_layers = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            if i == 0:
                h = layer(graph, h)
            else:
                gcn_out = layer(graph, h)
                if h.size(1) != gcn_out.size(1):
                    h = 0.6 * self.skip_proj(h) + 0.4 * gcn_out
                else:
                    h = 0.6 * h + 0.4 * gcn_out  

            if i < num_layers:
                h = self.activation(h)
        # h = unbatch_node_embeddings(graph, h)
        h = h.unsqueeze(0)
        return h


def calc_align_label(llm, tokenizer, input_text, graph, beta=0.88):
    device = llm.device
    token_ids = tokenizer.encode(input_text, return_tensors='pt', padding=False).to(device)

    token_embeddings = llm.model(token_ids).last_hidden_state
    node_feature = graph.ndata['embedding']
    node_feature = node_feature.to(device) if node_feature.device != device else node_feature

    node_feature = torch.unsqueeze(node_feature, dim=0)
    node_embeddings = F.normalize(node_feature, p=2, dim=-1)
    token_embeddings = F.normalize(token_embeddings, p=2, dim=-1)
    cos_sim = torch.bmm(node_embeddings, token_embeddings.transpose(-2, -1))
    label = torch.where(cos_sim > beta, torch.tensor(1), torch.tensor(0))
    return label.to(torch.float32)


class TextEncoderwithSoftPrompt(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        llm = AutoModelForCausalLM.from_pretrained("/public/home/yugy/causal_reasoning/LoRA_adapter").to(self.device)
        self.model = llm.model
        self.lm_head = llm.lm_head
        self.cross_entropy_fn = nn.CrossEntropyLoss()

    def get_token_embeddings(self, token_ids):
        token_ids = token_ids.to(self.device) if token_ids.device != self.device else token_ids
        token_embeddings = self.model.embed_tokens(token_ids) # 3维
        return token_embeddings

    def forward(self, token_embeddings):
        token_embeddings = token_embeddings.to(self.device) if token_embeddings.device != self.device else token_embeddings
        attention_mask = torch.ones([token_embeddings.size(0), token_embeddings.size(1)], dtype=torch.long, device=self.device)

        hidden_states = self.model(inputs_embeds=token_embeddings, attention_mask=attention_mask).last_hidden_state
        logits = self.lm_head(hidden_states)
        return logits

    def calc_loss(self, token_embeddings, token_ids):
        token_embeddings = token_embeddings.to(self.device) if token_embeddings.device != self.device else token_embeddings
        token_ids = token_ids.to(self.device) if token_ids.device != self.device else token_ids

        logits = self.forward(token_embeddings)
        shift_logit = logits[:, token_ids.size(1):-1, :].contiguous()
        loss = self.cross_entropy_fn(shift_logit.view(-1, shift_logit.size(-1)),
                                     token_ids[:, 1:].contiguous().view(-1))
        return loss



class JointReasoning(nn.Module):
    def __init__(self,
                 g_in_feat,
                 g_n_layers,
                 g_hidden_size,
                 g_out_size,
                 n_head
                 ):
        super().__init__()
        self.device = torch.device('cuda:0')
        self.dtype = torch.float32
        self.text_encoder = TextEncoderwithSoftPrompt(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("/public/home/yugy/causal_reasoning/gemma-7b-1t")
        self.graph_encoder = GraphEncoder(g_in_feat, g_n_layers, g_hidden_size, g_out_size).to(self.device)

        # ensure dim of text and node_embedding  same
        self.attention_layer = nn.MultiheadAttention(embed_dim=g_out_size,
                                                     num_heads=n_head,
                                                     dropout=0,
                                                     batch_first=True).to(self.device)
        self.info_nce_fn = InfoNCE()
        self.bce_loss_fn = nn.BCELoss()

    def get_token_embeddings(self, token_ids):
        token_ids = token_ids.to(self.device) if token_ids.device != self.device else token_ids
        return self.text_encoder.get_token_embeddings(token_ids)
    
    def get_node_embeddings(self, graph):
        graph = graph.to(self.device) if graph.device != self.device else graph
        node_feature = graph.ndata['embedding']
        node_feature = node_feature.to(self.dtype) if node_feature.dtype != self.dtype else node_feature
        node_embeddings = self.graph_encoder(graph, node_feature) # 3dim
        return node_embeddings
    
    def get_cross_attention(self, token_embeddings, node_embeddings):
        token_embeddings = token_embeddings.to(self.device) if token_embeddings.device != self.device else token_embeddings
        node_embeddings = node_embeddings.to(self.device) if node_embeddings.device != self.device else node_embeddings

        att_embeddings, att_weights = self.attention_layer(token_embeddings, node_embeddings, node_embeddings,
                                                           average_attn_weights=False)
        return att_embeddings, att_weights
    
    def fuse(self, token_embeddings, graph, align_labels):
        token_embeddings = token_embeddings.to(self.device) if token_embeddings.device != self.device else token_embeddings
        graph = graph.to(self.device) if graph.device != self.device else graph

        node_embeddings = self.get_node_embeddings(graph)
        att_embeddings, att_weights = self.get_cross_attention(token_embeddings, node_embeddings) 
        combined_embeddings = torch.cat((att_embeddings, token_embeddings), dim=1)
        if align_labels is None: 
            align_loss = 0
        else:
            align_labels = align_labels.to(self.device) if align_labels.device != self.device else align_labels
            align_labels = align_labels.to(self.dtype) if align_labels.dtype != self.dtype else align_labels
            align_loss = self.info_nce_fn(node_embeddings, token_embeddings, align_labels)
        return combined_embeddings, att_weights, align_loss
    
    def get_token_ids(self, input_text):
        token_ids = self.tokenizer.encode(input_text, return_tensors='pt', padding=False).to(self.device)
        return token_ids
    
    def get_text(self, token_ids):
        token_ids = token_ids.to('cpu') if token_ids.device.type != 'cpu' else token_ids
        token_ids = token_ids.flatten() if token_ids.dim() != 1 else token_ids
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_risk_score(self, att_weights):
        att_weights = att_weights.to(self.device) if att_weights.device != self.device else att_weights
        mean_weights = att_weights.reshape(-1, att_weights.size(-1)).mean(dim=0)
        score = torch.sigmoid(mean_weights)
        return score

    def forward(self, input_text, graph, align_labels, chain_labels, answer_labels):
        token_ids = self.get_token_ids(input_text)
        graph = graph.to(self.device) if graph.device != self.device else graph
        
        if align_labels is not None:
            align_labels = align_labels.to(self.device) if align_labels.device != self.device else align_labels
            align_labels = align_labels.to(self.dtype) if align_labels.dtype != self.dtype else align_labels

        chain_labels = chain_labels.to(self.device) if chain_labels.device != self.device else chain_labels
        chain_labels = chain_labels.to(self.dtype) if chain_labels.dtype != self.dtype else chain_labels
        chain_labels = chain_labels.flatten() if chain_labels.dim() != 1 else chain_labels

        token_embeddings = self.get_token_embeddings(token_ids)
        combined_embeddings, att_weights, align_loss = self.fuse(token_embeddings, graph, None)
        token_loss = self.text_encoder.calc_loss(combined_embeddings, token_ids)

        if answer_labels == 1:
            risk_score = self.get_risk_score(att_weights)
            chain_loss = self.bce_loss_fn(risk_score, chain_labels)
        else:
            chain_loss = 0

        ttl_loss = token_loss + chain_loss
        return ttl_loss, token_loss, align_loss, chain_loss

    def generate(self, token_ids, graph):
        token_ids = token_ids.to(self.device) if token_ids.device != self.device else token_ids
        graph = graph.to(self.device) if graph.device != self.device else graph

        token_embeddings = self.get_token_embeddings(token_ids)
        combined_embeddings, att_weights, _ = self.fuse(token_embeddings, graph, None)
        token_logits = self.text_encoder(combined_embeddings)

        predictions = token_logits[:, -1, :]
        return predictions, att_weights


def trainer(train_dataloader):
    jr = JointReasoning(
                        g_in_feat=3072,
                        g_n_layers=3,
                        g_hidden_size=1024,
                        g_out_size=3072,
                        n_head=8)

    optimizer = torch.optim.AdamW(jr.parameters(), lr=1e-2, weight_decay=1e-5)
    num_epochs = 15
    jr.train()
    # WRITER = SummaryWriter('/public/home/yugy/causal_reasoning/experiment_20240330_nobg')
    for epoch in range(num_epochs):
        train_losses = []
        for batch_index, batch_data in enumerate(train_dataloader):
            graphs, texts, _, align_labels = batch_data
            
            align_labels = align_labels[0]
            chain_labels = graphs.ndata['chain_or_not']

            text = texts[0]
            train_loss, text_loss, align_loss, chain_loss = jr(text, graphs, align_labels, chain_labels)

            if not train_loss.isnan().item():
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                train_losses.append(train_loss.item())
            
                print(f'epoch: {epoch}, batch: {batch_index}, '
                    f'total: {round(train_loss.item(), 6)}, '
                    f'text: {round(text_loss.item(), 6)}, '
                    f'align: {round(align_loss.item(), 6)}, '
                    f'chain: {round(chain_loss.item(), 6)}')

        train_losses = round(np.mean(train_losses), 6)
        print(f'epoch: {epoch},  total: {round(train_loss.item(), 6)}')
        # WRITER.add_scalars('Loss', {'train': train_losses}, epoch)

    # WRITER.close()






