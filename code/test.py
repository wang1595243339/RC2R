import warnings
warnings.filterwarnings("ignore")
from dataloader_supplychainKG import random_split, collate
from torch.utils.data import DataLoader
from soft_prompt_v3 import JointReasoning
import torch
import numpy as np
from tqdm import tqdm

jr = JointReasoning(g_in_feat=3072,
                    g_n_layers=5,
                    g_hidden_size=1024,
                    g_out_size=3072,
                    n_head=8,
                   )

train_dataset, test_dataset = random_split(data_dir="supplychain-KG/1200text_graph_label01_chain01_ncelabel/", limit=1200, train_ratio=0.9)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate)

#train
optimizer = torch.optim.AdamW(jr.parameters(), lr=1e-2, weight_decay=1e-5)
num_epochs = 5
jr.train()

for epoch in range(num_epochs):
    train_losses = []
    token_losses = []
    align_losses = []
    chain_losses = []

    for batch_data in tqdm(train_dataloader, total=len(train_dataloader), desc=f'epoch-{epoch + 1}'):
        # for batch_index, batch_data in enumerate(train_dataloader):
        graphs, texts, answer_labels, align_labels = batch_data

        align_labels = align_labels[0]
        chain_labels = graphs.ndata['chain_or_not']
        text = texts[0]
        answer_labels = answer_labels[0]
        train_loss, token_loss, align_loss, chain_loss = jr(text, graphs, align_labels, chain_labels, answer_labels)

        # if not train_loss.isnan().item():
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_losses.append(train_loss.item())
        token_losses.append(token_loss.item())
        # align_losses.append(align_loss.item())
        chain_losses.append(chain_loss.item() if type(chain_loss).__name__ == 'Tensor' else chain_loss)

    print(f'epoch: {epoch + 1}, '
          f'total: {round(np.mean(train_losses), 6)}, '
          f'token: {round(np.mean(token_losses), 6)}, '
          # f'align: {round(np.mean(align_losses), 6)}, '
          f'chain: {round(np.mean(chain_losses), 6)}')