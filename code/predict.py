import torch
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import json

import dgl
import torch
from pyecharts import options as opts
from pyecharts.charts import Sankey
from sklearn.metrics import accuracy_score, roc_auc_score


def iter_generate(model, input_text, graph, max_length=50, temperature=1.0, top_k=3):
    graph = graph.to(model.device)
    input_ids = model.get_token_ids(input_text)

    # 初始化生成的文本为输入文本
    generated = input_ids

    with torch.no_grad():  # 不计算梯度
        for _ in range(max_length):
            predictions, att_weights = model.generate(generated, graph)

            # 采用最后一个时间步的预测结果
            next_token_logits = predictions / temperature

            # 应用top-k采样
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            next_token = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)
            # 将新生成的词添加到生成的文本中
            generated = torch.cat((generated, next_token), dim=-1)
            # attention_mask = torch.cat([attention_mask, torch.ones([attention_mask.size(0), 1], dtype=torch.float32).to(attention_mask.device)], dim=-1)
            # 检查是否生成了结束符
            if next_token == model.tokenizer.eos_token_id:
                break

    generated_text = model.get_text(generated)
    risk_score = model.get_risk_score(att_weights)
    return generated_text, risk_score


def remove_answer(text):
    parts = text.split("\n")
    test_text_no_answer = parts[0] + "\n" + parts[1] + "\n" + parts[2] + "\n"
    return test_text_no_answer


def calc_iou_score(true_chain, risk_score):
    true_chain = set(torch.argwhere(true_chain > 0.5).flatten().cpu().tolist())
    pred_chain = set(torch.argwhere(risk_score > 0.5).flatten().cpu().tolist())
    iou_score = len(true_chain & pred_chain) / len(true_chain | pred_chain)
    return iou_score


jr.eval()
true_answers = []
pred_answers = []
explanations = []
iou_scores = []
risk_scores = []

for batch_data in tqdm(test_dataloader):

    graphs, texts, true_answer, _ = batch_data
    text = texts[0]
    text = remove_answer(text)
    true_answers.append(true_answer[0])

    generated_text, risk_score = iter_generate(jr, text, graphs, max_length=100)
    explanations.append(generated_text)
    risk_scores.append(risk_score)

    partial_answer = generated_text.split("\n")[3:4][0].lower()
    if "no" in partial_answer:
        pred_answers.append(0)
    elif "yes" in partial_answer:
        pred_answers.append(1)
    else:
        pred_answers.append(2)
    iou_score = calc_iou_score(graphs.ndata['chain_or_not'], risk_score)
    iou_scores.append(iou_score)

acc_score = accuracy_score(y_true=true_answers, y_pred=pred_answers)
auc_score = roc_auc_score(true_answers, pred_answers)
iou_score = np.mean(iou_scores)