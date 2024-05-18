FinDKG数据来源于https://github.com/xiaohui-victor-li/FinDKG，对数据进行整理，汇总600条数据。

数据：
'text'内容样式：背景描述 \n 请用YES或者NO回答因果问题 \n 因果问题 \n 回答（Yes or No）\n 解释<eos>
'graph' 是子图的信息，
 'node_feature'是节点特征
'label'是回答Yes or No

样例：
{'text': "A bill impacts the federal government, which in turn impacts workers. These workers then participate in unemployment benefits, which invest in the labor market.\nPlease answer the following causal reasoning questions using Yes or No based on the contextual content above:\nDoes the passing of a bill indirectly invest in the labor market through its impact on the federal government and workers' participation in unemployment benefits?\nYes\nThe causal chain starts with a bill impacting the federal government. This impact on the federal government subsequently affects workers, leading them to participate in unemployment benefits. The participation in unemployment benefits, in turn, invests in the labor market. Therefore, the passing of a bill indirectly invests in the labor market through this chain of impacts and participations.<eos>",
 'graph': Graph(num_nodes=12, num_edges=13,
       ndata_schemes={'node_type': Scheme(shape=(), dtype=torch.int32), 'embedding': Scheme(shape=(3072,), dtype=torch.float16), '_ID': Scheme(shape=(), dtype=torch.int32)}
       edata_schemes={'rel_type': Scheme(shape=(), dtype=torch.int32), 'time': Scheme(shape=(), dtype=torch.float32), 'train_mask': Scheme(shape=(), dtype=torch.bool), 'val_mask': Scheme(shape=(), dtype=torch.bool), 'test_mask': Scheme(shape=(), dtype=torch.bool), '_ID': Scheme(shape=(), dtype=torch.int32)}),
 'node_feature': tensor([[ -9.5469,   0.4131,   1.8926,  ...,   2.9062,   0.5283,   1.6230],
         [-11.8906,  -0.2205,   1.3135,  ...,   0.2656,   1.9580,   1.2656],
         [-13.4688,   0.0547,  -1.6963,  ...,   0.7031,  -1.5645,   1.9707],
         ...,
         [ -9.3438,   0.1959,  -1.2178,  ...,   2.1230,  -0.1973,   2.3809],
         [ -8.8906,  -0.1771,  -3.0391,  ...,  -0.1696,   1.4268,   1.1758],
         [-14.1641,   0.2329,   0.7754,  ...,  -1.1250,  -2.0352,  -0.6123]],
        dtype=torch.float16),
 'label': 1}
