SupplyChain-KG共有1200条数据。因果推理问题回答是yes标签为1，no标签为0。标签为1的有450条，标签0共750条。

数据：
‘text’中存放文本数据，组成结构：用YES或者NO回答问题 \n 问题背景 \n 因果推理问题 \n 回答（Yes or No）\n 解释<eos>。
'graph'是问题相关的知识图谱的子图，节点有_ID、nodetype（表明节点是消费者0or实体商店1or...）、embedding（节点的特征）、chain_or_not（节点是否在传染链路径上）这些属性，边有edge_type（表明边是哪一个relation）特征。可在dict文件夹中进行查阅，"edge_type2id.json"是边字典{“edge”:1,.....}共40个（1-40），"entity2id.json"是实体字典。字典格式{“entity”：0，....}共2400个（0-2399）。
 'chain'是传染链。
 'label'是问题的回答yes or no。
 'nce_label'是文本内容和节点的相似度。

KG文件夹存放整个供应链知识图谱

数据内容样例：
{'text': 'Please answer the following causal reasoning questions using Yes or No :\nMystic Moon Jewelry go out of business .BioBloom Botanicals have fewer orders.\nDoes the potential closure of Mystic Moon Jewelry lead to a decrease in orders for BioBloom Botanicals?\nYes\nThe causal chain starts with Mystic Moon Jewelry offering a product, the Smart UV Toothbrush Sanitizer, which is then sold online by XenonXanadu. XenonXanadu, in turn, purchases or orders finished products from BioBloom Botanicals. If Mystic Moon Jewelry were to go out of business, this would disrupt the supply chain starting from the very beginning. Without Mystic Moon Jewelry offering the Smart UV Toothbrush Sanitizer, XenonXanadu would not have this product to sell, leading to a decrease in orders from XenonXanadu to BioBloom Botanicals. Therefore, the risk to Mystic Moon Jewelry directly causes a risk to BioBloom Botanicals by potentially decreasing their orders.<eos>',
 'graph': Graph(num_nodes=13, num_edges=12,
       ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64), 'nodetype': Scheme(shape=(), dtype=torch.int32), 'embedding': Scheme(shape=(3072,), dtype=torch.float16), 'chain_or_not': Scheme(shape=(), dtype=torch.int32)}
       edata_schemes={'edge_type': Scheme(shape=(), dtype=torch.int32), '_ID': Scheme(shape=(), dtype=torch.int64)}),
 'chain': [1359, 1130, 1952, 1540],
 'label': 1,
 'nce_label': tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          ...,
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.],
          [0., 0., 0.,  ..., 0., 0., 0.]]], device='cuda:0')}
