import dgl
import pandas as pd
import torch
from pyecharts import options as opts
from pyecharts.charts import Sankey


def sankey_graph(attn_weight, graphs, entity_id_path='entity2id.txt', sankey_saved_path="sankey_20240324_1.html"):
    node_attn = attn_weight[0].mean(dim=0).mean(dim=0).tolist()

    g = dgl.remove_self_loop(graphs)
    triple_len = len(torch.unique(g.all_edges()[0]))

    entity2id = pd.read_csv(entity_id_path, sep='\t', header=None)
    entity2id = {entity2id.iloc[index, 1]: entity2id.iloc[index, 0] for index in entity2id.index}

    links = []
    for src_index, tgt_index in zip(*g.all_edges()):
        src_name = entity2id[g.ndata['_ID'][src_index.item()].item()]
        tgt_name = entity2id[g.ndata['_ID'][tgt_index.item()].item()]

        links.append({'source': src_name, 'target': tgt_name,
                      'value': (node_attn[src_index.item()] + node_attn[tgt_index.item()]) / 2.0})

    accessed = set()
    nodes = []
    for link in links:
        for k in link:
            if k == 'value': continue
            node = link[k]
            if node in accessed: continue
            nodes.append({'name': node})
            accessed.add(node)

    if triple_len == 3:
        levels = [opts.SankeyLevelsOpts(depth=0,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#fbb4ae"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=1,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#b3cde3"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=2,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#ccebc5"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=3,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#decbe4"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5))
                  ]
    elif triple_len == 4:
        levels = [opts.SankeyLevelsOpts(depth=0,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#fbb4ae"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=1,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#b3cde3"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=2,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#ccebc5"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=3,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#decbe4"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=4,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#fed9a6"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  ]
    elif triple_len == 5:
        levels = [opts.SankeyLevelsOpts(depth=0,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#fbb4ae"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=1,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#b3cde3"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=2,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#ccebc5"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=3,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#decbe4"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=4,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#fed9a6"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=5,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#e5f5d0"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  ]
    else:
        levels = [opts.SankeyLevelsOpts(depth=0,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#fbb4ae"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=1,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#b3cde3"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=2,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#ccebc5"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=3,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#decbe4"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=4,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#fed9a6"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=5,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#e5f5d0"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  opts.SankeyLevelsOpts(depth=6,
                                        itemstyle_opts=opts.ItemStyleOpts(color="#b3a2c7"),
                                        linestyle_opts=opts.LineStyleOpts(color="source", opacity=0.7, curve=0.5)),
                  ]

    sankey = (Sankey().add(series_name='risk contagion',
                           nodes=nodes,
                           links=links,
                           linestyle_opt=opts.LineStyleOpts(opacity=0.9,
                                                            curve=0.5,
                                                            color="source"
                                                            ),
                           label_opts=opts.LabelOpts(font_size=16,
                                                     position='left'
                                                     ),
                           levels=levels
                           )
              )
    sankey.render(sankey_saved_path)
