import json

import dgl
import torch
from pyecharts import options as opts
from pyecharts.charts import Sankey

plot_sankey_graph(risk_score_label, graphs)
def sankey_graph(risk_score, graphs, entity_id_path="entity2id.json", sankey_saved_path="sankey_20240324_1.html"):
    risk_score = risk_score.cpu().tolist()

    g = dgl.remove_self_loop(graphs)
    triple_len = len(torch.unique(g.all_edges()[0]))

    with open(entity_id_path, 'r') as f:
        entity2id_dict = json.load(f)
    id2entity_dict = {value: key for key, value in entity2id_dict.items()}

    links = []
    for src_index, tgt_index in zip(*g.all_edges()):
        src_name = id2entity_dict[g.ndata['_ID'][src_index.item()].item()]
        tgt_name = id2entity_dict[g.ndata['_ID'][tgt_index.item()].item()]

        links.append({'source': src_name, 'target': tgt_name,
                      'value': (risk_score[src_index]+risk_score[tgt_index])/2.0})

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
