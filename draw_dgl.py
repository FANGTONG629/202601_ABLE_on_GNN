import matplotlib.pyplot as plt
import networkx as nx
import dgl
import torch


def draw(g, img_name, save_dir, dataset, src_nid,tgt_nid):
    G = nx.Graph()

    node_types = g.ntypes
    for ntype in node_types:
        G.add_nodes_from(list(range(g.num_nodes(ntype))), node_type=ntype)

    # 添加边：从异构图的每种关系类型中提取边
    edge_types = g.canonical_etypes
    for etype in edge_types:
        src, dst = g.edges(form='uv', etype=etype)
        for u, v in zip(src.tolist(), dst.tolist()):
            G.add_edge(u, v, edge_type=f'{etype[1]}_{etype[2]}')

    # 绘制图形
    pos = nx.spring_layout(G)  # 使用spring布局
    plt.figure(figsize=(8, 8))

    # 绘制节点，按类型区分颜色
    if dataset == 1:
        color_map = {'attr': 'skyblue', 'user': 'lightgreen', 'item': 'lightcoral'}
    if dataset == 2:
        color_map = {'attr': 'skyblue', 'user': 'lightgreen', 'item': 'lightcoral'}
    if dataset == 3:
        color_map = {'attr': 'skyblue', 'user': 'lightgreen', 'item': 'lightcoral'}
    for ntype in node_types:
        node_indices = [n for n, attr in G.nodes(data=True) if attr['node_type'] == ntype]
        nx.draw_networkx_nodes(G, pos, nodelist=node_indices, node_color=color_map[ntype], label=ntype, node_size=500)

    # 绘制边
    nx.draw_networkx_edges(G, pos)

    # 添加标签
    nx.draw_networkx_labels(G, pos)

    # 显示图例
    plt.legend(scatterpoints=1)
    plt.title("Heterogeneous Graph Visualization")
    plt.savefig(save_dir+"/"+img_name+".png")  # 保存为图片文件
    #plt.show()

def draw_path(edges, img_name, save_dir, dataset, src_nid,tgt_nid):
    data_dict = {}

    for edge in edges:
        relation, src, dst = edge
        if relation not in data_dict:
            data_dict[relation] = ([], [])
        data_dict[relation][0].append(src)
        data_dict[relation][1].append(dst)

    # 转换为Tensor
    for key in data_dict:
        data_dict[key] = (torch.tensor(data_dict[key][0]), torch.tensor(data_dict[key][1]))

    # 创建DGL异构图
    g = dgl.heterograph(data_dict)

    # 转换为NetworkX图以便可视化
    G = g.to_networkx()

    # 绘制异构图
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)  # 使用spring布局

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="skyblue", alpha=0.9)

    # 绘制边
    nx.draw_networkx_edges(G, pos)

    # 添加标签
    nx.draw_networkx_labels(G, pos)
    plt.title("Heterogeneous Graph Visualization")