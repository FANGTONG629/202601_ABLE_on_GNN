import matplotlib.pyplot as plt
import networkx as nx
import dgl
import torch
import os
import numpy as np

from collections import defaultdict


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


def draw_able_graph(
    G_dict,
    img_name,
    save_dir,
    dataset_name,
    src_nid,
    tgt_nid,
    use_edge_mask=True,
    alpha_min=0.05,
    alpha_max=1.0,
):
    """
    Visualize ABLE-g G_M / G_W for lastfm-like heterogeneous graphs.

    Parameters
    ----------
    G_dict : dict
        {
            "graph": dgl.DGLHeteroGraph,
            "edge_mask": dict[canonical_etype -> Tensor[num_edges]]  (optional)
        }
    use_edge_mask : bool
        Whether to visualize edge mask by transparency
    """

    g = G_dict["graph"]
    edge_mask_dict = G_dict.get("edge_mask", None)

    # -------- lastfm schema --------
    if "lastfm" in dataset_name:
        src_ntype = "user"
        tgt_ntype = "artist"
        pred_etype = ("user", "likes", "artist")
    else:
        raise ValueError("draw_able_graph currently supports lastfm only")

    # -------- build networkx graph --------
    G = nx.Graph()

    # node id: (ntype, nid) to avoid collision
    for ntype in g.ntypes:
        for nid in range(g.num_nodes(ntype)):
            G.add_node(
                (ntype, nid),
                ntype=ntype
            )

    # edges
    edge_attr = {}  # ((u),(v)) -> alpha
    for etype in g.canonical_etypes:
        src, dst = g.edges(etype=etype)
        mask = None

        if use_edge_mask and edge_mask_dict is not None:
            mask = edge_mask_dict[etype].detach().cpu()

        for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
            u_node = (etype[0], u)
            v_node = (etype[2], v)

            G.add_edge(
                u_node,
                v_node,
                etype=etype
            )

            if mask is not None:
                alpha = mask[i].item()
                alpha = max(alpha_min, min(alpha, alpha_max))
            else:
                alpha = 1.0

            edge_attr[(u_node, v_node)] = alpha

    # -------- layout --------
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(9, 9))

    # -------- draw nodes --------
    node_colors = {
        "user": "lightgreen",
        "artist": "lightcoral",
        "attr": "skyblue",
    }

    for ntype in g.ntypes:
        nodelist = [n for n in G.nodes if n[0] == ntype]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_color=node_colors.get(ntype, "gray"),
            node_size=500,
            alpha=0.9,
            label=ntype
        )

    # highlight src / tgt
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[(src_ntype, int(src_nid))],
        node_color="gold",
        node_size=700,
        label="src"
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[(tgt_ntype, int(tgt_nid))],
        node_color="orange",
        node_size=700,
        label="tgt"
    )

    # -------- draw edges --------
    for (u, v), alpha in edge_attr.items():
        etype = G.edges[u, v]["etype"]
        if etype == pred_etype:
            color = "red"
            width = 2.5
        else:
            color = "gray"
            width = 1.0

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            alpha=alpha,
            width=width,
            edge_color=color
        )

    # -------- labels --------
    labels = {n: f"{n[0]}:{n[1]}" for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.legend()
    plt.title("ABLE-g Edge-Masked Graph Visualization")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{img_name}.png")
    plt.close()


def draw_able_graph_on_ax(
        ax,
        ghetero,
        edge_mask=None,
        dataset_name="lastfm",
        src_nid=None,
        tgt_nid=None,
        title=None,
):
    import networkx as nx
    from collections import defaultdict

    G = nx.MultiGraph()

    # ===== 节点 =====
    for ntype in ghetero.ntypes:
        for nid in range(ghetero.num_nodes(ntype)):
            G.add_node((ntype, nid), node_type=ntype)

    # ===== 边类型颜色映射 =====
    edge_color_map = {
        ('user', 'likes', 'artist'): 'orange',  # 预测边类型
        ('user', 'friends', 'user'): 'cyan',  # 用户-用户关系
        ('user', 'of', 'artist'): 'magenta',  # 艺术家-艺术家关系
    }

    # 为没有预定义颜色的边类型生成随机颜色
    used_colors = set(edge_color_map.values())
    available_colors = ['cyan', 'lime', 'brown', 'pink', 'olive', 'navy', 'teal']

    # ===== 边：按类型和透明度分组 =====
    edges_by_type_alpha = defaultdict(list)  # (etype, alpha) -> [(u,v)]
    center_edges = []  # 中心边（src-tgt）
    visible_nodes = set()  # 最终真正被画出边的节点

    for etype in ghetero.canonical_etypes:
        src, dst = ghetero.edges(form="uv", etype=etype)

        # 为当前边类型分配颜色
        if etype not in edge_color_map:
            # 从可用颜色中选择一个
            for color in available_colors:
                if color not in used_colors:
                    edge_color_map[etype] = color
                    used_colors.add(color)
                    break
            else:
                # 如果都用完了，使用默认颜色
                edge_color_map[etype] = 'gray'

        for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
            u_node = (etype[0], u)
            v_node = (etype[2], v)

            # 判断是否中心边（预测边）
            is_center = (
                    src_nid is not None
                    and tgt_nid is not None
                    and etype[0] == "user"
                    and etype[2] == "artist"
                    and u == src_nid
                    and v == tgt_nid
            )

            if is_center:
                center_edges.append((u_node, v_node, etype))
                continue

            # 计算透明度
            if edge_mask is not None and etype in edge_mask:
                alpha_val = float(edge_mask[etype][i].clamp(0, 1).item())
            else:
                alpha_val = 1.0

            # 分组（按边类型和透明度值）
            alpha_key = round(alpha_val, 3)
            edges_by_type_alpha[(etype, alpha_key)].append((u_node, v_node))

    # ===== layout =====
    pos = nx.spring_layout(G, seed=42)

    # ===== 绘制普通边（按类型和透明度）=====
    for (etype, alpha), edgelist in edges_by_type_alpha.items():
        if alpha <= 0.5:  # 过滤掉几乎透明的边
            continue

        for u, v in edgelist:  # 记录真正可见边的端点
            visible_nodes.add(u)
            visible_nodes.add(v)

        alpha_norm = (alpha - 0.5) / (1.0 - 0.5) # 重新映射 alpha 到 0~1
        alpha_norm = np.clip(alpha_norm, 0.0, 1.0)  # 防止数值溢出

        color = edge_color_map.get(etype, 'gray')

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edgelist,
            alpha=alpha_norm,
            width=0.7,
            edge_color=color,
            ax=ax,
        )

    # ===== 绘制中心边 =====
    if center_edges:
        for u_node, v_node, etype in center_edges:
            visible_nodes.add(u_node)
            visible_nodes.add(v_node)
            color = edge_color_map.get(etype, 'red')
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u_node, v_node)],
                edge_color=color,
                width=3.0,
                alpha=1.0,
                ax=ax,
            )

    # ===== 节点颜色 =====
    node_color_map = {
        "user": "lightgreen",
        "artist": "lightcoral",
        "attr": "skyblue",
    }

    for ntype in ghetero.ntypes:
        nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == ntype]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            node_color=node_color_map.get(ntype, "gray"),
            node_size=50,
            ax=ax,
            alpha=0.3
        )
        # ===== 给“真正参与结构的节点”画黑色边框 =====
        if visible_nodes:
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=list(visible_nodes),
                node_color=node_color_map.get(ntype, "gray"),  # 不覆盖原填充色
                node_size=50,
                edgecolors='black',  # 黑色边框
                linewidths=1.2,
                alpha=0.5,
                ax=ax,
            )



    # ===== 添加图例 =====
    legend_elements = []
    for etype, color in edge_color_map.items():
        if any(edges_by_type_alpha.get((etype, alpha), []) for alpha in [0.2, 0.5, 0.8, 1.0]):
            # 只显示实际存在的边类型
            legend_label = f"{etype[0]}→{etype[2]}"
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=legend_label))

    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # ===== 标签和标题 =====
    if title:
        ax.set_title(title, fontsize=10)

    ax.axis("off")

# def draw_able_graph_on_ax(
#     ax,
#     ghetero,
#     edge_mask=None,
#     dataset_name="lastfm",
#     src_nid=None,
#     tgt_nid=None,
#     title=None,
# ):
#     import networkx as nx
#     import numpy as np
#
#     G = nx.Graph()
#
#     # ===== 节点 =====
#     for ntype in ghetero.ntypes:
#         for nid in range(ghetero.num_nodes(ntype)):
#             G.add_node((ntype, nid), node_type=ntype)
#
#     # ===== 边（先收集，不画）=====
#     edges_by_alpha = {}  # alpha -> [(u,v), ...]
#
#     for etype in ghetero.canonical_etypes:
#         src, dst = ghetero.edges(form="uv", etype=etype)
#
#         for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
#             alpha = 1.0
#             if edge_mask is not None:
#                 alpha = float(edge_mask[etype][i].clamp(0, 1).item())
#
#             alpha = round(alpha, 2)  # 分桶，避免过多组
#             edges_by_alpha.setdefault(alpha, []).append(
#                 ((etype[0], u), (etype[2], v))
#             )
#
#     # ===== layout（只算一次）=====
#     pos = nx.spring_layout(G, seed=42)
#
#     # ===== 节点颜色 =====
#     color_map = {
#         "user": "lightgreen",
#         "artist": "lightcoral",
#         "attr": "skyblue",
#     }
#
#     for ntype in ghetero.ntypes:
#         nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == ntype]
#         nx.draw_networkx_nodes(
#             G,
#             pos,
#             nodelist=nodes,
#             node_color=color_map.get(ntype, "gray"),
#             node_size=20,
#             ax=ax,
#         )
#
#     # ===== 一次画一组边（关键）=====
#     for alpha, edgelist in edges_by_alpha.items():
#         nx.draw_networkx_edges(
#             G,
#             pos,
#             edgelist=edgelist,
#             alpha=alpha,
#             width=0.5,
#             ax=ax,
#         )
#
#     if title:
#         ax.set_title(title, fontsize=10)
#
#     ax.axis("off")
