import matplotlib.pyplot as plt
import networkx as nx
import dgl
import torch
import os
import numpy as np
from datetime import datetime
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
    g,
    img_name,
    save_dir,
    dataset_name,
    feat_nids,
    eweight_dict=None
): # 可以用来画没有掩码的邻居子图
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
    if dataset_name == 'lastfm':
        color_map = {
            'artist': '#9EBBD7',  # 雾霾天蓝 (Dusty Baby Blue)
            'user': '#A7CFAB'    # 灰调薄荷绿 (Grayish Mint Green)
        }
    elif dataset_name == 'aug_citation':
        color_map = {
            'paper': '#FBB4AE',   # 珊瑚粉 (核心)
            'fos': '#CCEBC5',     # 薄荷绿 (领域)
            'author': '#B3CDE3',  # 冰晶蓝 (人物)
            'ref': '#E5E5E5'      # 浅珍珠灰 (辅助/参考文献)
        }
    elif dataset_name == 'ACM':
        color_map = {
            'author': '#B0C4DE',  # 雾霾淡蓝 (Light Dusty Blue)
            'field': '#A9DFBF',   # 雾霾淡绿 (Light Dusty Green)
            'paper': '#E6B0C1'    # 雾霾粉红 (Light Dusty Pink)
        }
    for ntype in node_types:
        node_indices = [n for n, attr in G.nodes(data=True) if attr['node_type'] == ntype]
        nx.draw_networkx_nodes(G, pos, nodelist=node_indices, node_color=color_map[ntype], label=ntype, node_size=70, alpha=0.7)
    # 绘制边
    nx.draw_networkx_edges(G, pos, width=0.5, edge_color='gray', alpha=0.5)

    # 添加标签
    nx.draw_networkx_labels(G, pos, font_size=5, alpha=0.4)

    # 显示图例
    plt.legend(scatterpoints=1)
    plt.title("Heterogeneous Graph Visualization")
    plt.savefig(save_dir + "/" + img_name + ".png")  # 保存为图片文件
    plt.show()


def get_mask_delta(g, mask1=None, mask2=None):
    """
    计算两个掩码之间的差值 (mask3 = mask2 - mask1)。

    参数:
    ----
    g : DGLGraph 当前的异构图对象，用于在掩码为 None 时获取各关系的边数和设备信息。
    mask1 : dict[etype, Tensor] or None 第一个掩码字典。若为 None，则每条边的掩码默认为 1。
    mask2 : dict[etype, Tensor] or None 第二个掩码字典。若为 None，则每条边的掩码默认为 1。

    返回:
    ----
    mask3 : dict[etype, Tensor]
        作差后的掩码字典，保留所有结果（包括负数和 0）。
    """
    mask3 = {}
    device = g.device

    # 遍历图中所有的规范关系类型 (canonical_etypes)
    for etype in g.canonical_etypes:
        num_edges = g.num_edges(etype)
        # --- 处理 mask1 ---
        if mask1 is None or etype not in mask1:
            m1 = torch.ones(num_edges, device=device) # 如果传入为 None，每条边掩码默认为 1
        else: m1 = mask1[etype]
        # --- 处理 mask2 ---
        if mask2 is None or etype not in mask2:
            m2 = torch.ones(num_edges, device=device)
        else: m2 = mask2[etype]

        # --- 执行作差计算：mask3 = mask2 - mask1 ---
        mask3[etype] = m2 - m1 # 直接相减，保留负数、0 以及正数

    return mask3


def draw_able_graph_eweight(exres, dataset_name, img_name=None, save_dir="./outputs/GRAPH_EW"):
    """
        按照掩码 eweight_dict 调整边的透明度和粗细，展示解释子图。
        参考 visualize_neighborhood_tsne 结构提取数据。
        """
    # 1. 路径与命名准备
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. 提取数据结构 (参考 exres 结构)
    # 默认绘制第一对对抗样本中的 G_M (反事实解释图)
    adv_pairs = exres.get("adv_pairs", [])
    if not adv_pairs:
        print("No adversarial pairs found in exres.")
        return

    node_types = adv_pairs[0]['G_M']['graph'].ntypes
    if dataset_name == 'lastfm':
        color_map = {
            'user': '#9EBBD7',  # 雾霾天蓝 (Dusty Baby Blue)
            'artist': '#A7CFAB'  # 灰调薄荷绿 (Grayish Mint Green)
        }
    elif dataset_name == 'aug_citation':
        color_map = {
            'paper': '#FBB4AE',  # 珊瑚粉 (核心)
            'fos': '#CCEBC5',  # 薄荷绿 (领域)
            'author': '#B3CDE3',  # 冰晶蓝 (人物)
            'ref': '#E5E5E5'  # 浅珍珠灰 (辅助/参考文献)
        }
    elif dataset_name == 'ACM':
        color_map = {
            'author': '#B0C4DE',  # 雾霾淡蓝 (Light Dusty Blue)
            'field': '#A9DFBF',  # 雾霾淡绿 (Light Dusty Green)
            'paper': '#E6B0C1'  # 雾霾粉红 (Light Dusty Pink)
        }
    else:
        color_map = {ntype: plt.cm.Pastel1(i) for i, ntype in enumerate(node_types)}

    # 开始遍历所有样本对
    for pair in adv_pairs:
        idx = pair['idx']
        g = pair['G_M']['graph']

        # 定义两个绘图子任务：(任务名, mask1, mask2, 剪枝阈值)
        tasks = [
            (f"none_gm_{idx}", None, pair['G_M']['edge_mask'], 0.8),
            (f"gm_gw_{idx}", pair['G_M']['edge_mask'], pair['G_W']['edge_mask'], 0.4)
        ]

        for task_name, m1, m2, prune_threshold in tasks:
            # 2. 计算差值掩码
            mask_delta = get_mask_delta(g, mask1=m1, mask2=m2)

            # 3. 构建 NetworkX 图
            G = nx.Graph()
            for ntype in node_types:
                for i in range(g.num_nodes(ntype)):
                    G.add_node(f"{ntype}_{i}", node_type=ntype)

            # 4. 构建边并映射权重
            bins = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            edge_buckets = {b: [] for b in bins}
            cpu_masks = {etype: m.detach().cpu().numpy() for etype, m in mask_delta.items()}

            for etype in g.canonical_etypes:
                u_nt, _, v_nt = etype
                src, dst = g.edges(form='uv', etype=etype)
                mask = cpu_masks.get(etype)
                for i, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
                    m_val = float(mask[i])
                    if abs(m_val) < prune_threshold: continue  # 动态剪枝
                    u_n, v_n = f"{u_nt}_{u}", f"{v_nt}_{v}"
                    G.add_edge(u_n, v_n)
                    for b in bins:
                        if m_val <= b:
                            edge_buckets[b].append((u_n, v_n))
                            break

            # 5. 绘图
            pos = nx.spring_layout(G, k=0.3, iterations=30, seed=42)
            plt.figure(figsize=(10, 10))

            # 6. 绘制节点
            for node, attr in G.nodes(data=True):
                ntype = attr['node_type']
                G.nodes[node]['fc'] = 'white' if G.degree(node) == 0 else color_map.get(ntype, '#D3D3D3')
                G.nodes[node]['ec'] = 'gray' if G.degree(node) == 0 else 'white'
                G.nodes[node]['alp'] = 0.1 if G.degree(node) == 0 else 0.8

            for ntype in node_types:
                nodelist = [n for n, attr in G.nodes(data=True) if attr['node_type'] == ntype]
                if not nodelist: continue
                nx.draw_networkx_nodes(G, pos, nodelist=nodelist,
                                       node_color=[G.nodes[n]['fc'] for n in nodelist],
                                       edgecolors=[G.nodes[n]['ec'] for n in nodelist],
                                       node_size=150, linewidths=0.5, alpha=[G.nodes[n]['alp'] for n in nodelist])

            # 7. 绘制边
            for b in bins:
                if edge_buckets[b]:
                    c = 'red' if b < 0 else 'gray'
                    nx.draw_networkx_edges(G, pos, edgelist=edge_buckets[b],
                                           width=0.2 + 0.4 * abs(b), alpha=max(0.2, abs(b)), edge_color=c)

            # 8. 标签与图例
            #nx.draw_networkx_labels(G, pos, font_size=6, font_color="black", alpha=0.7)
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=nt,
                                      markerfacecolor=color_map.get(nt, '#D3D3D3'),
                                      markersize=12, markeredgecolor='white', markeredgewidth=0.5)
                               for nt in node_types if nt in color_map]
            plt.legend(handles=legend_elements, loc='upper right', frameon=True)
            plt.title(f"Task: {task_name} | Dataset: {dataset_name}\nRed: Weakened, Gray: Strengthened")
            plt.axis('off')

            # 9. 保存
            plt.savefig(os.path.join(save_dir, f"{task_name}_{dataset_name}.png"), bbox_inches='tight', dpi=300)
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

    # ===== 边和节点类型颜色映射 =====
    edge_color_map = {
        ('user', 'likes', 'artist'): 'orange', # 预测边类型
        ('author', 'likes', 'paper'): 'orange',
        ('paper', 'pf', 'field'): 'orange',
        ('user', 'friends', 'user'): 'cyan',  # 用户-用户关系
        ('user', 'of', 'artist'): 'magenta',  # 艺术家-艺术家关系
    }
    node_color_map = {
        "user": "lightgreen",
        "artist": "lightcoral",
        "attr": "skyblue",
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


    # ===== 节点颜色 =====
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

