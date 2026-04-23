import os
import torch
import dgl
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from utils import hetero_src_tgt_khop_in_subgraph


def visualize_subgraph_hops(g_full, src_type, src_id, dst_type, dst_id):
    """
        提取并可视化 1-hop 和 2-hop 子图
        """
    # 1. 提取 1-hop 子图
    _, _, sg1, _ = hetero_src_tgt_khop_in_subgraph(src_type, src_id, dst_type, dst_id, g_full, k=1)

    # 2. 提取 2-hop 子图
    _, _, sg2, _ = hetero_src_tgt_khop_in_subgraph(src_type, src_id, dst_type, dst_id, g_full, k=2)

    # 3. 将 DGL 子图转换为 NetworkX (使用 2-hop 作为基础图)
    # 为了方便染色，我们将节点 ID 转换为 "type_id" 格式
    G = nx.Graph()

    # 获取 1-hop 节点集合
    nodes_hop1 = set()
    for ntype in sg1.ntypes:
        orig_ids = sg1.ndata[dgl.NID][ntype]
        for idx in orig_ids.tolist():
            nodes_hop1.add(f"{ntype}_{idx}")

    # 构建 2-hop 的 NetworkX 图并确定颜色
    node_colors = []
    node_list = []

    # 遍历 sg2 中的边来构建 G
    for etype in sg2.canonical_etypes:
        u_ntype, _, v_ntype = etype
        u_ids, v_ids = sg2.edges(etype=etype)
        # 获取原始 ID
        u_orig = sg2.ndata[dgl.NID][u_ntype][u_ids]
        v_orig = sg2.ndata[dgl.NID][v_ntype][v_ids]

        for u, v in zip(u_orig.tolist(), v_orig.tolist()):
            u_name, v_name = f"{u_ntype}_{u}", f"{v_ntype}_{v}"
            G.add_edge(u_name, v_name)

    # 4. 分配颜色
    # 规则：如果在 hop1 集合中 -> 蓝色；如果不在但在 hop2 中 -> 绿色
    for node in G.nodes():
        node_list.append(node)
        if node in nodes_hop1:
            node_colors.append('#3498db')  # 蓝色
        else:
            node_colors.append('#2ecc71')  # 绿色

    # 5. 绘图
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    # 绘制边
    nx.draw_networkx_edges(G, pos, alpha=0.7, edge_color='gray', width=3.0)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_color=node_colors,
                           node_size=600, edgecolors='white', linewidths=1)

    # 绘制标签 (可选)
    nx.draw_networkx_labels(G, pos, font_size=9, alpha=0.7)

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='1-hop Neighbors',
               markerfacecolor='#3498db', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='2-hop Neighbors',
               markerfacecolor='#2ecc71', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.title(f"Subgraph Hops for Edge ({src_type}_{src_id.item()} -> {dst_type}_{dst_id.item()})")
    plt.axis('off')
    plt.savefig("./outputs/DEMO/hop.png", dpi=300, bbox_inches='tight', transparent=True)
    plt.show()



# 使用示例 (逻辑同前，只需确保数据加载正确)


# # --- 测试部分 ---
# if __name__ == "__main__":
#     # 参考 ABLE_g_run.py 的加载逻辑
#     # 这里假设你已经加载了 ACM 图数据 g
#     # 示例加载 (根据你的环境修改路径):
#     # g, _ = dgl.load_graphs("data/ACM.bin")
#     # g = g[0]
#
#     # 模拟数据环境 (若无数据请替换为你真实的加载代码)
#     try:
#         data_path = "./datasets/ACM"  # 参考你的目录结构
#         graphs, _ = dgl.load_graphs(data_path)
#         g = graphs[0]
#
#         # 选取一条 'paper'-'author' 的边进行测试
#         src_type, dst_type = 'paper', 'field'
#         u, v = g.edges(etype=('paper', 'pf', 'field'))
#
#         test_src = u[0]
#         test_dst = v[0]
#
#         print(f"Extracting subgraphs for edge: {test_src} -> {test_dst}")
#         visualize_subgraph_hops(g, src_type, test_src, dst_type, test_dst)
#
#     except Exception as e:
#         print(f"Error: {e}. 请确保 ACM.bin 路径正确且 ABLE_g.py 在同一目录下。")

def run_demo_visualizations(dataset_path="./datasets/ACM", save_dir="./outputs/DEMO"):

    # 1. 加载数据
    graphs, _ = dgl.load_graphs(dataset_path)
    g_full = graphs[0]

    # 随机选一条边 (以 paper-author 为例)
    u, v = g_full.edges(etype=('paper', 'pf', 'field'))
    idx = 0 # np.random.randint(0, len(u))
    src_id, dst_id = u[idx], v[idx]
    src_type, dst_type = 'paper', 'field'

    # 2. 提取 2-hop 子图 (参考 DocTools)
    _, _, sg, _ = hetero_src_tgt_khop_in_subgraph(src_type, src_id, dst_type, dst_id, g_full, k=2)

    # 3. 构建 NetworkX 图
    G = nx.Graph()
    target_edge = (f"{src_type}_{src_id.item()}", f"{dst_type}_{dst_id.item()}")

    edge_masks = {}  # 存储随机生成的掩码
    for etype in sg.canonical_etypes:
        u_nt, rel, v_nt = etype
        u_ids, v_ids = sg.edges(etype=etype)
        u_orig = sg.ndata[dgl.NID][u_nt][u_ids]
        v_orig = sg.ndata[dgl.NID][v_nt][v_ids]

        for ui, vi in zip(u_orig.tolist(), v_orig.tolist()):
            u_name, v_name = f"{u_nt}_{ui}", f"{v_nt}_{vi}"
            G.add_edge(u_name, v_name)
            # 随机生成 0~1 掩码，两端分布更多 (使用 Beta 分布模拟)
            if (u_name, v_name) not in edge_masks:
                mask_val = np.random.beta(0.5, 0.5)
                edge_masks[(u_name, v_name)] = mask_val

    # 4. 颜色配置 (ACM 甜美色系)
    color_map = {'author': '#B0C4DE',  # 雾霾淡蓝 (Light Dusty Blue)
            'field': '#A9DFBF',  # 雾霾淡绿 (Light Dusty Green)
            'paper': '#E6B0C1'}
    pos = nx.spring_layout(G, k=0.4, seed=42)

    # --- 绘图函数 ---
    def draw_stage(stage_name, show_mask=False, prune=False):
        plt.figure(figsize=(10, 8))

        #nx.draw_networkx_labels(G, pos, font_size=9, alpha=0.7)

        # 绘制节点
        for ntype in color_map.keys():
            nodelist = [n for n in G.nodes() if n.startswith(ntype)]
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=color_map[ntype],
                                   node_size=700, edgecolors='white', linewidths=1)

        # 绘制边
        for edge in G.edges():
            u, v = edge
            m_val = edge_masks.get(edge, edge_masks.get((v, u), 1.0))

            # 判断是否为目标边
            is_target = (edge == target_edge or edge == (target_edge[1], target_edge[0]))

            # 透明剪枝逻辑
            alpha = 0.8
            if prune and m_val < 0.2: alpha = 0.0  # 阶段3：变透明

            # 颜色逻辑
            if is_target:
                color, style, width = 'red', '--', 3.5  # 目标边红色虚线
            elif m_val < 0.2 and not prune:
                if not show_mask: color, style, width = 'gray', '-', 2.5
                else: color, style, width = '#90EE90', '-', 2.5  # 阶段2：小于0.2浅绿色
            else:
                color, style, width = 'gray', '-', 2.5

            nx.draw_networkx_edges(G, pos, edgelist=[edge], edge_color=color,
                                   style=style, width=width, alpha=alpha)

            # 标注掩码值
            if show_mask and alpha > 0:
                nx.draw_networkx_edge_labels(G, pos, edge_labels={edge: f"{m_val:.2f}"}, font_size=12)

        #plt.title(f"ACM 2-Hop Demo: {stage_name}")
        plt.axis('off')
        save_path = os.path.join(save_dir, f"{stage_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
        print(f"Saved: {save_path}")
        plt.close()

    # 执行三个阶段
    draw_stage("1_Base_Target_Red_Dashed", show_mask=False, prune=False)
    draw_stage("2_Mask_Labeled_Green_Below_0.2", show_mask=True, prune=False)
    draw_stage("3_Pruned_Transparent_Below_0.2", show_mask=True, prune=True)


if __name__ == "__main__":
    # 路径请根据实际情况修改
    run_demo_visualizations(dataset_path="datasets/ACM")