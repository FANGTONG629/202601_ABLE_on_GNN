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


# if __name__ == "__main__":
#     # 路径请根据实际情况修改
#     run_demo_visualizations(dataset_path="datasets/ACM")

import dgl
import torch
import os


def print_node_attributes(dataset_path="./datasets/ACM", attr_name='name'):
    """
    读取数据集并打印每种节点类型的指定属性信息。

    参数:
    ----------
    dataset_path : str
        数据集文件路径。
    attr_name : str
        要查看的属性名称（如 'name', 'label', 'feat'）。
    """
    # 1. 加载数据
    if not os.path.exists(dataset_path):
        print(f"错误: 找不到路径 {dataset_path}")
        return

    graphs, _ = dgl.load_graphs(dataset_path)
    g = graphs[0]

    print(f"--- 数据集信息: {dataset_path} ---")
    print(f"节点类型: {g.ntypes}")
    print("-" * 50)

    # 2. 遍历每种节点类型
    for ntype in g.ntypes:
        num_nodes = g.num_nodes(ntype)
        print(f"节点类型: [{ntype}] | 总数: {num_nodes}")

        # 检查该节点类型是否有任何属性
        available_attrs = g.nodes[ntype].data.keys()
        if not available_attrs:
            print(f"  -> 该节点类型没有任何属性数据。")
            print("-" * 30)
            continue

        print(f"  可用属性: {list(available_attrs)}")

        # 3. 尝试读取并打印目标属性
        if attr_name in g.nodes[ntype].data:
            data = g.nodes[ntype].data[attr_name]

            # 如果是文本名字（通常是 list 或 tensor）
            if isinstance(data, (list, torch.Tensor)):
                # 如果是 Tensor，尝试打印前 5 个作为示例
                # 如果你想看全部 71 个 field，可以将切片去掉
                sample_size = num_nodes
                samples = data[:sample_size]

                print(f"  属性 '{attr_name}' 内容预览:")
                if torch.is_tensor(samples):
                    # 如果存储的是索引或独热编码，打印其值
                    print(samples.tolist())
                else:
                    print(samples)
        else:
            print(f"  -> 属性 '{attr_name}' 不存在于此节点类型中。")

        print("-" * 30)


# if __name__ == "__main__":
#     # 执行脚本
#     # 提示：在 ACM 数据集中，field 的名字可能存储在 'name' 或 'label' 中
#     print_node_attributes(dataset_path="./datasets/ACM", attr_name='name')

import pandas as pd

# def plot_sensitivity(data, fixed_param, vary_param, dataset_name, ax):
#     """
#     绘制灵敏度曲线图
#     fixed_param: 固定的参数名 (e.g. 'Lambda_1')
#     vary_param: 变化的参数名 (e.g. 'Lambda_2')
#     """
#     # 筛选固定参数为 0.01 的数据
#     sub_df = data[data[fixed_param] == 0.01].copy()
#     # 获取变化参数的所有取值并排序
#     vary_values = sorted(sub_df[vary_param].unique())
#
#     for i, val in enumerate(vary_values):
#         line_data = sub_df[sub_df[vary_param] == val].sort_values('Radius')
#
#         # 绘制带误差棒的折线
#         ax.errorbar(
#             line_data['Radius'], line_data['mean_pct'], yerr=line_data['std_pct'],
#             label=f'{vary_param}={val}',
#             color=colors[i], marker='o', capsize=5, elinewidth=1.5, markeredgewidth=1.5
#         )
#
#     ax.set_title(f'{dataset_name} (Fixed {fixed_param}=0.01)', fontsize=12)
#     ax.set_xlabel('Radius', fontsize=10)
#     ax.set_ylabel('Conditional Recovery Rate (%)', fontsize=10)
#     ax.set_ylim(0, 105)  # 设置纵轴范围
#     ax.set_xticks(sub_df['Radius'].unique())
#     ax.grid(True, linestyle='--', alpha=0.6)
#     ax.legend(title='Variables')

# # 1. 加载数据
# file_path = './data/performance_l.xlsx'
# df = pd.read_excel(file_path)
#
# # 确保数值类型正确
# cols_to_fix = ['Radius', 'Lambda_1', 'Lambda_2', 'G_W|G_M:Mean', 'G_W|G_M:Var']
# df[cols_to_fix] = df[cols_to_fix].apply(pd.to_numeric)
#
# # 转换百分比
# df['mean_pct'] = df['G_W|G_M:Mean'] * 100
# df['std_pct'] = df['G_W|G_M:Var'] * 100
#
# datasets = df['Dataset'].unique()
# colors = ['#D62728', '#1f77b4', '#2ca02c']  # 红色，蓝色，绿色
#
# # 2. 遍历数据集画图
# for ds in datasets:
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
#     ds_data = df[df['Dataset'] == ds]
#
#     # 第一张图：固定 Lambda_1=0.01，观察不同 Lambda_2
#     plot_sensitivity(ds_data, 'Lambda_1', 'Lambda_2', ds, ax1)
#
#     # 第二张图：固定 Lambda_2=0.01，观察不同 Lambda_1
#     plot_sensitivity(ds_data, 'Lambda_2', 'Lambda_1', ds, ax2)
#
#     plt.tight_layout()
#     # 保存图片（可选）
#     # plt.savefig(f"{ds}_sensitivity_analysis.png", dpi=300)
#     plt.show()

def plot_sensitivity(data, fixed_param, vary_param, dataset_name, ax):
    """
    fixed_param: 'Lambda_1' 或 'Lambda_2'
    """
    # 筛选固定参数为 0.01 的数据
    sub_df = data[data[fixed_param] == 0.01].copy()
    vary_values = sorted(sub_df[vary_param].unique())

    # 根据固定参数决定要画的辅助指标
    # 固定 Lambda_1 时，画 G_W (辅助) 和 G_W|G_M (核心)
    # 固定 Lambda_2 时，画 G_M (辅助) 和 G_W|G_M (核心)
    is_l1_fixed = (fixed_param == 'Lambda_1')
    aux_metric_mean = 'G_W: Mean' if is_l1_fixed else 'G_M: Mean'
    aux_metric_var = 'G_W:Var' if is_l1_fixed else 'G_M: Var'
    aux_label = 'R_rec (G_W)' if is_l1_fixed else 'R_flip (G_M)'

    # 颜色设置：暖色调用于辅助指标 (实线)，冷色调用于 G_W|G_M (虚线)
    warm_colors = ['#f15e14', '#f49f00', '#eb596a']  # 橙红系
    cool_colors = ['#18840e', '#2653ae', '#1fd0af']  # 蓝系

    for i, val in enumerate(vary_values):
        line_data = sub_df[sub_df[vary_param] == val].sort_values('Radius')

        # 1. 绘制辅助指标数据 (暖色系, 实线, 80%透明度)
        ax.errorbar(
            line_data['Radius'], line_data[aux_metric_mean] * 100,
            yerr=line_data[aux_metric_var] * 100,
            label=f'{aux_label} ({vary_param}={val})',
            color=warm_colors[i % len(warm_colors)], linestyle='-', alpha=0.7,
            marker='o', capsize=3, elinewidth=1
        )

        # 2. 绘制核心指标 G_W|G_M (冷色系, 虚线, 80%透明度)
        ax.errorbar(
            line_data['Radius'], line_data['G_W|G_M:Mean'] * 100,
            yerr=line_data['G_W|G_M:Var'] * 100,
            label=f'R_cond (G_W|G_M) ({vary_param}={val})',
            color=cool_colors[i % len(cool_colors)], linestyle='--', alpha=0.7,
            marker='s', capsize=3, elinewidth=1
        )

    ax.set_title(f'{dataset_name} (Fixed {fixed_param}=0.01)', fontsize=12)
    ax.set_xlabel('Radius', fontsize=10)
    ax.set_ylabel('Success Rate (%)', fontsize=10)
    ax.set_ylim(50, 105)  # 调大范围以包含 G_M 有时较低的情况
    ax.set_xticks(sub_df['Radius'].unique())
    ax.grid(True, linestyle=':', alpha=0.5)
    # 将图例放在外面或缩小，防止遮挡线段
    ax.legend(prop={'size': 7}, loc='lower left', ncol=3)





def plot_sensitivity_2(data, fixed_param, vary_param, dataset_name, ax):
    """
    优化样式：采用阴影区域 (Shaded Area) 代替传统误差棒
    """
    # 筛选固定参数为 0.01 的数据
    sub_df = data[data[fixed_param] == 0.01].copy()
    vary_values = sorted(sub_df[vary_param].unique())

    # 逻辑判断：辅助指标选择
    is_l1_fixed = (fixed_param == 'Lambda_1')
    aux_metric_mean = 'G_W: Mean' if is_l1_fixed else 'G_M: Mean'
    aux_metric_var = 'G_W:Var' if is_l1_fixed else 'G_M: Var'
    aux_label = '$R_{rec}$ ($G_W$)' if is_l1_fixed else '$R_{flip}$ ($G_M$)'

    # 调色盘：冷暖色系增强对比
    # 辅助指标：暖色系 (实线)
    warm_palette = ['#f15e14', '#f49f00', '#eb596a']
    # 核心指标 G_W|G_M：冷色系 (虚线)
    cool_palette = ['#18840e', '#2653ae', '#1fd0af']

    for i, val in enumerate(vary_values):
        line_data = sub_df[sub_df[vary_param] == val].sort_values('Radius')
        x = line_data['Radius'].values

        # --- 1. 绘制辅助指标 (实线 + 阴影) ---
        mean_aux = line_data[aux_metric_mean].values * 100
        std_aux = line_data[aux_metric_var].values * 100

        ax.plot(x, mean_aux, label=f'{aux_label} ({vary_param}={val})',
                color=warm_palette[i % 3], linestyle='-', linewidth=2, marker='o', markersize=4, alpha=0.6)
        # 填充阴影区域 (alpha 设置为 0.15~0.2 比较高级)
        ax.fill_between(x, mean_aux - std_aux, mean_aux + std_aux,
                        color=warm_palette[i % 3], alpha=0.15)

        # --- 2. 绘制核心指标 G_W|G_M (虚线 + 阴影) ---
        mean_cond = line_data['G_W|G_M:Mean'].values * 100
        std_cond = line_data['G_W|G_M:Var'].values * 100

        ax.plot(x, mean_cond, label=f'$R_{{cond}}$ ($G_W|G_M$) ({vary_param}={val})',
                color=cool_palette[i % 3], linestyle='--', linewidth=2, marker='s', markersize=4, alpha=0.6)
        # 填充阴影区域
        ax.fill_between(x, mean_cond - std_cond, mean_cond + std_cond,
                        color=cool_palette[i % 3], alpha=0.15)

    # 细节美化
    ax.set_title(f'{dataset_name} (Fixed {fixed_param}=0.01)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Radius', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_ylim(50, 105)

    # 设置网格，只保留 y 轴主网格，更加清爽
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.3)
    ax.xaxis.grid(False)

    # 去掉上方和右方的边框 (学术图表常用)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xticks(sub_df['Radius'].unique())
    # 图例调整：两列显示，放在图表下方
    ax.legend(prop={'size': 8}, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False)


def plot_sensitivity3(data, fixed_param, vary_param, dataset_name, ax):
    """
    线宽加粗、字体调大、图例优化版
    """
    # 筛选固定参数为 0.01 的数据
    sub_df = data[data[fixed_param] == 0.01].copy()
    vary_values = sorted(sub_df[vary_param].unique())

    is_l1_fixed = (fixed_param == 'Lambda_1')
    aux_metric_mean = 'G_W:Mean' if is_l1_fixed else 'G_M: Mean'
    aux_metric_var = 'G_W:Var' if is_l1_fixed else 'G_M: Var'
    aux_label = '$R_{rec}$ ($G_W$)' if is_l1_fixed else '$R_{flip}$ ($G_M$)'

    # 颜色设置
    warm_colors = ['#f15e14', '#f49f00', '#eb596a']
    cool_colors = ['#18840e', '#2653ae', '#1fd0af']

    # 设置线宽和标记大小的全局变量
    LW = 2.5  # 线宽
    MS = 8  # 标记大小
    ELW = 1.5  # 误差棒线宽

    for i, val in enumerate(vary_values):
        line_data = sub_df[sub_df[vary_param] == val].sort_values('Radius')

        # 1. 绘制辅助指标数据 (实线)
        ax.errorbar(
            line_data['Radius'], line_data[aux_metric_mean] * 100,
            yerr=line_data[aux_metric_var] * 100,
            label=f'{aux_label} ({val})',
            color=warm_colors[i % len(warm_colors)],
            linestyle='-',
            linewidth=LW,  # 加粗线条
            marker='o',
            markersize=MS,  # 调大标记点
            alpha=0.8,
            capsize=4,
            elinewidth=ELW
        )

        # 2. 绘制核心指标 G_W|G_M (虚线)
        ax.errorbar(
            line_data['Radius'], line_data['G_W|G_M:Mean'] * 100,
            yerr=line_data['G_W|G_M:Var'] * 100,
            label=f'$R_{{cond}}$ ({val})',
            color=cool_colors[i % len(cool_colors)],
            linestyle='--',
            linewidth=LW,  # 加粗线条
            marker='s',
            markersize=MS,  # 调大标记点
            alpha=0.8,
            capsize=4,
            elinewidth=ELW
        )

    # 标题和标签字体调大
    ax.set_title(f'{dataset_name} (Fixed {fixed_param}=0.01)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Radius ($r$)', fontsize=14)
    ax.set_ylabel('Success Rate (%)', fontsize=14)

    # 刻度字体调大
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_ylim(50, 105)
    ax.set_xticks(sub_df['Radius'].unique())
    ax.grid(True, linestyle=':', alpha=0.6)

    # 图例字体调大，并根据情况调整位置
    # ncol=2 可以让图例不那么瘦长，frameon=True 增加边框提升清晰度
    ax.legend(prop={'size': 10, 'weight': 'normal'},
              loc='best',
              ncol=2,
              frameon=True,
              edgecolor='gray')

# # 加载与绘图部分逻辑不变
# # ... [保持之前的读取和循环部分] ...
# # 1. 加载数据
# file_path = './data/performance_l.xlsx'  # 确保是 xlsx 文件，若是 csv 请改用 read_csv 并注意编码
# try:
#     df = pd.read_excel(file_path)
# except:
#     df = pd.read_csv(file_path + " - Sheet1.csv")  # 兼容你之前的命名
#
# # 强制转换数值类型并处理可能的空值
# numeric_cols = ['Radius', 'Lambda_1', 'Lambda_2', 'G_M:Mean', 'G_M:Var',
#                 'G_W:Mean', 'G_W:Var', 'G_W|G_M:Mean', 'G_W|G_M:Var']
# for col in numeric_cols:
#     df[col] = pd.to_numeric(df[col], errors='coerce')
#
# datasets = df['Dataset'].dropna().unique()
#
# # 2. 遍历数据集画图
# for ds in datasets:
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
#     ds_data = df[df['Dataset'] == ds]
#
#     # 第一张图：固定 Lambda_1=0.01 (展示 G_W 与 G_W|G_M)
#     plot_sensitivity3(ds_data, 'Lambda_1', 'Lambda_2', ds, ax1)
#
#     # 第二张图：固定 Lambda_2=0.01 (展示 G_M 与 G_W|G_M)
#     plot_sensitivity3(ds_data, 'Lambda_2', 'Lambda_1', ds, ax2)
#
#     plt.tight_layout()
#     #plt.savefig(f"{ds}_sensitivity_analysis_3.png", dpi=300)
#     plt.show()

def plot_radius_influence(data, dataset_name, ax):
    """
    在 lambda1=0.01 且 lambda2=0.01 时，绘制不同 Radius 对三个指标的影响
    """
    # 筛选基准参数：lambda1=0.01 且 lambda2=0.01
    sub_df = data[(data['Lambda_1'] == 0.01) & (data['Lambda_2'] == 0.01)].copy()

    # 按 Radius 排序确保折线连续
    sub_df = sub_df.sort_values('Radius')

    x = sub_df['Radius']

    # 定义要绘制的三个指标及其对应的标准差列名、样式和颜色
    metrics_config = {
        'G_M': {
            'mean': 'G_M:Mean',
            'var': 'G_M:Var',
            'label': '$R_{flip}$ ($G_M$)',
            'color': '#FF8C00',  # 暖色-橙色
            'marker': 'o',
            'ls': '-'
        },
        'G_W': {
            'mean': 'G_W:Mean',
            'var': 'G_W:Var',
            'label': '$R_{rec}$ ($G_W$)',
            'color': '#1f77b4',  # 冷色-蓝色
            'marker': '^',
            'ls': '-'
        },
        'G_W|G_M': {
            'mean': 'G_W|G_M:Mean',
            'var': 'G_W|G_M:Var',
            'label': '$R_{cond}$ ($G_W|G_M$)',
            'color': '#d62728',  # 强调色-红色
            'marker': 's',
            'ls': '--'  # 核心指标用虚线区分
        }
    }

    for key, cfg in metrics_config.items():
        mean_val = sub_df[cfg['mean']] * 100
        std_val = sub_df[cfg['var']] * 100

        # 绘制带误差棒的折线
        ax.errorbar(
            x, mean_val, yerr=std_val,
            label=cfg['label'],
            color=cfg['color'],
            linestyle=cfg['ls'],
            marker=cfg['marker'],
            alpha=0.7,
            capsize=4,  # 误差棒横线
            elinewidth=1.2,  # 误差棒竖线宽度
            linewidth=1.5,
            markersize=6
        )

    # 布局美化
    ax.set_title(f'{dataset_name}\n($\lambda_1=0.01, \lambda_2=0.01$)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Radius ($r$)', fontsize=10)
    ax.set_ylabel('Success Rate (%)', fontsize=10)
    ax.set_ylim(40, 105)  # 设置范围以包含所有波动
    ax.set_xticks(x.unique())
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(prop={'size': 9}, loc='lower left')


def plot_radius_influence2(data, dataset_name, ax):
    """
    线宽加粗、字体调大、图例放大版
    用于绘制不同 Radius 对三个指标的影响趋势
    """
    # 筛选基准参数：lambda1=0.01 且 lambda2=0.01
    sub_df = data[(data['Lambda_1'] == 0.01) & (data['Lambda_2'] == 0.01)].copy()

    # 按 Radius 排序确保折线连续
    sub_df = sub_df.sort_values('Radius')
    x = sub_df['Radius']

    # 配置参数：增加线宽 LW, 标记大小 MS, 误差棒线宽 ELW
    LW = 3  # 主线宽
    MS = 10  # 标记点大小
    ELW = 2  # 误差棒竖线宽度
    CAP = 5  # 误差棒横帽大小

    metrics_config = {
        'G_M': {
            'mean': 'G_M:Mean',
            'var': 'G_M:Var',
            'label': '$R_{flip}$ ($G_M$)',
            'color': '#FF8C00',
            'marker': 'o',
            'ls': '-'
        },
        'G_W': {
            'mean': 'G_W:Mean',
            'var': 'G_W:Var',
            'label': '$R_{rec}$ ($G_W$)',
            'color': '#1f77b4',
            'marker': '^',
            'ls': '-'
        },
        'G_W|G_M': {
            'mean': 'G_W|G_M:Mean',
            'var': 'G_W|G_M:Var',
            'label': '$R_{cond}$ ($G_W|G_M$)',
            'color': '#d62728',
            'marker': 's',
            'ls': '--'
        }
    }

    for key, cfg in metrics_config.items():
        mean_val = sub_df[cfg['mean']] * 100
        std_val = sub_df[cfg['var']] * 100

        ax.errorbar(
            x, mean_val, yerr=std_val,
            label=cfg['label'],
            color=cfg['color'],
            linestyle=cfg['ls'],
            marker=cfg['marker'],
            alpha=0.8,
            capsize=CAP,
            elinewidth=ELW,
            linewidth=LW,  # 线条加粗
            markersize=MS  # 标记点加大
        )

    # ---- 字体与布局调整 ----
    # 标题字体调大
    ax.set_title(f'{dataset_name} Sensitivity to $r$\n($\lambda_1=0.01, \lambda_2=0.01$)',
                 fontsize=20, fontweight='bold', pad=15)

    # 轴标签字体调大
    ax.set_xlabel('Radius ($r$)', fontsize=16)
    ax.set_ylabel('Success Rate (%)', fontsize=16)

    # 坐标轴数字刻度调大
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_ylim(40, 105)
    ax.set_xticks(x.unique())
    ax.grid(True, linestyle=':', alpha=0.6)

    # 图例调大：size 调至 11-12，增加 frameon 使其在复杂背景下更清晰
    ax.legend(prop={'size': 14, 'weight': 'normal'},
              loc='lower left',
              frameon=True,
              edgecolor='gray')


#############################################
# # 扰动半径折线图
# # 1. 数据加载与预处理
# file_path = './data/performance_l.xlsx'
# try:
#     df = pd.read_excel(file_path)
# except Exception as e:
#     print(f"读取文件失败: {e}")
#     exit()
#
# # 强制转换数值列
# numeric_cols = ['Radius', 'Lambda_1', 'Lambda_2', 'G_M:Mean', 'G_M:Var',
#                 'G_W:Mean', 'G_W:Var', 'G_W|G_M:Mean', 'G_W|G_M:Var']
# for col in numeric_cols:
#     df[col] = pd.to_numeric(df[col], errors='coerce')
#
# datasets = df['Dataset'].unique()
#
# # 2. 遍历数据集，每个数据集画一张图
# for ds in datasets:
#     # 创建画布，大小可根据需要调整
#     fig, ax = plt.subplots(figsize=(7, 6))
#     ds_data = df[df['Dataset'] == ds]
#
#     plot_radius_influence2(ds_data, ds.upper(), ax)
#
#     plt.tight_layout()
#     # 保存图片
#     plt.savefig(f"{ds}_radius_impact.png", dpi=300)
#     plt.show()


########################################
# 正则化系数柱状图
# 1. 加载数据
# file_path = './data/performance_l.xlsx'
# df = pd.read_excel(file_path)
#
# # 清理列名空格
# df.columns = [c.strip() for c in df.columns]
#
# # 统一映射列名
# column_map = {
#     'G_M: Mean': 'gm_m', 'G_M:Mean': 'gm_m',
#     'G_M: Var': 'gm_v', 'G_M:Var': 'gm_v',
#     'G_W: Mean': 'gw_m', 'G_W:Mean': 'gw_m',
#     'G_W: Var': 'gw_v', 'G_W:Var': 'gw_v',
#     'G_W|G_M Mean': 'cond_m', 'G_W|G_M:Mean': 'cond_m',
#     'G_W|G_M:Var': 'cond_v'
# }
# df = df.rename(columns=column_map)
#
# # 确保数值类型并【只转换一次百分比】
# target_cols = ['gm_m', 'gm_v', 'gw_m', 'gw_v', 'cond_m', 'cond_v']
# for col in target_cols:
#     df[col] = pd.to_numeric(df[col], errors='coerce') * 100
#
# # 确保参数列也是数值
# for col in ['Radius', 'Lambda_1', 'Lambda_2']:
#     df[col] = pd.to_numeric(df[col], errors='coerce')
#
# datasets = df['Dataset'].dropna().unique()
# bar_width = 0.35
#
# # 2. 遍历数据集生成图表
# for ds in datasets:
#     ds_data = df[(df['Dataset'] == ds) & (df['Radius'] == 0.3)].copy()
#
#     # 设置全局绘图参数
#     LW = 3.0  # 折线与边框线宽
#     ELW = 2.0  # 误差棒线宽
#     MS = 10  # 标记点大小
#     CAP = 6  # 误差棒横帽大小
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
#
#     # --- 第一张图：固定 Lambda_1 = 0.01, 观察 Lambda_2 ---
#     l1_fixed = ds_data[ds_data['Lambda_1'] == 0.01].sort_values('Lambda_2')
#     l1_fixed = l1_fixed[l1_fixed['Lambda_2'].isin([0.001, 0.01, 0.1])]
#     idx = np.arange(len(l1_fixed))
#
#     # 绘制柱状图 (增加 edgecolor 和 linewidth)
#     ax1.bar(idx - bar_width / 2, l1_fixed['gw_m'], bar_width, yerr=l1_fixed['gw_v'],
#             label='$R_{rec}$ ($G_W$)', color='#FF8C00', alpha=0.7,
#             capsize=CAP, error_kw={'elinewidth': ELW}, edgecolor='none', linewidth=1.5)
#     ax1.bar(idx + bar_width / 2, l1_fixed['cond_m'], bar_width, yerr=l1_fixed['cond_v'],
#             label='$R_{cond}$ ($G_W|G_M$)', color='#1f77b4', alpha=0.7,
#             capsize=CAP, error_kw={'elinewidth': ELW}, edgecolor='none', linewidth=1.5)
#
#     # 连接柱子顶部的折线 (加粗并加大标记)
#     ax1.plot(idx - bar_width / 2, l1_fixed['gw_m'], color='#FF8C00', marker='o', linewidth=LW, markersize=MS, alpha=0.5)
#     ax1.plot(idx + bar_width / 2, l1_fixed['cond_m'], color='#1f77b4', marker='s', linewidth=LW, markersize=MS,
#              alpha=0.5)
#
#     ax1.set_title(f'{ds.upper()} - Impact of $\lambda_2$\n($\lambda_1=0.01$)', fontsize=20, fontweight='bold', pad=15)
#     ax1.set_xlabel('$\lambda_2$', fontsize=18, labelpad=10)
#     ax1.set_ylabel('Success Rate (%)', fontsize=18, labelpad=10)
#     ax1.set_xticks(idx)
#     ax1.set_xticklabels(l1_fixed['Lambda_2'], fontsize=16)
#     ax1.tick_params(axis='y', labelsize=16)
#     ax1.set_ylim(0, 105)
#     ax1.legend(fontsize=14, loc='lower right', frameon=True, edgecolor='gray')
#     ax1.grid(axis='y', linestyle=':', alpha=0.6)
#
#     # --- 第二张图：固定 Lambda_2 = 0.01, 观察 Lambda_1 ---
#     l2_fixed = ds_data[ds_data['Lambda_2'] == 0.01].sort_values('Lambda_1')
#     l2_fixed = l2_fixed[l2_fixed['Lambda_1'].isin([0.001, 0.01, 0.1])]
#     idx2 = np.arange(len(l2_fixed))
#
#     # 绘制柱状图
#     ax2.bar(idx2 - bar_width / 2, l2_fixed['gm_m'], bar_width, yerr=l2_fixed['gm_v'],
#             label='$R_{flip}$ ($G_M$)', color='#eb596a', alpha=0.7,
#             capsize=CAP, error_kw={'elinewidth': ELW}, edgecolor='none', linewidth=1.5)
#     ax2.bar(idx2 + bar_width / 2, l2_fixed['cond_m'], bar_width, yerr=l2_fixed['cond_v'],
#             label='$R_{cond}$ ($G_W|G_M$)', color='#1f77b4', alpha=0.7,
#             capsize=CAP, error_kw={'elinewidth': ELW}, edgecolor='none', linewidth=1.5)
#
#     # 连接柱子顶部的折线 (加粗并加大标记)
#     ax2.plot(idx2 - bar_width / 2, l2_fixed['gm_m'], color='#eb596a', marker='o', linewidth=LW, markersize=MS,
#              alpha=0.5)
#     ax2.plot(idx2 + bar_width / 2, l2_fixed['cond_m'], color='#1f77b4', marker='s', linewidth=LW, markersize=MS,
#              alpha=0.5)
#
#     ax2.set_title(f'{ds.upper()} - Impact of $\lambda_1$\n($\lambda_2=0.01$)', fontsize=20, fontweight='bold', pad=15)
#     ax2.set_xlabel('$\lambda_1$', fontsize=18, labelpad=10)
#     ax2.set_ylabel('Success Rate (%)', fontsize=18, labelpad=10)
#     ax2.set_xticks(idx2)
#     ax2.set_xticklabels(l2_fixed['Lambda_1'], fontsize=16)
#     ax2.tick_params(axis='y', labelsize=16)
#     ax2.set_ylim(50, 105)
#     ax2.legend(fontsize=14, loc='lower right', frameon=True, edgecolor='gray')
#     ax2.grid(axis='y', linestyle=':', alpha=0.6)
#
#     plt.tight_layout()
#     plt.savefig(f"{ds}_lambda_bar_enhanced.png", dpi=300)
#     plt.show


# #######################################
# # 处理掩码和稀疏度excel文件
# import glob
# # ============================================================
# # Excel 文件夹路径
# # ============================================================
# #excel_dir = "./outputs/EXCELS/paths2_err"
# excel_dir = "./outputs/EXCELS/paths3"
#
# # 获取所有 xlsx 文件
# excel_files = glob.glob(os.path.join(excel_dir, "*.xlsx"))
#
# # ============================================================
# # 用于保存所有 excel 的统计结果
# # ============================================================
# all_results = []
#
# # ============================================================
# # 遍历每个 excel 文件
# # ============================================================
# for excel_path in excel_files:
#
#     print("\n" + "=" * 80)
#     print(f"Processing: {os.path.basename(excel_path)}")
#     print("=" * 80)
#
#     try:
#         # 读取工作表
#         df_path = pd.read_excel(excel_path, sheet_name="Path_Mask_Weights")
#         df_sparsity = pd.read_excel(excel_path, sheet_name="Sparsity_Info")
#
#         # ============================================================
#         # Part 1: 统计 path 长度
#         # ============================================================
#
#         # 按 path 聚合
#         path_lengths = (
#             df_path
#             .groupby(["Sample_Idx", "task", "Path_ID"])["Abs_Score"]
#             .sum()
#             .reset_index(name="Path_Length")
#         )
#
#         print("=" * 60)
#         print("Path Length Statistics")
#         print("=" * 60)
#
#         # 用于保存当前文件统计结果
#         result_row = {
#             "file_name": os.path.basename(excel_path)
#         }
#
#         # 分别统计 none_gm 和 gm_gw
#         for task_keyword in ["none_gm", "gm_gw"]:
#
#             task_df = path_lengths[
#                 path_lengths["task"].str.contains(task_keyword, na=False)
#             ]
#
#             if len(task_df) == 0:
#                 print(f"\n[{task_keyword}] No data found.")
#
#                 result_row[f"{task_keyword}_avg_path"] = None
#                 result_row[f"{task_keyword}_max_path"] = None
#                 result_row[f"{task_keyword}_min_path"] = None
#
#                 continue
#
#             avg_len = task_df["Path_Length"].mean()
#             max_len = task_df["Path_Length"].max()
#             min_len = task_df["Path_Length"].min()
#
#             print(f"\n[{task_keyword}]")
#             print(f"Average Path Length : {avg_len:.6f}")
#             print(f"Maximum Path Length : {max_len:.6f}")
#             print(f"Minimum Path Length : {min_len:.6f}")
#
#             # 保存结果
#             result_row[f"{task_keyword}_avg_path"] = avg_len
#             result_row[f"{task_keyword}_max_path"] = max_len
#             result_row[f"{task_keyword}_min_path"] = min_len
#
#         # ============================================================
#         # Part 2: 统计稀疏度
#         # ============================================================
#
#         print("\n" + "=" * 60)
#         print("Sparsity Statistics")
#         print("=" * 60)
#
#         # 提前在路径表中过滤出 Path_ID == 2 (即提取出的最长一条关键路径)，统计各样本各任务的条目数(边数)
#         path_id2_df = df_path[df_path["Path_ID"] == 2]
#         path_counts = (
#             path_id2_df
#             .groupby(["Sample_Idx", "task"])
#             .size()
#             .reset_index(name="path_edge_count")
#         )
#
#         #for graph_type in ["g_m", "g_w"]:
#         for graph_type in ["none_gm", "gm_gw"]:
#
#             graph_df = df_sparsity[
#                 df_sparsity["task"].str.contains(graph_type, na=False)
#             ]
#
#             if len(graph_df) == 0:
#                 print(f"\n[{graph_type}] No data found.")
#                 result_row[f"{graph_type}_sparsity"] = None
#                 continue
#             print(f"\n[{graph_type}]")
#
#             # 干预稀疏度
#             avg_sparsity = graph_df["Sparsity_Ratio"].mean()
#             final_intervention_sparsity = 1.0 - avg_sparsity
#             result_row[f"{graph_type}_sparsity"] = final_intervention_sparsity
#             print(f"Intervention Sparsity (1 - Sparsity_Ratio) : {final_intervention_sparsity:.6f}")
#
#             # 路径稀疏度
#             graph_df = pd.merge(graph_df, path_counts, on=["Sample_Idx", "task"], how="left")
#             graph_df["path_edge_count"] = graph_df["path_edge_count"].fillna(0)
#             # 逐个样本计算单点路径稀疏度：1 - (路径边数 / 图中总边数)
#             graph_df["sample_path_spa"] = 1.0 - (graph_df["path_edge_count"] / graph_df["Total_Edges"])
#             # 对所有样本求均值，作为当前文件的最终路径稀疏度
#             final_path_spa = graph_df["sample_path_spa"].mean()
#             # 存入 Excel，对应列名分化为 none_gm_spa 和 gm_gw_spa
#             result_row[f"{graph_type}_path_spa"] = final_path_spa
#             print(f"Path Sparsity         (1 - Path_Edges/Total_Edges): {final_path_spa:.6f}")
#
#         # 当前 excel 的结果加入总列表
#         all_results.append(result_row)
#
#     except Exception as e:
#         print(f"[ERROR] Failed processing {excel_path}")
#         print(e)
#
# # ============================================================
# # 所有结果保存为一个新的 excel
# # ============================================================
# summary_df = pd.DataFrame(all_results)
#
# o_path = "./outputs/EXCELS"
# output_path = os.path.join(o_path, "all_ps4_statistics_summary.xlsx")
#
# summary_df.to_excel(output_path, index=False)
#
# print("\n" + "=" * 80)
# print(f"All statistics saved to:")
# print(output_path)
# print("=" * 80)


# ################################
# # 处理对比实验稀疏度柱状图
# # 1. 内置实验数据（9个数，百分比单位：如 15.5 代表 15.5%）
# datasets = ['Last.fm', 'Aug_Citation', 'ACM']
# cage_data = [84.65, 89.15, 98.59]         # CAGE 稀疏度
# sem_data = [99.02, 98.67, 99.78]            # SemExplainer 稀疏度
# our_data = [94.27, 97.91, 91.96]            # 本方法(ABLE-g) 稀疏度
# our_data_pa = [99.96, 99.99, 92.74]            # 本方法(ABLE-g-PA) 稀疏度
#
# # 2. 设置柱状图宽度与横轴坐标槽位
# x = np.arange(len(datasets))
# width = 0.22
#
# plt.figure(figsize=(8.5, 4.5), dpi=120)
#
# # 3. 绘制多组对比柱状图
# b1 = plt.bar(x - 1.5*width, cage_data, width, label='CAGE', color='#4682B4', edgecolor='none', alpha=0.8)
# b2 = plt.bar(x - 0.5*width, sem_data, width, label='SemExplainer', color='#2ECC71', edgecolor='none', alpha=0.8)
# b3 = plt.bar(x + 0.5*width, our_data, width, label='Ours', color='#ffc905', edgecolor='none', alpha=0.8)
# b4 = plt.bar(x + 1.5*width, our_data_pa, width, label='Ours-PA', color='#ff4000', edgecolor='none', alpha=0.8)
#
# # 为每组柱子添加头部数值标签
# plt.bar_label(b1, fmt='%.2f', padding=1, fontsize=12)
# plt.bar_label(b2, fmt='%.2f', padding=1, fontsize=12)
# plt.bar_label(b3, fmt='%.2f', padding=1, fontsize=12)
# plt.bar_label(b4, fmt='%.2f', padding=1, fontsize=12)
#
# # 4. 优化图表细节、标签及区间约束
# plt.ylabel('Sparsity Ratio (%)', fontsize=16)
# plt.xticks(x, datasets, fontsize=14)
# plt.ylim(80, 103)
#
# # 5. 美化图例与网格
# plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=12, loc='lower right')
# plt.grid(axis='y', linestyle='--', alpha=0.5)
# plt.tight_layout()
#
# # 6. 显示或保存图像
# plt.savefig('sparsity_comparison2.png', bbox_inches='tight') # 如需保存取消本行注释
# plt.show()


#################################
# 处理对比实验扰动成功率柱状图
# 1. 内置实验数据：均值（Mean）与标准差（Std）
datasets = ['Last.fm', 'Aug_Citation', 'ACM']

# 格式：[Last.fm, Aug_Citation, ACM]
# # FID-
# cage_mean = [27.85, 11.74, 68.64]
# cage_std  = [3.1,  3.9,  1.5]
#
# sem_mean  = [18.13, 10.99, 24.55]
# sem_std   = [4.1,  4.1,  5.8]
#
# our_mean  = [6.35, 2.92, 34.27]
# our_std   = [0.2,  1.2,  3.2]

# FID+
cage_mean = [90.38, 70.81, 91.49]
cage_std  = [0.1,  1.3,  1.2]

sem_mean  = [90.41, 71.56, 91.81]
sem_std   = [2.1,  1.1,  2.7]

our_mean  = [88.88, 99.95, 99.68]
our_std   = [0.7,  0.05,  0.2]

# 2. 设置布局参数
x = np.arange(len(datasets))
width = 0.25

plt.figure(figsize=(7, 4.5), dpi=120)

# 3. 绘制带有误差棒的柱状图 (yerr传入标准差, capsize设置横线宽度, zorder=3让柱子在网格线上方)
b1 = plt.bar(x - width, cage_mean, width, yerr=cage_std, label='CaGE',
             color='#4682B4', edgecolor='none', alpha=0.8, capsize=4, zorder=3)
b2 = plt.bar(x, sem_mean, width, yerr=sem_std, label='SemExplainer',
             color='#2ECC71', edgecolor='none', alpha=0.8, capsize=4, zorder=3)
b3 = plt.bar(x + width, our_mean, width, yerr=our_std, label='Ours',
             color='#E67E22', edgecolor='none', alpha=0.9, capsize=4, zorder=3)

# 4. 在柱头标注均值数值
plt.bar_label(b1, fmt='%.2f', padding=1, fontsize=12)
plt.bar_label(b2, fmt='%.2f', padding=1, fontsize=12)
plt.bar_label(b3, fmt='%.2f', padding=1, fontsize=12)

# 5. 优化图表细节与区间约束
#plt.ylabel('$R_{flip}$ (%)', fontsize=18)
#plt.ylabel('$R_{rev}$ (%)', fontsize=18)
# plt.ylabel('FID- (%)', fontsize=18)
plt.ylabel('FID+ (%)', fontsize=18)
plt.xticks(x, datasets, fontsize=14)
# plt.ylim(0, 100)  # 纵轴严格限定在 60~100 范围（放宽到105防止数据标签出界）
#plt.ylim(65, 100)
# plt.ylim(0, 100) # FID-
plt.ylim(0, 105) # FID+

# 6. 美化图例与网格dingding
plt.legend(frameon=True, facecolor='white', edgecolor='none', fontsize=12, loc='lower left')
plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
plt.tight_layout()

# 7. 先保存，后显示（彻底解决保存出来是空白图片的问题）
# plt.savefig('FID-_comparison.png', bbox_inches='tight')
plt.savefig('FID+_comparison.png', bbox_inches='tight')
plt.show()




# #################################
# # 处理干预稀疏度实验扰动成功率折线
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import re
#
# # ==========================================
# # 1. 读取数据与预处理
# # ==========================================
# df = pd.read_excel('./outputs/EXCELS/all_ps4_statistics_summary.xlsx')
#
#
# # 解析文件名提取关键参数 (数据集名称, lambda1, lambda2)
# def parse_filename(filename):
#     # 正则匹配：以数据集开头，紧跟 lambda1 和 lambda2 的数值
#     pattern = r"^([a-zA-Z0-9_\.-]+)_lambda([0-9.]+)_([0-9.]+)_"
#     match = re.match(pattern, filename)
#     if match:
#         return match.group(1), float(match.group(2)), float(match.group(3))
#     return None, None, None
#
#
# # 应用解析函数
# df[['dataset', 'lambda1', 'lambda2']] = df['file_name'].apply(
#     lambda x: pd.Series(parse_filename(x))
# )
#
# # 将稀疏度小数转换为百分比 (%)
# df['sp1'] = df['none_gm_sparsity'] * 100
# df['sp2'] = df['gm_gw_sparsity'] * 100
#
# # 核心超参数横轴刻度
# lambda_ticks = [0.001, 0.01, 0.1, 0.3]
# x_indices = np.arange(len(lambda_ticks))  # 用 0, 1, 2, 3 作为等距横轴，防止坐标轴被拉伸
#
# # 颜色系定义
# # cold_colors = ['#1abc9c', '#3498db', '#9b59b6', '#2c3e50']  # 冷色系 (绿、蓝、紫、深蓝紫)
# # warm_colors = ['#f1c40f', '#e67e22', '#e74c3c', '#c0392b']  # 暖色系 (黄、橙、红、深红)
#
# cold_colors = ['#00715b', '#005e9d', '#2b007a', '#003e7c']
# warm_colors = ['#ffe061', '#ffb16c', '#ff897d', '#ff8fb6']
#
# unique_datasets = df['dataset'].dropna().unique()
#
# print(unique_datasets)
#
# for dataset_name in unique_datasets:
#     sub_df = df[df['dataset'] == dataset_name]
#
#     fig, ax = plt.subplots(figsize=(7.5, 6), dpi=120)  # 稍微加宽一点点，留足3列图例的空间
#
#     # 用来按顺序严格收集这 9 条线的数据对象和它们的 label
#     line_objects = []
#     line_labels = []
#
#     # ----------------------------------------------------
#     # (1) 绘制第 1 类线：第一阶段 Sp1 随 Lambda1 变化的虚线
#     # ----------------------------------------------------
#     sp1_grouped = sub_df.groupby('lambda1')['sp1'].mean().reindex(lambda_ticks)
#     l_sp1, = ax.plot(x_indices, sp1_grouped.values, linestyle='--', linewidth=4.5, ms=8,
#                      marker='o', color='black', alpha=0.9, label='$S_{p1}$ vs $\lambda_1$', zorder=6)
#
#     # 收集第一类线
#     line_objects.append(l_sp1)
#     line_labels.append(fr'$S_{{p1}}$vs$\lambda_1$')
#
#     # 【核心小技巧】：因为第一列只有1条线，而二、三列各有4条线。为了让多列排版高度对齐，
#     # 我们在第一列下面手动连续塞入 3 个“隐形占位符（空线与空标签）”，把空间顶起来。
#     for _ in range(3):
#         dummy_line = plt.plot([], [], linestyle='', marker='')[0]  # 创建一条什么都没有的空线
#         line_objects.append(dummy_line)
#         line_labels.append('')  # 空白文本
#
#     # ----------------------------------------------------
#     # (2) 绘制第 2 类线：冷色实线 (固定 Lambda1，随 Lambda2 变化)
#     # ----------------------------------------------------
#     for idx, l1_val in enumerate(lambda_ticks):
#         fixed_l1_df = sub_df[sub_df['lambda1'] == l1_val].sort_values('lambda2')
#         sp2_vs_l2 = fixed_l1_df.groupby('lambda2')['sp2'].mean().reindex(lambda_ticks)
#
#         l_cold, = ax.plot(x_indices, sp2_vs_l2.values, linestyle='-', linewidth=4, marker='s', ms=6,
#                           color=cold_colors[idx], alpha=0.7, zorder=11)
#
#         # 收集第二类线
#         line_objects.append(l_cold)
#         line_labels.append(fr'$S_{{p2}}$vs$\lambda_2$ ($\lambda_1$={l1_val})')
#
#     # ----------------------------------------------------
#     # (3) 绘制第 3 类线：暖色实线 (固定 Lambda2，随 Lambda1 变化)
#     # ----------------------------------------------------
#     # for idx, l2_val in enumerate(lambda_ticks):
#     #     fixed_l2_df = sub_df[sub_df['lambda2'] == l2_val].sort_values('lambda1')
#     #     sp2_vs_l1 = fixed_l2_df.groupby('lambda1')['sp2'].mean().reindex(lambda_ticks)
#     #
#     #     l_warm, = ax.plot(x_indices, sp2_vs_l1.values, linestyle='-', linewidth=3.5, marker='^', ms=6,
#     #                       color=warm_colors[idx], alpha=0.4, zorder=10)
#     #
#     #     # 收集第三类线
#     #     line_objects.append(l_warm)
#     #     line_labels.append(fr'$S_{{p2}}$vs$\lambda_1$ ($\lambda_2$={l2_val})')
#
#     # ----------------------------------------------------
#     # 核心转换：重新计算索引，让原本“按列排布”的数据强行实现三列完美对齐
#     # ----------------------------------------------------
#     # 此时我们一共有 12 个对象（1条真虚线+3条占位空线+4条冷色线+4条暖色线）
#     # 重新洗牌，让一、二、三列在横向铺开时正好分别对应：[第1类线、第2类线、第3类线]
#     reordered_objects = []
#     reordered_labels = []
#
#     # 4行3列转换公式
#     for c in range(2):
#         for r in range(4):
#             target_idx = c * 4 + r  # 强行把原本按列存储的数据以行的方式提取出来
#             reordered_objects.append(line_objects[target_idx])
#             reordered_labels.append(line_labels[target_idx])
#
#     # ----------------------------------------------------
#     # 3. 完美配置图例与其他图表细节
#     # ----------------------------------------------------
#     # 使用重排后的对象，设置 ncol=3，强制分成3列排布，位置锁定在图内右上角
#     ax.legend(reordered_objects, reordered_labels, loc="lower right",
#               ncol=2, frameon=True, facecolor='white', edgecolor='none', fontsize=11)
#
#     # 细节美化
#     ax.set_title(f'Sparsity Ablation Analysis - {dataset_name}', fontsize=20, fontweight='bold', pad=15)
#     ax.set_xlabel('Hyperparameter Setting ($\lambda$)', fontsize=16)
#     ax.set_ylabel('Intervention Sparsity Ratio (%)', fontsize=16)
#
#     ax.set_xticks(x_indices)
#     ax.set_xticklabels([str(t) for t in lambda_ticks], fontsize=14)
#     ax.set_ylim(75, 100)
#
#     ax.grid(axis='both', linestyle='--', alpha=0.5)
#     plt.tight_layout()
#
#     # 先保存，后显示
#     output_filename = f'sparsity_ablation_{dataset_name}3.png'
#     plt.savefig(output_filename, bbox_inches='tight')
#     print(f"[成功] 矩阵式对齐图例已导出: {output_filename}")
#     plt.show()
# # # ==========================================
# # # 2. 开始为每个数据集独立绘制折线图
# # # ==========================================
# # unique_datasets = df['dataset'].dropna().unique()
# #
# # for dataset_name in unique_datasets:
# #     # 筛选当前数据集的数据
# #     sub_df = df[df['dataset'] == dataset_name]
# #
# #     plt.figure(figsize=(9, 6), dpi=120)
# #
# #     # ----------------------------------------------------
# #     # (1) 绘制第一阶段 Sp1 随 Lambda1 变化的虚线
# #     # ----------------------------------------------------
# #     # 因为 lambda2 不影响 sp1，所以直接按 lambda1 分组求均值
# #     sp1_grouped = sub_df.groupby('lambda1')['sp1'].mean().reindex(lambda_ticks)
# #     plt.plot(x_indices, sp1_grouped.values, linestyle='--', linewidth=3,
# #              marker='o', color='black', alpha=0.9, label='$S_{p1}$ vs $\lambda_1$ (Stage 1)', zorder=5)
# #
# #     # ----------------------------------------------------
# #     # (2) 绘制第二阶段 Sp2 随 Lambda2 变化 (固定 Lambda1) 的冷色实线
# #     # ----------------------------------------------------
# #     for idx, l1_val in enumerate(lambda_ticks):
# #         # 筛选固定 lambda1 后的数据
# #         fixed_l1_df = sub_df[sub_df['lambda1'] == l1_val].sort_values('lambda2')
# #         # 聚合防止有重复实验数据
# #         sp2_vs_l2 = fixed_l1_df.groupby('lambda2')['sp2'].mean().reindex(lambda_ticks)
# #
# #         plt.plot(x_indices, sp2_vs_l2.values, linestyle='-', linewidth=2.7, marker='s', ms=5,
# #                  color=cold_colors[idx], alpha=0.7,
# #                  label='$S_{{p2}}$ vs $\lambda_2$ ($\lambda_1$={})'.format(l1_val), zorder=3)
# #
# #     # ----------------------------------------------------
# #     # (3) 绘制第二阶段 Sp2 随 Lambda1 变化 (固定 Lambda2) 的暖色实线
# #     # ----------------------------------------------------
# #     for idx, l2_val in enumerate(lambda_ticks):
# #         # 筛选固定 lambda2 后的数据
# #         fixed_l2_df = sub_df[sub_df['lambda2'] == l2_val].sort_values('lambda1')
# #         # 聚合
# #         sp2_vs_l1 = fixed_l2_df.groupby('lambda1')['sp2'].mean().reindex(lambda_ticks)
# #
# #         plt.plot(x_indices, sp2_vs_l1.values, linestyle='-', linewidth=2.2, marker='^', ms=4,
# #                  color=warm_colors[idx], alpha=0.4,
# #                  label='$S_{{p2}}$ vs $\lambda_1$ ($\lambda_2$={})'.format(l2_val), zorder=4)
# #
# #     # ----------------------------------------------------
# #     # 4. 优化图表细节与规范限制
# #     # ----------------------------------------------------
# #     plt.title(f'Sparsity Ablation Analysis - {dataset_name}', fontsize=18, fontweight='bold', pad=15)
# #     plt.xlabel('Hyperparameter Setting ($\lambda$)', fontsize=14)
# #     plt.ylabel('Intervention Sparsity Ratio (%)', fontsize=14)
# #
# #     # 关键修改：将横轴刻度从非线性数值映射为等距的字符串标签，防止0.001和0.3间距过大
# #     plt.xticks(x_indices, [str(t) for t in lambda_ticks], fontsize=12)
# #     # 纵轴严格限制在 0 ~ 25%
# #     plt.ylim(0, 20)
# #
# #     # 调整图例放在图表右侧，防止遮挡曲线
# #     plt.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='none', fontsize=9)
# #     #plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, fontsize=9)
# #     plt.grid(axis='both', linestyle='--', alpha=0.5)
# #     plt.tight_layout()
# #
# #     # 保存与展示
# #     output_filename = f'sparsity_ablation_{dataset_name}.png'
# #     plt.savefig(output_filename, bbox_inches='tight')
# #     print(f"[成功] 已生成并保存数据集 {dataset_name} 的消融实验图至: {output_filename}")
# #     plt.show()