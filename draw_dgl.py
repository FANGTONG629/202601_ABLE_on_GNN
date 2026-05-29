from tkinter import image_names

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



def draw_able_graph_eweight(exres, dataset_name, img_name=None, save_dir="./outputs/GRAPH_EW"):
    """
        按照掩码 eweight_dict 调整边的透明度和粗细，展示解释子图。
        参考 visualize_neighborhood_tsne 结构提取数据。
        """
    # 1. 路径与命名准备
    from utils import get_mask_delta
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 2. 提取数据结构 (参考 exres 结构)
    # 默认绘制第一对对抗样本中的 G_M (反事实解释图)
    adv_pairs = exres.get("adv_pairs", [])
    predictions = exres.get("predictions", [])
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

def get_labels_dict(G, dataset_name):
    labels_dict = {}
    # 尝试导入平铺的名称列表

    import datasets.ACM_field as acmf

    for node in G.nodes():
        ntype, nid = node.split('_')[0], node.split('_')[1]

        # 如果是 ACM 且节点类型是 field
        if dataset_name == 'ACM' and ntype == 'field' :
            try:
                # 直接通过序号索引获取名称 (如 Hardware-ARITHMETIC AND LOGIC STRUCTURES)
                labels_dict[node] = acmf.get_main_name_by_index(int(nid))
            except IndexError:
                labels_dict[node] = node  # 越界则 fallback
        else:
            # 其他节点类型保持原样，或者缩短一点 (如 paper_123 -> p123)
            labels_dict[node] = node
    return labels_dict

def draw_able_weight_path(exres, dataset_name, img_name="", save_dir="./outputs/GRAPH_EW"):
    """
        按照掩码 eweight_dict 调整边的透明度和粗细，展示解释子图。
        参考 visualize_neighborhood_tsne 结构提取数据。
        """
    # 路径与命名准备
    from utils import get_mask_delta, get_paths, comp_g_paths_to_paths
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 提取数据结构 (参考 exres 结构)
    # 默认绘制第一对对抗样本中的 G_M (反事实解释图)
    adv_pairs = exres.get("adv_pairs", [])
    predictions = exres.get("predictions", [])
    if not adv_pairs:
        print("No adversarial pairs found in exres.")
        return

    # 提取当前样本的查询节点 ID 和类型
    q_src_id, q_src_nt = predictions[0]['src_nid'], exres['src_ntype']
    q_tgt_id, q_tgt_nt = predictions[0]['tgt_nid'], exres['tgt_ntype']

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
        # ===== [新增] global → local 映射 =====
        global2local = {}
        for ntype in g.ntypes:
            global_ids = g.ndata[dgl.NID][ntype].tolist()
            global2local[ntype] = {gid: i for i, gid in enumerate(global_ids)}

        # 定义两个绘图子任务：(任务名, mask1, mask2, 剪枝阈值)
        tasks = [
            (f"none_gm_{idx}", None, pair['G_M']['edge_mask'], 0.8),
            (f"gm_gw_{idx}", pair['G_M']['edge_mask'], pair['G_W']['edge_mask'], 0.4)
        ]

        for task_name, m1, m2, prune_threshold in tasks:
            mask_delta = get_mask_delta(g, mask1=m1, mask2=m2)# 计算差值掩码

            # 构建 global edge → mask 映射(忽略 etype)
            edge_mask_lookup = {}
            for etype in g.canonical_etypes:
                u, v = g.edges(etype=etype)
                u_nt, _, v_nt = etype
                u_global = g.ndata[dgl.NID][u_nt][u]
                v_global = g.ndata[dgl.NID][v_nt][v]
                masks = mask_delta[etype]
                for i in range(len(u)):
                    key = (u_nt, int(u_global[i]), v_nt, int(v_global[i]))
                    edge_mask_lookup.setdefault(key, []).append(masks[i].item())

            comp_path = get_paths( # 获取路径
                src_nid=torch.tensor([q_src_id]),
                src_ntype=q_src_nt,
                tgt_nid=torch.tensor([q_tgt_id]),
                tgt_ntype=q_tgt_nt,
                ghetero=g,
                edge_mask_dict=mask_delta,
                num_paths=7,
            )

            # ===== [修改] 只过滤“直连路径（长度为1）” =====
            filtered_paths = []
            for path in comp_path:
                # 只检查长度为1的路径
                if len(path) == 1:continue  # ❌ 丢掉直连路径
                filtered_paths.append(path)
                if len(filtered_paths)>=3: break

            paths_list = comp_g_paths_to_paths(g, filtered_paths) # 转换回原图 id

            if not paths_list:
                print("No paths found for task:", task_name)
                continue
            #else: print(paths_list)


            # 构建 NetworkX 图
            G = nx.Graph()
            path_nodes = set()


            # 2. 修改路径遍历逻辑，动态获取当前段的类型
            for path in paths_list:
                for can_etype, u_id, v_id in path:  # 必须解包 can_etype
                    this_u_nt, _, this_v_nt = can_etype  # 明确获取当前段的源/目标类型

                    u_n = f"{this_u_nt}_{int(u_id)}"
                    v_n = f"{this_v_nt}_{int(v_id)}"

                    # 查正向与反向，使用当前段真实的类型
                    key1 = (this_u_nt, int(u_id), this_v_nt, int(v_id))
                    key2 = (this_v_nt, int(v_id), this_u_nt, int(u_id))

                    masks = []
                    if key1 in edge_mask_lookup:
                        masks += edge_mask_lookup[key1]
                    if key2 in edge_mask_lookup:
                        masks += edge_mask_lookup[key2]

                    if not masks:
                        continue

                    m_val = sum(masks) / len(masks)
                    G.add_edge(u_n, v_n, weight=m_val)
                    path_nodes.update([u_n, v_n])


            # 绘图
            pos = nx.spring_layout(G, k=0.2, iterations=50, seed=42)
            plt.figure(figsize=(8, 8))

            # 绘制节点
            src_global_id = int(g.ndata[dgl.NID][q_src_nt][q_src_id])
            tgt_global_id = int(g.ndata[dgl.NID][q_tgt_nt][q_tgt_id])
            start_node = f"{q_src_nt}_{int(src_global_id)}"
            end_node = f"{q_tgt_nt}_{int(tgt_global_id)}"

            for node in G.nodes():
                # 提取节点类型
                # 注意：如果 node 是 "paper_1245"，这里能正确拿到 "paper"
                ntype = node.split('_')[0]

                # 判定是否为终端节点
                is_terminal = (node == start_node) or (node == end_node)
                # print(f"{node} {start_node} {end_node}")

                # 设置 fc (填充色), ec (边框色), lw (线宽), size (大小)
                G.nodes[node]['fc'] = color_map.get(ntype, '#D3D3D3')
                G.nodes[node]['ec'] = 'black' if is_terminal else 'white'
                G.nodes[node]['lw'] = 2.5 if is_terminal else 0.5
                G.nodes[node]['sz'] = 600 if is_terminal else 400

            # --- 2. 分类型批量绘制 (确保参数全是 List 格式) ---
            for ntype in node_types:
                # 选出属于当前类型的节点
                nodelist = [n for n in G.nodes() if n.split('_')[0] == ntype]
                if not nodelist: continue

                nx.draw_networkx_nodes(
                    G, pos,
                    nodelist=nodelist,
                    node_color=[G.nodes[n]['fc'] for n in nodelist],
                    edgecolors=[G.nodes[n]['ec'] for n in nodelist],  # 关键：这里必须是列表
                    linewidths=[G.nodes[n]['lw'] for n in nodelist],  # 关键：这里必须是列表
                    node_size=[G.nodes[n]['sz'] for n in nodelist],  # 关键：这里必须是列表
                    alpha=1.0
                )

            # 绘制边 - 负红正蓝
            edge_labels = {}
            for u, v, d in G.edges(data=True):
                m_val = d['weight']
                color = 'red' if m_val < 0 else '#3498db'  # 负红，正蓝 (使用深天蓝)
                width = 3.0

                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                                       width=width,
                                       edge_color=color,
                                       alpha=max(0.3, abs(m_val)))
                edge_labels[(u, v)] = f"{m_val:.2f}"
            mask_values = [abs(float(val)) for val in edge_labels.values()]
            print(mask_values)

            # --- [新增] 绘制边标签 (掩码值) ---
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=6,
                font_color='darkgray',  # 建议使用深灰色，避免抢走节点标签的视线
                alpha=0.9,
                label_pos=0.5,  # 标签位于边的中心
                rotate=True  # 标签随边旋转，更易读
            )
            # 标签与图例
            labels_dict = get_labels_dict(G, dataset_name)
            nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=7, font_color="black", alpha=0.7)
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=nt,
                                      markerfacecolor=color_map.get(nt, '#D3D3D3'),
                                      markersize=12, markeredgecolor='white', markeredgewidth=0.5)
                               for nt in node_types if nt in color_map]
            plt.legend(handles=legend_elements, loc='upper right', frameon=True)
            plt.title(f"Task: {task_name} | Dataset: {dataset_name}\nRed: Weakened, Blue: Strengthened")
            plt.axis('off')

            # 9. 保存
            plt.savefig(os.path.join(save_dir, f"{dataset_name}{img_name}_{task_name}.png"), bbox_inches='tight', dpi=300)
            plt.close()


def count_able_weight_path(exres, dataset_name, img_name="", threshold=0.7, save_dir="./outputs/GRAPH_EW"):
    """
    提取关键路径的掩码信息并整理为结构化数据，便于后续导出 Excel。
    """
    import os
    import torch
    import dgl
    import networkx as nx
    from utils import get_mask_delta, get_paths, comp_g_paths_to_paths

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 存储所有路径信息的列表
    all_path_data = []
    # 存储所有稀疏度统计信息的列表
    sparsity_data = []

    adv_pairs = exres.get("adv_pairs", [])
    predictions = exres.get("predictions", [])
    if not adv_pairs:
        print("No adversarial pairs found in exres.")
        return []

    q_src_id, q_src_nt = predictions[0]['src_nid'], exres['src_ntype']
    q_tgt_id, q_tgt_nt = predictions[0]['tgt_nid'], exres['tgt_ntype']

    for pair in adv_pairs:
        idx = pair['idx']
        g = pair['G_M']['graph']

        # 定义任务
        tasks = [
            (f"none_gm_{idx}", None, pair['G_M']['edge_mask'], 0.8, "正向探测"),
            (f"gm_gw_{idx}", pair['G_M']['edge_mask'], pair['G_W']['edge_mask'], 0.4, "双向锚定")
        ]

        for task_name, m1, m2, prune_threshold, stage_name in tasks:
            mask_delta = get_mask_delta(g, mask1=m1, mask2=m2)

            # --------------------------------------------------------
            # 【修改部位 2：核心逻辑纠正】基于 mask_delta 计算真正的干预稀疏度
            # --------------------------------------------------------
            total_edges_count = 0
            above_threshold_count = 0

            # 遍历 delta_mask 中每种边类型的变化量张量
            for etype, delta_tensor in mask_delta.items():
                if delta_tensor is not None:
                    # 累加当前子图边类型的总边数
                    total_edges_count += delta_tensor.numel()
                    # 累加由于因果探测/锚定，导致边权改变量（绝对值）大于指定阈值的边数
                    above_threshold_count += torch.sum(torch.abs(delta_tensor) > threshold).item()

            sparsity_ratio = (above_threshold_count / total_edges_count) if total_edges_count > 0 else 0.0

            # 为了兼容你后续的 Excel 多表多工作表合并脚本，这里的 key 依然保持你原脚本的格式
            sparsity_row = {
                "Dataset": dataset_name,
                "Sample_Idx": img_name,
                "task": task_name,  # 映射为 g_m_idx 或 g_w_idx
                "Total_Edges": total_edges_count,
                "Above_Threshold_Edges": above_threshold_count,
                "Sparsity_Ratio": round(sparsity_ratio, 4)
            }
            sparsity_data.append(sparsity_row)
            # --------------------------------------------------------

            # 构建映射
            edge_mask_lookup = {}
            for etype in g.canonical_etypes:
                u, v = g.edges(etype=etype)
                u_nt, rel, v_nt = etype
                u_global = g.ndata[dgl.NID][u_nt][u]
                v_global = g.ndata[dgl.NID][v_nt][v]
                masks = mask_delta[etype]
                for i in range(len(u)):
                    key = (u_nt, int(u_global[i]), v_nt, int(v_global[i]))
                    edge_mask_lookup.setdefault(key, []).append(masks[i].item())

            comp_path = get_paths(
                src_nid=torch.tensor([q_src_id]),
                src_ntype=q_src_nt,
                tgt_nid=torch.tensor([q_tgt_id]),
                tgt_ntype=q_tgt_nt,
                ghetero=g,
                edge_mask_dict=mask_delta,
                num_paths=7,
            )

            filtered_paths = []
            for path in comp_path:
                if len(path) == 1: continue
                filtered_paths.append(path)
                if len(filtered_paths) >= 3: break

            paths_list = comp_g_paths_to_paths(g, filtered_paths)

            if not paths_list:
                continue

            # 遍历提取出的路径信息
            for p_idx, path in enumerate(paths_list):
                for can_etype, u_id, v_id in path:
                    this_u_nt, rel, this_v_nt = can_etype

                    key1 = (this_u_nt, int(u_id), this_v_nt, int(v_id))
                    key2 = (this_v_nt, int(v_id), this_u_nt, int(u_id))

                    masks = []
                    if key1 in edge_mask_lookup:
                        masks += edge_mask_lookup[key1]
                    if key2 in edge_mask_lookup:
                        masks += edge_mask_lookup[key2]

                    if not masks:
                        m_val = 0.0
                    else:
                        m_val = sum(masks) / len(masks)

                    # 整理成一行字典数据
                    row = {
                        "Dataset": dataset_name,
                        "Sample_Idx": img_name,
                        "task": task_name,
                        "Path_ID": f"{p_idx}",
                        "Src_Node": f"{this_u_nt}_{int(u_id)}",
                        "Relation": rel,
                        "Tgt_Node": f"{this_v_nt}_{int(v_id)}",
                        "Mask_Score": round(m_val, 4),
                        "Abs_Score": round(abs(m_val), 4)
                    }
                    all_path_data.append(row)

    return all_path_data, sparsity_data





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

