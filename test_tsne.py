import dgl
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
from datetime import datetime



def get_graph_level_embedding(model, g, feat_nids, eweight_dict=None, is_nb=False):
    """
    将异构图压缩为全图向量（一图一点）
    :param g: DGL 异构图
    :param node_embeds_dict: 字典 {ntype: tensor}
    :param edge_weights_dict: 字典 {canonical_etype: tensor} 即可解释性掩码 eweight
    """
    model.eval()
    device = next(model.parameters()).device  # 动态获取模型所在设备

    # 强制开启局部作用域，防止图数据污染
    with g.local_scope():
        # 确保 eweight_dict 格式正确且已对齐
        if eweight_dict is not None:
            for etype, w in eweight_dict.items():
                # 检查维度：必须等于子图的边数
                if w.shape[0] != g.num_edges(etype):
                    print(f"Warning: Etype {etype} weight mismatch! Using default.")
                    #continue
                g.edges[etype].data['_edge_weight'] = w.detach().to(next(model.parameters()).device)
        elif eweight_dict is None and is_nb is False:
            print(f"Warning: eweight_dict is None! Using default.")
        with torch.no_grad():
            torch.cuda.synchronize()
            h_dict = model.encoder(g, feat_nids, eweight_dict)
            # 全局池化（readout），如果eweight_dict不为None则已经是加权状态
            graph_embeds = []
            for ntype, h in h_dict.items():
                #h = F.normalize(h, p=2, dim=1)
                g.nodes[ntype].data['h'] = h
                # 使用 DGL 的内置池化函数,此时的 h 已经是经过 encoder“加权加工”后的特征了
                g_h = dgl.mean_nodes(g, 'h', ntype=ntype)
                graph_embeds.append(g_h)
            # 拼接
            final_graph_vector = torch.cat(graph_embeds, dim=1)
    return final_graph_vector


def visualize_neighborhood_tsne(model, exres, title="Decision Boundary Exploration"):
    """
    可视化邻域子图分布
    :param results: 包含多个子图预测结果的列表 [cite: 343]
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tsne_config = {
        'n_components': 3,
        'perplexity': 20,
        'learning_rate' : "auto",
        'n_iter' : 3000,
        'init' : 'pca',
        'random_state' : 42
        }
    tsne_config['save_dir'] = (
        f"./outputs/GRAPHS/tsne_p{tsne_config['perplexity']}"
        f"_lr{tsne_config['learning_rate']}"
        f"_iter{tsne_config['n_iter']}"
        f"_init{tsne_config['init']}"
        f"_{timestamp}.png"
    )
    adv_pairs = exres.get("adv_pairs", [])
    predictions = exres.get("predictions", [])

    all_embeddings = []
    point_types = []  # 记录是哪种点：NEIGH, GM, GW
    point_labels = []  # 记录预测值：0 或 1
    point_idxs = []  # 记录原始索引，用于找“第一个邻居”

    print(f"正在准备数据：{len(predictions)}个邻居，{len(adv_pairs)}对对抗样本")

    # 1. 提取邻居子图 Embedding (NEIGH)
    for res in predictions:
        h = get_graph_level_embedding(model, res['g'], res['feat_nids'], None, True)
        all_embeddings.append(h)
        point_types.append("NEIGH")
        point_labels.append(int(res['pred']))
        point_idxs.append(res['idx'])

    for pair, pred_info in zip(adv_pairs, predictions):
        current_feat_nids = pred_info['feat_nids']
        # 1.G_M
        gm_h = get_graph_level_embedding(model, pair['G_M']['graph'], current_feat_nids, pair['G_M']['edge_mask'])
        all_embeddings.append(gm_h)
        point_types.append("GM")
        point_labels.append(f"G_M (Pred: {pair['pred_m']})")
        point_idxs.append(-1)  # 对抗点不参与“第一个”判定
        # 2.G_W
        gw_h = get_graph_level_embedding(model, pair['G_W']['graph'], current_feat_nids, pair['G_W']['edge_mask'])
        all_embeddings.append(gw_h)
        point_types.append("GW")
        point_labels.append(f"G_W (Pred: {pair['pred_w']})")
        point_idxs.append(-1)  # 对抗点不参与“第一个”判定

    # --- t-SNE 降维 ---
    all_embeddings = torch.cat(all_embeddings, dim=0).detach().cpu().numpy()
    print(np.std(all_embeddings, axis=0).mean())
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_embeddings) - 1),
                random_state=42, init='pca', learning_rate='auto')
    vis_2d = tsne.fit_transform(all_embeddings)

    # --- 分组绘图 ---
    plt.figure(figsize=(12, 9))

    for i in range(len(vis_2d)):
        p_type = point_types[i]
        p_label = point_labels[i]
        p_idx = point_idxs[i]

        x, y = vis_2d[i, 0], vis_2d[i, 1]

        # A. 处理邻居子图 (空心正方形 's')
        if p_type == "NEIGH":
            color = 'black' if p_label == 1 else 'gray'
            size = 180 if p_idx == 0 else 80  # 第一个邻居加粗加大
            edge_w = 3 if p_idx == 0 else 1.5
            plt.scatter(x, y, marker='s', s=size, edgecolors=color,
                        facecolors='none', linewidths=edge_w,
                        label="Neighbor" if i == 0 else "")  # 避免重复图例

        # B. 处理对抗对 G_M (x型)
        elif p_type == "GM":
            color = 'darkblue' if p_label == 'G_M (Pred: 1)' else 'lightblue'
            plt.scatter(x, y, marker='x', s=100, c=color, alpha=0.8)

        # C. 处理对抗对 G_W (o型)
        elif p_type == "GW":
            color = 'darkred' if p_label == 'G_W (Pred: 1)' else 'mistyrose'
            plt.scatter(x, y, marker='o', s=100, c=color, alpha=0.8)

    # 手动构建图例，防止重复
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='black', label='Neighbor (Pred 1)', markerfacecolor='none', markersize=10,
               linestyle='None'),
        Line2D([0], [0], marker='s', color='gray', label='Neighbor (Pred 0)', markerfacecolor='none', markersize=10,
               linestyle='None'),
        Line2D([0], [0], marker='x', color='darkblue', label='G_M (Pred 1)', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='x', color='lightblue', label='G_M (Pred 0)', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='o', color='darkred', label='G_W (Pred 1)', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='o', color='mistyrose', label='G_W (Pred 0)', markersize=10, linestyle='None'),
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f"t-SNE Visualization: {title}")
    plt.grid(True, alpha=0.3)
    plt.savefig(tsne_config['save_dir'],bbox_inches='tight')

    # 未画邻域样本点
    # all_graph_h = []
    # labels = []
    #
    # print("正在提取全图 Embedding...")
    # # 遍历每一个对抗样本对，将 G_M 和 G_W 转化为两个独立的点
    # # 使用 zip 同步遍历，确保顺序对应
    # # pair 包含图结构 G_M/G_W 和预测结果
    # # pred_info 包含该样本采样时的特征 feat_nids
    # for pair, pred_info in zip(adv_pairs, predictions):
    #     # 核心修改：从对应的 predictions 记录中提取 feat_nids
    #     current_feat_nids = pred_info['feat_nids']
    #
    #     # 1. 处理 G_M (反事实/对抗样本)
    #     gm_h = get_graph_level_embedding(model, pair['G_M']['graph'], current_feat_nids, pair['G_M']['edge_mask'])
    #     all_graph_h.append(gm_h)
    #     labels.append(f"G_M (Pred: {pair['pred_m']})")
    #
    #     # 2. 处理 G_W (锚点/边界样本)
    #     gw_h = get_graph_level_embedding(model, pair['G_W']['graph'], current_feat_nids, pair['G_W']['edge_mask'])
    #     all_graph_h.append(gw_h)
    #     labels.append(f"G_W (Pred: {pair['pred_w']})")
    #
    # # 检查是否有数据
    # if not all_graph_h:
    #     print("Error: 没有提取到任何 Embedding 数据。")
    #     return
    #
    #
    # # 将 Tensor 列表转换为 numpy 矩阵，点数为 2 * len(adv_pairs)
    # all_graph_h = torch.cat(all_graph_h, dim=0).detach().cpu().numpy()
    # labels = np.array(labels)
    #
    # # t-SNE 降维：perplexity 需小于样本总数
    # n_samples = all_graph_h.shape[0]
    # tsne = TSNE(n_components=tsne_config['n_components'],
    #             perplexity=min(tsne_config['perplexity'], n_samples - 1),
    #             learning_rate=tsne_config['learning_rate'],
    #             n_iter=tsne_config['n_iter'],
    #             init=tsne_config['init'],
    #             random_state=42)
    # vis_2d = tsne.fit_transform(all_graph_h)
    #
    # # 绘图：通过颜色和形状区分对抗样本 (G_M) 与锚点样本 (G_W)
    # plt.figure(figsize=(12, 9))
    # # sns.scatterplot(
    # #     x=vis_2d[:, 0], y=vis_2d[:, 1],
    # #     hue=labels, style=labels,
    # #     s=120, palette='Set1'
    # # )
    # # 定义映射规则：(类别标签) -> (颜色, 形状, 标签文本)
    # mapping = {
    #     "G_M (Pred: 1)": ("darkblue", "x", "G_M (Label 1)"),
    #     "G_M (Pred: 0)": ("lightblue", "x", "G_M (Label 0)"),
    #     "G_W (Pred: 1)": ("darkred", "o", "G_W (Label 1)"),
    #     "G_W (Pred: 0)": ("mistyrose", "o", "G_W (Label 0)")  # salmon 为浅红
    # }
    #
    # for key, (color, marker, label_text) in mapping.items():
    #     indices = [i for i, l in enumerate(labels) if l == key]
    #     if indices:
    #         plt.scatter(vis_2d[indices, 0], vis_2d[indices, 1],
    #                     c=color, marker=marker, label=label_text, s=100, alpha=0.8)
    #
    #
    #
    # plt.title(f'Graph-Level t-SNE for Adversarial Pairs: {title}')
    # plt.xlabel("t-SNE Dimension 1")
    # plt.ylabel("t-SNE Dimension 2")
    # #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.legend(loc='best')
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.tight_layout()
    # plt.savefig(tsne_config['save_dir'],bbox_inches='tight')
    #plt.show()