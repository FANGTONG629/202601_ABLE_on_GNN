import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from tqdm.auto import tqdm
from collections import defaultdict
from utils import get_ntype_hetero_nids_to_homo_nids, get_homo_nids_to_ntype_hetero_nids, get_ntype_pairs_to_cannonical_etypes
from utils import hetero_src_tgt_khop_in_subgraph, get_neg_path_score_func, k_shortest_paths_with_max_length
import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics import log_loss
from draw_dgl import draw, draw_able_graph
import time
from metrics import eval_fid, eval_SI_avg, eval_spa

import matplotlib.pyplot as plt


def plot_edge_mask_distribution(edge_mask_dict, g, topk_edges=10000):
    """
    edge_mask_dict: dict, key=etype, value=tensor(edge_mask)
    g: DGL 图，用于获取 etype 列表
    topk_edges: 对于大图，只取前 topk_edges 条边做直方图
    """
    for etype in g.canonical_etypes:
        mask = edge_mask_dict[etype].detach().cpu()
        if mask.numel() == 0:
            continue

        # 如果边很多，只取前 topk_edges 条
        mask_plot = mask[:topk_edges].numpy()

        plt.figure(figsize=(5, 3))
        plt.hist(mask_plot, bins=20, color='skyblue', edgecolor='k')
        plt.title(f"Edge Mask Distribution: {etype}")
        plt.xlabel("Mask Value")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.show()



def get_edge_mask_dict(ghetero):
    device = ghetero.device
    edge_mask_dict = {}
    for etype in ghetero.canonical_etypes:
        num_edges = ghetero.num_edges(etype)
        num_nodes = ghetero.edge_type_subgraph([etype]).num_nodes()

        std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * num_nodes))
        edge_mask_dict[etype] = torch.nn.Parameter(torch.randn(num_edges, device=device) * std)

    return edge_mask_dict

def normalize_m(edge_mask_dict):
    """
    对所有 etype 的边掩码做【全局线性映射到 0~1】
    只有当存在值 > 1 或 < 0 时才触发

    参数
    ----
    edge_mask_dict : dict[etype] -> Tensor(num_edges,)

    返回
    ----
    new_dict : 归一化后的新字典（不改原 Tensor）
    """
    # ===== 收集所有掩码拼接成一个大向量 =====
    all_vals = []
    for mask in edge_mask_dict.values():
        if mask.numel() > 0:
            all_vals.append(mask.view(-1))
    if len(all_vals) == 0:
        return edge_mask_dict  # 空图直接返回

    all_vals = torch.cat(all_vals)

    global_min = all_vals.min()
    global_max = all_vals.max()

    # ===== 判断是否需要归一化 =====
    if global_max <= 1.0 and global_min >= 0.0:
        return edge_mask_dict  # 已经在 0~1，无需处理

    # ===== 线性映射到 0~1 =====
    denom = (global_max - global_min).clamp(min=1e-8)

    new_dict = {}
    for etype, mask in edge_mask_dict.items():
        if mask.numel() == 0:
            new_dict[etype] = mask
        else:
            new_dict[etype] = (mask - global_min) / denom

    return new_dict


class ABLEg(nn.Module):
    def __init__(self,
                 model,
                 dataset_name='lastfm',
                 lr=0.1,
                 num_epochs=25,
                 log=False,
                 lambda_1=0.01,
                 lambda_2=1,
                 lambda_m=0.5):
        super(ABLEg, self).__init__()
        self.model = model
        self.src_ntype = model.src_ntype
        self.tgt_ntype = model.tgt_ntype
        self.lr = lr
        self.num_epochs = num_epochs
        self.log = log
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_m = lambda_m
        self.dataset_name = dataset_name


    def _init_masks(self, ghetero):
        """初始化掩码，包含原始边和候选边"""
        return get_edge_mask_dict(ghetero);

    def generate_neiborhood(self,
                            src_nid,
                            tgt_nid,
                            feat_nids,
                            ghetero,
                            radius=0.5,
                            n_samples=10,
                            random_seed=42
                            ):
        rng = random.Random(random_seed)

        neighborhoods = [{
            "g": ghetero,
            "src_nid": src_nid,
            "tgt_nid": tgt_nid,
            "feat_nids": feat_nids
        }]

        # ====== 找扰动边 canonical etype ======
        # 需要扰动的边是所有类型的边除去目标边，下为目标边
        # lastfm: user-likes-artist
        # ACM: paper-pf-field
        # aug_citation: author-likes-paper
        etype_counts = {
            etype: ghetero.num_edges(etype)
            for etype in ghetero.canonical_etypes
            if etype[0] != self.src_ntype or etype[2] != self.tgt_ntype
        }
        total_aux_edges = sum(etype_counts.values())
        if total_aux_edges == 0:
            return neighborhoods

        # ====== 生成扰动子图 ======
        for _ in range(n_samples - 1):
            g_pert = ghetero.clone()  # 深拷贝子图

            for etype, num_edges in etype_counts.items():
                num_perturb = max(1, int(num_edges * radius))
                # ---- 随机加边 ----
                add_k = rng.randint(1, num_perturb)
                if add_k > 0:
                    stype, _, dtype = etype
                    num_src = g_pert.num_nodes(stype)
                    num_dst = g_pert.num_nodes(dtype)
                    if num_src > 0 and num_dst > 0: # 只有当源节点和目标节点在子图中都存在时，才进行加边
                        add_src = torch.randint(0, num_src, (add_k,), device=ghetero.device, dtype=torch.long)
                        add_dst = torch.randint(0, num_dst, (add_k,), device=ghetero.device, dtype=torch.long)
                        g_pert = dgl.add_edges(g_pert, add_src, add_dst, etype=etype)
                    else: # 如果子图中没有这类节点，无法加边
                        add_k = 0

                # ---- 随机删边 ----
                del_k = num_perturb - add_k
                if del_k > 0:
                    del_eids = rng.sample(range(num_edges), k=min(del_k, num_edges))
                    del_eids = torch.tensor(del_eids, device=ghetero.device, dtype=torch.long)
                    g_pert = dgl.remove_edges(g_pert, del_eids, etype=etype)

            neighborhoods.append({
                "g": g_pert,
                "src_nid": src_nid,
                "tgt_nid": tgt_nid,
                "feat_nids": feat_nids
            })

        return neighborhoods

    def generate_pair(self,
                      src_nid,
                      tgt_nid,
                      feat_nids,
                      ghetero,
                      y
                      ):
        device = ghetero.device

        # ================== 第一阶段：优化 M_A ==================
        # 初始化边掩码 M_A
        m_edge_mask_dict = self._init_masks(ghetero)
        optimizer = torch.optim.Adam(list(m_edge_mask_dict.values()), lr=self.lr)

        y_tensor = torch.tensor([y], dtype=torch.float32, device=device)

        g_m_loss = 0.0

        for epoch in range(self.num_epochs):
            optimizer.zero_grad()

            # 创建掩码后的边权重字典
            M_A = {
                etype: torch.sigmoid((m_edge_mask_dict[etype].sigmoid() - 0.5) * 12)
                for etype in m_edge_mask_dict
            }

            eweight_M = {
                etype: 1.0 - M_A[etype]
                for etype in M_A
            }

            # 使用原始黑箱模型进行预测，传入掩码后的边权重
            score = self.model(
                src_nid,
                tgt_nid,
                ghetero,  # 新图
                feat_nids,
                eweight_dict=eweight_M  # 关键：传入掩码后的边权重
            )

            # 计算损失函数
            pred_loss = F.binary_cross_entropy_with_logits(score, 1 - y_tensor) # 预测损失：F(G_M) 与真实标签 Y 相反
            reg_loss = 0# 正则化损失：M_A 的绝对值最小化
            for etype in ghetero.canonical_etypes:
                mask = m_edge_mask_dict[etype]
                if mask.numel() > 0:
                    #reg_loss += torch.abs(mask.mean())
                    reg_loss += torch.mean(torch.abs(mask))

            total_loss = pred_loss + self.lambda_1 * reg_loss # 联合优化

            g_m_loss = total_loss

            total_loss.backward()
            optimizer.step()

            # if self.log and epoch % 5 == 0:
            #     print(f"Epoch {epoch}: pred_loss={pred_loss.item():.4f}, "
            #           f"reg_loss={reg_loss.item():.4f}, total_loss={total_loss.item():.4f}")

        # 获取最终的掩码结果
        M_A_detached = {  # 在接下来的优化中，认为M_A的值是常数
            etype: (1.0 - torch.sigmoid((m_edge_mask_dict[etype].sigmoid() - 0.5) * 12)).detach()
            for etype in ghetero.canonical_etypes
        }

        # ====== 显式 G_M ======
        G_M = {
            "graph": ghetero,
            "edge_mask": {
                etype: (1.0 - torch.sigmoid((m_edge_mask_dict[etype].sigmoid() - 0.5) * 12)).clone().detach()
                for etype in ghetero.canonical_etypes
            },
            "loss": g_m_loss
        }

        # ================== 第二阶段：固定 A_M，优化 W_A ==================
        # 初始化边掩码 W_A
        w_edge_mask_dict = self._init_masks(ghetero)
        optimizer_w = torch.optim.Adam(list(w_edge_mask_dict.values()), lr=self.lr)

        g_w_loss = 0.0

        for epoch in range(self.num_epochs):
            optimizer_w.zero_grad()

            W_A = {
                etype: torch.sigmoid((w_edge_mask_dict[etype].sigmoid() - 0.5) * 12)
                for etype in ghetero.canonical_etypes
            }

            eweight_W = {
                etype: M_A_detached[etype] * W_A[etype] + self.lambda_m * W_A[etype]
                for etype in ghetero.canonical_etypes
            }

            score_w = self.model(
                src_nid,
                tgt_nid,
                ghetero,
                feat_nids,
                eweight_dict=eweight_W
            )

            # 预测应与 Y 相似
            pred_loss_w = F.binary_cross_entropy_with_logits(score_w, y_tensor)

            # W_A 的正则
            reg_loss_w = 0
            for etype in ghetero.canonical_etypes:
                mask = w_edge_mask_dict[etype]
                if mask.numel() > 0:
                    #reg_loss_w += torch.abs(mask.mean())
                    reg_loss_w += torch.mean(torch.abs(mask))

            total_loss_w = pred_loss_w + self.lambda_2 * reg_loss_w

            g_w_loss = total_loss_w

            total_loss_w.backward()
            optimizer_w.step()

            # if self.log and epoch % 5 == 0:
            #     print(f"Epoch {epoch}: pred_loss={pred_loss_w.item():.4f}, "
            #           f"reg_loss={reg_loss_w.item():.4f}, total_loss={total_loss_w.item():.4f}")

        # ===== 第二阶段得到的最终边权（可能 >1）=====
        final_eweight_W = {
            etype: M_A_detached[etype] * torch.sigmoid((w_edge_mask_dict[etype].sigmoid() - 0.5) * 12)
                   + self.lambda_m * torch.sigmoid((w_edge_mask_dict[etype].sigmoid() - 0.5) * 12)
            for etype in ghetero.canonical_etypes
        }

        final_eweight_W = normalize_m(final_eweight_W)

        # ====== 显式 G_W ======
        G_W = {
            "graph": ghetero,
            "edge_mask": {
                etype: final_eweight_W[etype].clone().detach()
                for etype in ghetero.canonical_etypes
            },
            "loss":g_w_loss
        }

        return G_M, G_W

    def explain(self, src_nid, tgt_nid, ghetero, radius=0.5, n_samples=50,
                random_seed=42, num_hops=2):
        print("="*80)
        print(f"[ABLE-g] Explain link: ({self.src_ntype}, {int(src_nid)}) -> ({self.tgt_ntype}, {int(tgt_nid)})")

        # ------提取 k-hop 子图------
        sg_src_nid, sg_tgt_nid, sg, sg_feat_nids = hetero_src_tgt_khop_in_subgraph(
            self.src_ntype,
            src_nid,
            self.tgt_ntype,
            tgt_nid,
            ghetero,
            num_hops
        )

        print(sg)

        # ------生成邻居------
        neighborhoods = self.generate_neiborhood(
            src_nid=sg_src_nid,
            tgt_nid=sg_tgt_nid,
            ghetero=sg,
            feat_nids=sg_feat_nids,
            radius=radius,
            n_samples=n_samples,
            random_seed=random_seed
        )

        print(f"[ABLE-g] Generated {len(neighborhoods)} neighborhood subgraphs")

        results = []
        self.model.eval()

        with torch.no_grad():
            for i, subgraph_sample in enumerate(neighborhoods):
                g_sub = subgraph_sample["g"]
                src_sub = subgraph_sample["src_nid"]
                tgt_sub = subgraph_sample["tgt_nid"]
                feat_sub = subgraph_sample["feat_nids"]

                score = self.model(
                    src_sub,
                    tgt_sub,
                    g_sub,
                    feat_sub
                )

                pred = (score > 0).int().item()

                results.append({
                    "idx": i,
                    "score": score.item(),
                    "pred": pred,
                    "g": g_sub,
                    "src_nid": src_sub,
                    "tgt_nid": tgt_sub,
                    "feat_nids": feat_sub
                })


        # ------生成对抗样本对------
        # 有梯度
        adv_pairs = []  # 保存 (G_M, G_W)
        for res in tqdm(results, desc="Generating adversarial pairs"):
            g_sub = res["g"]
            src_sub = res["src_nid"]
            tgt_sub = res["tgt_nid"]
            feat_sub = res["feat_nids"]
            y = res["pred"]  # ⚠️ 用该 neighborhood 自己的预测作为 y

            G_M, G_W = self.generate_pair(
                src_nid=src_sub,
                tgt_nid=tgt_sub,
                feat_nids=feat_sub,
                ghetero=g_sub,
                y=y
            )

            # ================== Adversarial Pair Debug ==================
            with torch.no_grad():
                # -------- Original --------
                score_orig = self.model(
                    src_sub,
                    tgt_sub,
                    g_sub,
                    feat_sub
                )
                logit_orig = score_orig.item()
                pred_orig = int(logit_orig > 0)

                # -------- G_M (should be flipped) --------
                g_m = G_M["graph"]
                eweight_m = G_M["edge_mask"]

                score_m = self.model(
                    src_sub,
                    tgt_sub,
                    g_m,
                    feat_sub,
                    eweight_dict=eweight_m
                )
                logit_m = score_m.item()
                pred_m = int(logit_m > 0)

                # -------- G_W (should return) --------
                g_w = G_W["graph"]
                eweight_w = G_W["edge_mask"]

                score_w = self.model(
                    src_sub,
                    tgt_sub,
                    g_w,
                    feat_sub,
                    eweight_dict=eweight_w
                )
                logit_w = score_w.item()
                pred_w = int(logit_w > 0)

            # -------- Print --------
            # print("[Adversarial Pair Check]")
            # print(f"  Original: logit={logit_orig:+.4f}, pred={pred_orig}")
            #
            # flip_flag = "✅ flipped" if pred_m != pred_orig else "❌ not flipped"
            # print(f"  G_M     : logit={logit_m:+.4f}, pred={pred_m}  {flip_flag}")
            #
            # print(f"  G_W     : logit={logit_w:+.4f}, pred={pred_w}")

            adv_pairs.append({
                "idx": res["idx"],
                "G_M": G_M,
                "G_W": G_W,
                "y": y,
                "pred_orig": pred_orig,
                "pred_m": pred_m,
                "pred_w": pred_w
            })


        return {
            "src_ntype": self.src_ntype,
            "tgt_ntype": self.tgt_ntype,
            "predictions": results,
            "adv_pairs": adv_pairs
        }

    def get_lambda(self, idx=1):
        if idx == 2: return self.lambda_2
        else: return self.lambda_1