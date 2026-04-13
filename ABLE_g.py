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
from draw_dgl import draw
import time
from metrics import eval_fid, eval_SI_avg, eval_spa


def get_edge_mask_dict(ghetero):
    device = ghetero.device
    edge_mask_dict = {}
    for etype in ghetero.canonical_etypes:
        num_edges = ghetero.num_edges(etype)
        num_nodes = ghetero.edge_type_subgraph([etype]).num_nodes()

        std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / (2 * num_nodes))
        edge_mask_dict[etype] = torch.nn.Parameter(torch.randn(num_edges, device=device) * std)

    return edge_mask_dict

def get_feat_mask_dict(feat_dict):
    """
    Feature-wise mask:
    对每个节点类型，生成一个长度为 feat_dim 的可训练向量
    表示每个特征在所有节点上的生效程度（全局广播）
    返回 dict[ntype] -> nn.Parameter([feat_dim])
    """
    device = next(iter(feat_dict.values())).device
    feat_mask_dict = {}

    for ntype, X in feat_dict.items():
        feat_dim = X.shape[1]  # embedding 的维度 D
        # Xavier 初始化
        std = torch.nn.init.calculate_gain('relu') * np.sqrt(2.0 / feat_dim)
        feat_mask_dict[ntype] = nn.Parameter(
            torch.randn(feat_dim, device=device) * std
        )

    return feat_mask_dict




class ABLEg(nn.Module):
    def __init__(self,
                 model,
                 lr=0.01,
                 num_epochs=50,
                 log=False):
        super(ABLEg, self).__init__()
        self.model = model
        self.src_ntype = model.src_ntype
        self.tgt_ntype = model.tgt_ntype
        self.lr = lr
        self.num_epochs = num_epochs
        self.log = log

    def _init_masks(self, ghetero, feat_dict):
        return get_edge_mask_dict(ghetero), get_feat_mask_dict(feat_dict)

    def generate_neiborhood(self,
                            src_nid,
                            tgt_nid,
                            feat_nids,
                            ghetero,
                            radius=0.5,
                            n_samples=50,
                            random_seed=42,
                            num_hops=2):
        rng = random.Random(random_seed)

        ntype_hetero_nids_to_homo_nids = get_ntype_hetero_nids_to_homo_nids(ghetero)
        homo_src_nid = ntype_hetero_nids_to_homo_nids[(self.src_ntype, int(src_nid))]
        homo_tgt_nid = ntype_hetero_nids_to_homo_nids[(self.tgt_ntype, int(tgt_nid))]


        neighborhoods = [{
            "g": ghetero,
            "src_nid": src_nid,
            "tgt_nid": tgt_nid,
            "feat_nids": feat_nids
        }]

        # ====== 找 user-user canonical etype ======
        ntype_pairs_to_cannonical_etypes = get_ntype_pairs_to_cannonical_etypes(ghetero)
        if ('user', 'user') not in ntype_pairs_to_cannonical_etypes:
            return neighborhoods

        uu_etype = ntype_pairs_to_cannonical_etypes[('user', 'user')]
        uu_src, uu_dst = ghetero.edges(etype=uu_etype)
        num_uu_edges = uu_src.shape[0]
        if num_uu_edges == 0:
            return neighborhoods

        num_perturb = max(1, int(radius * num_uu_edges))

        # ====== 生成扰动子图 ======
        for _ in range(n_samples - 1):
            g_pert = ghetero.clone()  # 深拷贝子图

            # ---- 随机删边 ----
            del_k = rng.randint(0, num_perturb)
            if del_k > 0:
                del_eids = rng.sample(range(num_uu_edges), k=min(del_k, num_uu_edges))
                del_eids = torch.tensor(del_eids, device=ghetero.device)
                g_pert = dgl.remove_edges(g_pert, del_eids, etype=uu_etype)

            # ---- 随机加边 ----
            add_k = num_perturb - del_k
            if add_k > 0:
                user_nids = g_pert.nodes('user')
                num_users = user_nids.shape[0]
                add_src = torch.randint(0, num_users, (add_k,), device=ghetero.device)
                add_dst = torch.randint(0, num_users, (add_k,), device=ghetero.device)
                g_pert = dgl.add_edges(g_pert, add_src, add_dst, etype=uu_etype)

            neighborhoods.append({
                "g": g_pert,
                "src_nid": src_nid,
                "tgt_nid": tgt_nid,
                "feat_nids": feat_nids
            })

        return neighborhoods

    def generate_pair(self, src_nid, tgt_nid, feat_nids, ghetero, y):
        device = ghetero.device

        # --- 子图 embedding 对齐 ---
        # feat_dict: dict[ntype] -> FloatTensor [N,D] (子图节点 embedding)
        for ntype in feat_nids:
            idx = feat_nids[ntype]
            print(f"[DEBUG] ntype={ntype}, dtype={idx.dtype}, shape={idx.shape}, first10={idx[:10]}")
            feat_nids[ntype] = idx.long()  # 强制 LongTensor

        feat_dict = {
            ntype: self.model.encoder.emb.weight[ntype][feat_nids[ntype]]
            for ntype in ghetero.ntypes
        }

        # ====== 掩码1 ====== (M_X, M_A)
        m_edge_mask_dict, m_feat_mask_dict = self._init_masks(ghetero, feat_dict)
        # 全局 feature-wise mask: shape [D]，会广播到所有节点
        params = list(m_edge_mask_dict.values()) + list(m_feat_mask_dict.values())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        y_tensor = torch.tensor([y], dtype=torch.float32, device=device)
        self.model.eval()
        sigmoid = torch.nn.Sigmoid()

        for step in range(self.num_epochs):
            optimizer.zero_grad()

            # --- 特征 mask 广播 ---
            X_M = {
                ntype: feat_dict[ntype] * sigmoid(m_feat_mask_dict[ntype]).unsqueeze(0)  # [1,D] -> 广播到 [N,D]
                for ntype in feat_dict
            }
            A_M = {
                etype: sigmoid(m_edge_mask_dict[etype])
                for etype in ghetero.canonical_etypes
            }

            # 前向
            score = self.model(src_nid, tgt_nid, ghetero, X_M, A_M)
            loss = F.binary_cross_entropy_with_logits(score, 1 - y_tensor)  # 翻转预测
            loss.backward()
            optimizer.step()

        G_M = {"feat_dict": X_M, "eweight_dict": A_M}

        # ====== 掩码2 ====== (W_X, W_A)
        w_edge_mask_dict, w_feat_mask_dict = self._init_masks(ghetero, feat_dict)
        params2 = list(w_edge_mask_dict.values()) + list(w_feat_mask_dict.values())
        optimizer2 = torch.optim.Adam(params2, lr=self.lr)

        for step in range(self.num_epochs):
            optimizer2.zero_grad()

            # G_W
            X_W = {
                ntype: X_M[ntype] * sigmoid(w_feat_mask_dict[ntype]).unsqueeze(0)  # [1,D] -> 广播
                for ntype in X_M
            }
            A_W = {
                etype: A_M[etype] * sigmoid(w_edge_mask_dict[etype])
                for etype in A_M
            }

            score2 = self.model(src_nid, tgt_nid, ghetero, X_W, A_W)
            loss2 = F.binary_cross_entropy_with_logits(score2, y_tensor)  # 保持预测
            loss2.backward()
            optimizer2.step()

        G_W = {"feat_dict": X_W, "eweight_dict": A_W}

        return G_M, G_W



    def explain(self, src_nid, tgt_nid, ghetero, radius=0.5, n_samples=50,
                random_seed=42, num_hops=2):
        print("="*80)
        print(f"[ABLE-g] Explain link: ({self.src_ntype}, {int(src_nid)}) -> ({self.tgt_ntype}, {int(tgt_nid)})")

        # ====== 提取 k-hop 子图 ======
        sg_src_nid, sg_tgt_nid, sg, sg_feat_nids = hetero_src_tgt_khop_in_subgraph(
            self.src_ntype,
            src_nid,
            self.tgt_ntype,
            tgt_nid,
            ghetero,
            num_hops
        )

        neighborhoods = self.generate_neiborhood(
            src_nid=sg_src_nid,
            tgt_nid=sg_tgt_nid,
            ghetero=sg,
            feat_nids=sg_feat_nids,
            radius=radius,
            n_samples=n_samples,
            random_seed=random_seed,
            num_hops=num_hops
        )

        print(f"[ABLE-g] Generated {len(neighborhoods)} neighborhood subgraphs")

        print("\n[Neighborhoods Overview]")
        for i, nb in enumerate(neighborhoods):
            g = nb["g"]
            print(f"\n--- Neighborhood {i} ---")
            print(f"src_nid: {nb['src_nid']}, tgt_nid: {nb['tgt_nid']}")

            # 节点信息
            for ntype in g.ntypes:
                print(f"  #Nodes[{ntype}]: {g.num_nodes(ntype)}")

            # 边信息
            for etype in g.canonical_etypes:
                print(f"  #Edges{etype}: {g.num_edges(etype)}")

        if len(neighborhoods) > 2:
            print("\n" + "=" * 80)
            print("[Detailed View] Neighborhood 2 (3rd one)")

            nb = neighborhoods[2]
            g = nb["g"]

            print(f"src_nid: {nb['src_nid']}, tgt_nid: {nb['tgt_nid']}")

            # ---- 节点列表 ----
            for ntype in g.ntypes:
                nids = g.nodes(ntype)
                print(f"\n[{ntype}]")
                print(f"  num_nodes = {len(nids)}")
                print(f"  node_ids (first 10): {nids[:10].tolist()}")

            # ---- 边列表（只打印前 10 条，防止刷屏）----
            for etype in g.canonical_etypes:
                src, dst = g.edges(etype=etype)
                print(f"\nEdges {etype}:")
                print(f"  num_edges = {src.shape[0]}")
                if src.shape[0] > 0:
                    for i in range(min(10, src.shape[0])):
                        print(f"    {int(src[i])} -> {int(dst[i])}")

        print("\n[Center Node Check]")
        print("src in graph:", nb["src_nid"] < g.num_nodes(self.src_ntype))
        print("tgt in graph:", nb["tgt_nid"] < g.num_nodes(self.tgt_ntype))

        # ------生成邻居------
        # 无梯度
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

            adv_pairs.append({
                "idx": res["idx"],
                "G_M": G_M,
                "G_W": G_W,
                "y": y
            })

        return {
            "predictions": results,
            "adv_pairs": adv_pairs
        }


