import os
import torch
import argparse
import pickle
import random
from tqdm.auto import tqdm
from pathlib import Path

from metrics import eval_SI_avg
from utils import set_seed, print_args, set_config_args, hetero_src_tgt_khop_in_subgraph
from data_processing import load_dataset
from model import HeteroRGCN, HeteroLinkPredictionModel
from draw_dgl import draw_able_graph_on_ax
import matplotlib.pyplot as plt
import numpy as np
import time
import statistics
from ABLE_g import ABLEg
from utils import evaluate_random_runs_ex
from test_tsne import visualize_neighborhood_tsne



parser = argparse.ArgumentParser(description='Explain link predictor')
parser.add_argument('--device_id', type=int, default=-1)

'''
Dataset args
'''
parser.add_argument('--dataset_dir', type=str, default='datasets')  # 数据集目录
parser.add_argument('--dataset_name', type=str, default='lastfm')  # 数据集名称 lastfm,aug_citation,synthetic
parser.add_argument('--valid_ratio', type=float, default=0.1)  # 验证集比例
parser.add_argument('--test_ratio', type=float, default=0.2)  # 测试集比例

'''
GNN args
'''
parser.add_argument('--emb_dim', type=int, default=128)  # 嵌入维度
parser.add_argument('--hidden_dim', type=int, default=128)  # 隐藏层维度
parser.add_argument('--out_dim', type=int, default=128)  # 输出层维度
parser.add_argument('--saved_model_dir', type=str, default='saved_models')  # 保存模型的目录
parser.add_argument('--saved_model_name', type=str, default='')  # 保存模型的名称

'''
Link predictor args
'''
parser.add_argument('--src_ntype', type=str, default='user', help='prediction source node type')  # 源节点类型
parser.add_argument('--tgt_ntype', type=str, default='item', help='prediction target node type')  # 目标节点类型
parser.add_argument('--pred_etype', type=str, default='likes', help='prediction edge type')  # 预测边类型
parser.add_argument('--link_pred_op', type=str, default='dot', choices=['dot', 'cos', 'ele', 'cat'],
                    help='operation passed to dgl.EdgePredictor')  # 链路预测操作

'''
Explanation args
'''
parser.add_argument('--num_explain', type=int, default=5,
                    help='number of test samples to explain') #挑选多少个样本进行解释
parser.add_argument('--num_neighbor', type=int, default=10,
                    help='how much neighbor a sample generates') #一个样本生成多少个邻居
parser.add_argument('--radius', type=float, default=0.5, help='neighborhood disturbance range')  # 对邻居点的干扰半径
parser.add_argument('--lr', type=float, default=0.1, help='explainer learning_rate')  # 解释器的学习率
parser.add_argument('--lambda_1', type=float, default=0.01, help='first-stage regularization weight')  # 第一阶段正则化权重
parser.add_argument('--lambda_2', type=float, default=0.01, help='second-stage regularization weight')  # 第二阶段正则化权重
parser.add_argument('--lambda_m', type=float, default=0.5, help='Second-stage residual scaling factor')  # 第二阶段残差缩放系数
parser.add_argument('--num_hops', type=int, default=2, help='computation graph number of hops')  # 计算图的跳数，默认为2
parser.add_argument('--num_epochs', type=int, default=25, help='How many epochs to learn the mask')  # 学习掩码的训练轮数
parser.add_argument('--num_runs', type=int, default=4, help='How many tests to run')  # 测试轮数
parser.add_argument('--prune_max_degree', type=int, default=200,
                    help='prune the graph such that all nodes have degree smaller than max_degree. No prune if -1')  # 剪枝最大度数，200
parser.add_argument('--save_excel', default=False, action='store_true',
                    help='Whether to save the explanation excel')  # 是否导出excel数据
parser.add_argument('--save_explanation', default=False, action='store_true',
                    help='Whether to save the explanation')  # 是否保存解释
parser.add_argument('--saved_explanation_dir', type=str, default='saved_explanations',
                    help='directory of saved explanations')  # 保存解释的目录
parser.add_argument('--config_path', type=str, default='', help='path of saved configuration args')  # 保存的配置参数路径

'''
draw dgl
'''
parser.add_argument('--draw', type=int, default=0, help='draw dgl')

'''
Example:
python ABLE_g_run.py --device_id 0 --dataset_name aug_citation --radius 0.5 --num_epochs 25 --num_explain 100 --num_neighbor 10 --num_runs 4 --lambda_1 0.01 --lambda_2 0.1
python ABLE_g_run.py --device_id 0 --dataset_name lastfm --radius 0.3 --num_epochs 25 --num_explain 100 --num_neighbor 10 --num_runs 4
python ABLE_g_run.py --device_id 0 --dataset_name ACM --radius 0.9 --num_epochs 25 --num_explain 100 --num_neighbor 10 --num_runs 4 --num_hops 2 --emb_dim 64 --hidden_dim 64 --out_dim 64 --lambda_1 0.01 --lambda_2 0.1

'''

args = parser.parse_args()

if args.config_path:
    args = set_config_args(args, args.config_path, args.dataset_name, 'pagelink')

if 'citation' in args.dataset_name:  # 数据集节点类型
    args.src_ntype = 'author'
    args.tgt_ntype = 'paper'

elif 'synthetic' in args.dataset_name:
    args.src_ntype = 'user'
    args.tgt_ntype = 'item'

elif 'lastfm' in args.dataset_name:
    args.src_ntype = 'user'
    args.tgt_ntype = 'artist'

elif 'ACM' in args.dataset_name:
    args.src_ntype = 'paper'
    args.tgt_ntype = 'field'

if torch.cuda.is_available() and args.device_id >= 0:
    device = torch.device('cuda', index=args.device_id)
else:
    device = torch.device('cpu')

if args.link_pred_op in ['cat']:  # 根据链路预测操作类型配置预测参数。
    pred_kwargs = {"in_feats": args.out_dim, "out_feats": 1}
else:
    pred_kwargs = {}

if not args.saved_model_name:
    args.saved_model_name = f'{args.dataset_name}_model'  # 保存模型名称

print_args(args)  # 打印参数
set_seed(0)  # 设置随机种子

processed_g = load_dataset(args.dataset_dir, args.dataset_name, args.valid_ratio, args.test_ratio)[1]
mp_g, train_pos_g, train_neg_g, val_pos_g, val_neg_g, test_pos_g, test_neg_g = [g.to(device) for g in
                                                                                processed_g]  # 加载数据集到指定设备

print(mp_g)

for ntype in mp_g.ntypes:
    mp_g.nodes[ntype].data['h0'] = 0.5 * torch.randn(mp_g.num_nodes(ntype), args.emb_dim).to(device)
# studible random embedding

encoder = HeteroRGCN(mp_g, args.emb_dim, args.hidden_dim, args.out_dim)
model = HeteroLinkPredictionModel(encoder, args.src_ntype, args.tgt_ntype, args.link_pred_op, **pred_kwargs)
state = torch.load(f'{args.saved_model_dir}/{args.saved_model_name}.pth', map_location='cpu')
model.load_state_dict(state)  # 初始化模型并加载预训练模型参数

able_g = ABLEg(model,
               lr=args.lr,
               num_epochs=args.num_epochs,
               log=True,
               lambda_1=args.lambda_1,
               lambda_2=args.lambda_2,
               lambda_m=args.lambda_m,
               dataset_name=args.dataset_name,
               ).to(device) # init able explainer

# ========== ABLE-g explanation ==========
test_src_nids, test_tgt_nids = test_pos_g.edges()
test_ids = list(range(test_src_nids.shape[0]))

set_seed(5)
num_explain = min(args.num_explain, len(test_ids))
sample_ids = random.sample(test_ids, num_explain)

# 打印excel的批量测试版本
print(f"\n[ABLE-g RUN] Explaining {num_explain} test samples\n")


overall_stats, exres = evaluate_random_runs_ex(
        able_g=able_g,
        model=model,
        mp_g=mp_g,
        test_pos_g=test_pos_g,
        num_explain=args.num_explain,
        n_runs=args.num_runs,
        nbh_n_samples=args.num_neighbor,
        nbh_radius=args.radius,
        num_hops=args.num_hops,
        dataset_name=args.dataset_name,
        num_epochs=args.num_epochs,
        device=device,
        is_save_excel=args.save_excel,
        is_save_explanation=args.save_explanation
    )

print("Overall stats:", overall_stats)

#visualize_neighborhood_tsne(model, exres)


#测试：指定样本
# idx = 158
# i = 0
#
# while i<args.num_runs:
#     if idx >= len(test_src_nids):
#         print(f"Warning: Index {idx} is out of test set range (max {len(test_src_nids) - 1})")
#         continue
#
#     while True:
#         src_nid = test_src_nids[idx].to(device)
#         tgt_nid = test_tgt_nids[idx].to(device)
#         # 预检查子图规模
#         _, _, sg, _ = hetero_src_tgt_khop_in_subgraph(
#             args.src_ntype, src_nid, args.tgt_ntype, tgt_nid, mp_g, args.num_hops
#         )
#         total_edges = sum([sg.num_edges(et) for et in sg.canonical_etypes])
#         if total_edges >= 10:
#             break  # 合格，跳出 while 继续后续解释
#         else:
#             print(f"  [Skip] Sample {idx} has only {total_edges} edges, picking another...")
#             idx = random.randint(0, len(test_ids) - 1)
#
#     print(f"\n>>> Test edge index: {idx}")
#     print(f">>> src_nid={int(src_nid)}, tgt_nid={int(tgt_nid)}")
#
#     # 原图上的预测（对照）
#     with torch.no_grad():
#         base_score = model(
#             src_nid.unsqueeze(0),
#             tgt_nid.unsqueeze(0),
#             mp_g
#         )
#         base_prob = base_score.sigmoid().item()
#
#     # ABLE-g explain
#     results = able_g.explain(
#         src_nid=src_nid,
#         tgt_nid=tgt_nid,
#         ghetero=mp_g,
#         radius=args.radius,
#         n_samples=args.num_neighbor,  # 调试阶段建议小一点
#         num_hops=args.num_hops
#     )
#
#
#     print(f"[Summary] neighborhood predictions:")
#     for r in results["predictions"]:
#         print(f"  neigh {r['idx']}: score={r['score']}, pred={r['pred']}")
#     i = i + 1

# 测试：带可视化图
# for idx in sample_ids:
#     while True:
#         src_nid = test_src_nids[idx].to(device)
#         tgt_nid = test_tgt_nids[idx].to(device)
#         # 预检查子图规模
#         _, _, sg, _ = hetero_src_tgt_khop_in_subgraph(
#             args.src_ntype, src_nid, args.tgt_ntype, tgt_nid, mp_g, args.num_hops
#         )
#         total_edges = sum([sg.num_edges(et) for et in sg.canonical_etypes])
#         if total_edges >= 10:
#             break  # 合格，跳出 while 继续后续解释
#         else:
#             print(f"  [Skip] Sample {idx} has only {total_edges} edges, picking another...")
#             idx = random.randint(0, len(test_ids) - 1)
#
#     print(f"\n>>> Test edge index: {idx}")
#     print(f">>> src_nid={int(src_nid)}, tgt_nid={int(tgt_nid)}")
#
#     # 原图上的预测（对照）
#     with torch.no_grad():
#         base_score = model(
#             src_nid.unsqueeze(0),
#             tgt_nid.unsqueeze(0),
#             mp_g
#         )
#         base_prob = base_score.sigmoid().item()
#
#     # print(f"[Original graph prediction]")
#     # print(f"  logit = {base_score.item():.6f}")
#     # print(f"  prob  = {base_prob:.6f}")
#     # print(f"  label = {int(base_prob > 0.5)}")
#
#     # ABLE-g explain
#     results = able_g.explain(
#         src_nid=src_nid,
#         tgt_nid=tgt_nid,
#         ghetero=mp_g,
#         radius=args.radius,
#         n_samples=args.num_neighbor,     # 调试阶段建议小一点
#         num_hops=args.num_hops
#     )
#
#     print(f"[Summary] neighborhood predictions:")
#     for r in results["predictions"]:
#         print(f"  neigh {r['idx']}: score={r['score']}, pred={r['pred']}")
#
#     for r in results["adv_pairs"]:
#         fig, axes = plt.subplots(
#             nrows=1,
#             ncols=2,
#             figsize=(10, 5)
#         )
#         G_M = r["G_M"]
#         G_W = r["G_W"]
#         draw_able_graph_on_ax(
#             ax=axes[0],
#             ghetero=G_M["graph"],
#             edge_mask=G_M.get("edge_mask"),
#             src_nid=src_nid,
#             tgt_nid=tgt_nid,
#             title=f"G_M (pair {r['idx']})"
#         )
#
#         draw_able_graph_on_ax(
#             ax=axes[1],
#             ghetero=G_W["graph"],
#             edge_mask=G_W.get("edge_mask"),
#             src_nid=src_nid,
#             tgt_nid=tgt_nid,
#             title=f"G_W (pair {r['idx']})"
#         )
#
#         plt.tight_layout()
#         plt.show()
#         plt.close()
#         if r['idx']>1:
#             break











