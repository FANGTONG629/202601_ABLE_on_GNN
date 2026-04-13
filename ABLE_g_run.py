import os
import torch
import argparse
import pickle
import random
from tqdm.auto import tqdm
from pathlib import Path

from metrics import eval_SI_avg
from utils import set_seed, print_args, set_config_args
from data_processing import load_dataset
from model import HeteroRGCN, HeteroLinkPredictionModel
import numpy as np
import time
from ABLE_g import ABLEg

parser = argparse.ArgumentParser(description='Explain link predictor')
parser.add_argument('--device_id', type=int, default=-1)

'''
Dataset args
'''
parser.add_argument('--dataset_dir', type=str, default='datasets')  # 数据集目录
parser.add_argument('--dataset_name', type=str, default='lastfm')  # 数据集名称
parser.add_argument('--valid_ratio', type=float, default=0.1)  # 验证集比例
parser.add_argument('--test_ratio', type=float, default=0.2)  # 测试集比例
parser.add_argument('--max_num_samples', type=int, default=-1,
                    help='maximum number of samples to explain, for fast testing. Use all if -1')  # 最大预测样本数

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
                    help='number of test samples to explain')

parser.add_argument('--lr', type=float, default=0.1, help='explainer learning_rate')  # 解释器的学习率
parser.add_argument('--alpha', type=float, default=1.0, help='explainer on-path edge regularizer weight')  # 路径边正则化权重
parser.add_argument('--beta', type=float, default=1.0, help='explainer off-path edge regularizer weight')  # 非路径边正则化权重
parser.add_argument('--num_hops', type=int, default=2, help='computation graph number of hops')  # 计算图的跳数，默认为2
parser.add_argument('--num_epochs', type=int, default=100, help='How many epochs to learn the mask')  # 学习掩膜的训练轮数
parser.add_argument('--num_paths', type=int, default=6, help='How many paths to generate')  # 生成的路径数量，默认40
parser.add_argument('--max_path_length', type=int, default=5, help='max lenght of generated paths')  # 生成路径的最大长度，默认5
parser.add_argument('--k_core', type=int, default=1, help='k for the k-core graph')  # k-core图k值，为2
parser.add_argument('--prune_max_degree', type=int, default=200,
                    help='prune the graph such that all nodes have degree smaller than max_degree. No prune if -1')  # 剪枝最大度数，200
parser.add_argument('--save_explanation', default=False, action='store_true',
                    help='Whether to save the explanation')  # 是否保存解释
parser.add_argument('--saved_explanation_dir', type=str, default='saved_explanations',
                    help='directory of saved explanations')  # 保存解释的目录
parser.add_argument('--config_path', type=str, default='', help='path of saved configuration args')  # 保存的配置参数路径
'''
Ablation args
'''
parser.add_argument('--HSM', type=int, default=0, help='CRETE w/o HSM')
parser.add_argument('--HID_RAN', type=int, default=0, help='CRETE w/o HID RAN')
parser.add_argument('--HID_AC', type=int, default=0, help='CRETE w/o HID AC')
parser.add_argument('--CON', type=int, default=0, help='CRETE w/o CON')
'''
draw dgl
'''
parser.add_argument('--draw', type=int, default=0, help='draw dgl')

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

if 'lastfm' in args.dataset_name:
    for ntype in mp_g.ntypes:
        mp_g.nodes[ntype].data['h0'] = 0.5 * torch.randn(mp_g.num_nodes(ntype), args.emb_dim).to(device)
# studible random embedding

encoder = HeteroRGCN(mp_g, args.emb_dim, args.hidden_dim, args.out_dim)
model = HeteroLinkPredictionModel(encoder, args.src_ntype, args.tgt_ntype, args.link_pred_op, **pred_kwargs)
state = torch.load(f'{args.saved_model_dir}/{args.saved_model_name}.pth', map_location='cpu')
model.load_state_dict(state)  # 初始化模型并加载预训练模型参数

able_g = ABLEg(model, log=True).to(device) # init able explainer

# ========== ABLE-g explanation ==========
test_src_nids, test_tgt_nids = test_pos_g.edges()
test_ids = list(range(test_src_nids.shape[0]))

set_seed(5)
num_explain = min(args.num_explain, len(test_ids))
sample_ids = random.sample(test_ids, num_explain)

print(f"\n[ABLE-g RUN] Explaining {num_explain} test samples\n")

for idx in sample_ids:
    src_nid = test_src_nids[idx].to(device)
    tgt_nid = test_tgt_nids[idx].to(device)

    print(f"\n>>> Test edge index: {idx}")
    print(f">>> src_nid={int(src_nid)}, tgt_nid={int(tgt_nid)}")

    # 原图上的预测（对照）
    with torch.no_grad():
        base_score = model(
            src_nid.unsqueeze(0),
            tgt_nid.unsqueeze(0),
            mp_g
        )
        base_prob = base_score.sigmoid().item()

    print(f"[Original graph prediction]")
    print(f"  logit = {base_score.item():.6f}")
    print(f"  prob  = {base_prob:.6f}")
    print(f"  label = {int(base_prob > 0.5)}")

    # ABLE-g explain
    results = able_g.explain(
        src_nid=src_nid,
        tgt_nid=tgt_nid,
        ghetero=mp_g,
        radius=0.5,
        n_samples=10,     # 调试阶段建议小一点
        num_hops=args.num_hops
    )

    print(f"[Summary] neighborhood predictions:")
    for r in results:
        print(f"  neigh {r['idx']}: score={r['score']}, pred={r['pred']}")


