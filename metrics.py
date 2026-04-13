""" fid_eval.py

    Calculating FID+ and FID- for the explainer.
"""

import numpy as np


# Evaluating node classification for a single input
def eval_node_class(pred_label, node_idx, ypred):
    return (pred_label[node_idx] == np.argmax(ypred.detach().cpu().numpy()))

def eval_fid(bool_eval_list, bool_eval_rev_list):
    return sum(bool_eval_list) / len(bool_eval_list), sum(bool_eval_rev_list) / len(bool_eval_rev_list)


def eval_SI_avg(eval_v_list, eval_v_rev_list, eval_view_list_list, eval_rev_view_list_list):
    result = []
    for eval_v, eval_v_rev, eval_view_list, eval_rev_view_list in zip(eval_v_list, eval_v_rev_list, eval_view_list_list,
                                                                      eval_rev_view_list_list):
        result.append(eval_SI(eval_v, eval_v_rev, eval_view_list, eval_rev_view_list))
    result_array = np.array(result)
    return np.mean(result_array[:, 0]), np.mean(result_array[:, 1])

def eval_SI(eval_v, eval_v_rev, eval_view_list, eval_rev_view_list):
    # return eval_v - (sum(eval_view_list) / len(eval_view_list)), eval_v_rev - (sum(eval_rev_view_list) / len(eval_rev_view_list))
    return int(eval_v > (sum(eval_view_list) / len(eval_view_list))), int(eval_v_rev >= (
                sum(eval_rev_view_list) / len(eval_rev_view_list)))

def eval_spa(ex_spa, syn_spa):
    return sum(ex_spa) / len(ex_spa), sum(syn_spa) / len(syn_spa)