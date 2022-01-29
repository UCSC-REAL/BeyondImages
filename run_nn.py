
import numpy as np


from utils.dataloader import DataLoader
from utils.misc import set_global_seeds
import torch
from hoc import *
from myutils import *
import torch.nn.functional as F
import copy


def cosDistance(features):
    # features: N*M matrix. N features, each features is M-dimension.
    features = F.normalize(features, dim=1) # each feature's l2-norm should be 1 
    similarity_matrix = torch.matmul(features, features.T)
    distance_matrix = 1.0 - similarity_matrix
    return distance_matrix

def count_2nn_acc(KINDS, feat_cord, label, cluster_sum):
    # feat_cord = torch.tensor(final_feat)
    cnt = [[] for _ in range(3)]
    cnt[0] = torch.zeros(KINDS)
    cnt[1] = torch.zeros(KINDS, KINDS)
    cnt[2] = torch.zeros(KINDS, KINDS, KINDS)
    dist = cosDistance(feat_cord).cpu().numpy()
    # print(dist.shape)
    # print(f'Use Euclidean distance')
    # dist = distEuclidean(feat_cord, feat_cord)
    
    max_val = np.max(dist)
    am = np.argmin(dist,axis=1)
    for i in range(cluster_sum):
        dist[i][am[i]] = 10000.0 + max_val
    min_dis_id = np.argmin(dist,axis=1)
    for i in range(cluster_sum):
        dist[i][min_dis_id[i]] = 10000.0 + max_val
    min_dis_id2 = np.argmin(dist,axis=1)
    for x1 in range(cluster_sum):
        cnt[0][label[x1]] += 1
        cnt[1][label[x1]][label[min_dis_id[x1]]] += 1
        cnt[2][label[x1]][label[min_dis_id[x1]]][label[min_dis_id2[x1]]] += 1

    return cnt


def get_T_global_min_new(data_set, max_step=501, T0=None, p0=None, lr=0.1, NumTest=50, all_point_cnt=15000, KINDS = 5):

    # Build Feature Clusters --------------------------------------
    # KINDS = 5
    # NumTest = 50
    # all_point_cnt = min(all_point_cnt,data_set['noisy_label'].shape[0])
    # print(f'Use {all_point_cnt} in each round. Total rounds {NumTest}.')

    p_estimate = [[] for _ in range(3)]
    p_estimate[0] = torch.zeros(KINDS)
    p_estimate[1] = torch.zeros(KINDS, KINDS)
    p_estimate[2] = torch.zeros(KINDS, KINDS, KINDS)
    # p_estimate_rec = torch.zeros(NumTest, 3)
    for idx in range(NumTest):
        print(idx, flush=True)
        # global
        sample = np.random.choice(range(data_set['feature'].shape[0]), all_point_cnt, replace=False)
        # final_feat, noisy_label = get_feat_clusters(data_set, sample)
        final_feat = data_set['feature'][sample]
        noisy_label = data_set['noisy_label'][sample]
        cnt_y_3 = count_y(KINDS, final_feat, noisy_label, all_point_cnt)
        for i in range(3):
            cnt_y_3[i] /= all_point_cnt
            p_estimate[i] = p_estimate[i] + cnt_y_3[i] if idx != 0 else cnt_y_3[i]

    for j in range(3):
        p_estimate[j] = p_estimate[j] / NumTest

    # print(p_estimate)
    # exit()
    loss_min, E_calc, P_calc, _ = calc_func(KINDS, p_estimate, False, torch.device('cpu'), max_step, T0, p0, lr=lr)
    E_calc = E_calc.cpu().numpy()
    P_calc = P_calc.cpu().numpy()
    return E_calc, P_calc


def run_T_weight(args_org, use_clean = False, use_pca = False):
    args= copy.deepcopy(args_org)
    set_global_seeds(args.seed)
    name = args.dataset
    data_set = DataLoader(args.dataset)
    X_train, X_test, y_train, y_test, y_train_noisy, _ = data_set.prepare_train_test_val(args.__dict__)


    dataset = {'feature': X_train, 'clean_label': y_train, 'noisy_label': y_train_noisy}
    KINDS = int(max(y_train)+1)
    args.num_class = KINDS
    args.use_pca = use_pca
    args.use_clean = use_clean
    args.ydim = 2
    args.xdim = 64


    args.device = 'cpu'

    T,p = estimate_T_HOC_weight(args, dataset)
    
    T_real = np.zeros((KINDS,KINDS))
    for i in range(y_train_noisy.shape[0]):
        T_real[y_train[i]][y_train_noisy[i]] += 1

    T_real = T_real/np.sum(T_real,1)[:,None]
    p_real = np.array([np.mean(y_train==i) for i in range(KINDS)])
    print(f'T_real = \n{np.round(T_real*100,1)}')
    print(f'p_real = \n{np.round(p_real*100,1)}')
    print(f'T_est = \n{np.round(T*100,1)}')
    print(f'p_est = \n{np.round(p*100,1)}')
    error = np.sum(np.abs(T_real-T))/KINDS/2
    if use_pca:
        print(f'[Error HOC weight] [{args.dataset}] [A] [{args.mi_type}] [{args.div}] {error}')
    else:
        print(f'[Error HOC weight] [{args.dataset}] [X] [{args.mi_type}] [{args.div}] {error}')
    return error



    
def run(args):

    

    args.seed = 1
    

   
    error_ours_X = run_T_weight(args, use_pca = False)

    error_ours_Z = run_T_weight(args, use_pca = True)
    
    return error_ours_X, error_ours_Z
    

if __name__ == '__main__':
    from utils.parser import parse_args
    run(parse_args())
