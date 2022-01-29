from myutils import *
import torch
import torch.nn.functional as F
import time
from utils_mi import mutual_information_2d_classification



def extract_feature(model_extract, dataloader, start_from_1 = False):
    """
    model_extract: the model for extract representations.   
        Usage: representation, prediction = model_extract(inputs)
    
    dataloader: format: noisy_label, clean_label, features(may be a tuple for NLP tasks)
    """
    model_extract.eval()
    feature_rec, label_noisy_rec, label_clean_rec = [], [], []

    cnt = 0 
    for _, data_tuple in enumerate(dataloader):
        if len(data_tuple) == 3: # format noisy_label, clean_label, feature
            label, label_clean, feature = data_tuple
            embedded, _ = model_extract(feature)
        else:  # format noisy_label, clean_label, text, offsets:
            (label, label_clean, text, offsets) = data_tuple
            embedded, _ = model_extract(text, offsets)
        feature_rec.append(embedded.detach().cpu())
        label_noisy_rec.append(label.detach().cpu())
        label_clean_rec.append(label_clean.detach().cpu())
        cnt += 1
        if cnt > 2000:
            print(f'dataset is too large. Only use the first {cnt-1} epochs ({label.shape[0]*(cnt-1)} points).')
            break

    feature_rec = torch.cat(feature_rec, dim = 0)
    label_clean_rec = torch.cat(label_clean_rec, dim = 0)
    label_noisy_rec = torch.cat(label_noisy_rec, dim = 0)
    if start_from_1:
        dataset_extracted = {'feature': feature_rec.cpu().numpy(), 'clean_label': label_clean_rec.cpu().numpy() - 1, 'noisy_label': label_noisy_rec.cpu().numpy() - 1}
    else:
        dataset_extracted = {'feature': feature_rec.cpu().numpy(), 'clean_label': label_clean_rec.cpu().numpy(), 'noisy_label': label_noisy_rec.cpu().numpy()}
    
    return dataset_extracted




def get_T_global_min_new(data_set, max_step=501, T0=None, p0=None, lr=0.1, NumTest=50, all_point_cnt=15000, KINDS = 5, mi = None):

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
        cnt_y_3 = count_y(KINDS, final_feat, noisy_label, all_point_cnt, mi=mi)
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

def estimate_T_HOC_weight(args, dataset_extracted):
    # ---------------- estimate W ----------------
    KINDS = args.num_class
    num_train = dataset_extracted['noisy_label'].shape[0]
    all_point_cnt = min(max(num_train//2,KINDS**2 * 2),15000)  # 5000 for 5-class classification, 15000 for 10-class or more
    print(f'num_train = {num_train}, use {all_point_cnt} points in each round', flush=True)
    

    for mi_loop in range(1):
        mi = []

        
        use_pca = args.use_pca
        if use_pca:
            X = dataset_extracted['feature']
            cov_mat = np.dot(X.transpose(),X)/X.shape[0]
            eig_val, eig_vec = np.linalg.eig(cov_mat) # each column is an eigen vector
            eig_vec = np.real(eig_vec)
            # tmp = np.dot(eig_vec, X.transpose())
            tmp = np.dot(X, eig_vec)
            norm_para = np.sqrt(np.mean(tmp**2,axis=0))
            


        for i in range(dataset_extracted['feature'].shape[1]):
            if use_pca:
                feature = tmp[:,i] / norm_para[i] 

            else:
                feature = dataset_extracted['feature'][:,i]

            if args.use_clean:
                label = dataset_extracted['clean_label']

            else:
                label = dataset_extracted['noisy_label']

            mi.append(mutual_information_2d_classification(feature,label.reshape(-1), normalized = args.norm, smooth = args.smooth, div = args.div, ydim = args.ydim, xdim = args.xdim))
            

        if args.mi_type == 'None':
            mi = 1
        elif args.mi_type == 'plain':  
            mi = np.array(mi)
            mi += -min(mi) + 1e-8
            mi /= max(mi)
        elif args.mi_type == 'log':  
            mi = np.log(mi)
            mi += -min(mi) + 1e-8
            mi /= max(mi)
        elif args.mi_type == 'sfmx':  
            mi = F.softmax(torch.tensor(mi), dim = 0).numpy()
        elif args.mi_type == 'linRank':  
            mi = np.array(mi)
            tmp = np.linspace(1e-8,1,mi.shape[0])
            loc = np.argsort(mi)
            mi[loc] = tmp    
        elif args.mi_type == 'quadRank':  
            mi = np.array(mi)
            tmp = np.linspace(1e-8,1,mi.shape[0]) ** 2
            loc = np.argsort(mi)
            mi[loc] = tmp     
        elif args.mi_type == 'sqrtRank':  
            mi = np.array(mi)
            tmp = np.sqrt(np.linspace(1e-8,1,mi.shape[0]))
            loc = np.argsort(mi)
            mi[loc] = tmp   
        elif args.mi_type == 'logRank':  
            mi = np.array(mi)
            tmp = np.log(np.linspace(1e-8,1,mi.shape[0]))
            tmp += -min(tmp) + 1e-8
            tmp /= max(tmp)
            loc = np.argsort(mi)
            mi[loc] = tmp   
        elif args.mi_type == 'expRank':  
            mi = np.array(mi)
            tmp = np.exp(np.linspace(1e-8,1,mi.shape[0]))
            tmp += -min(tmp) + 1e-8
            tmp /= max(tmp)
            loc = np.argsort(mi)
            mi[loc] = tmp
        elif args.mi_type == 'logShiftRank':  
            mi = np.array(mi)
            tmp = np.log(1+np.linspace(1e-8,1,mi.shape[0]))
            tmp += -min(tmp) + 1e-8
            tmp /= max(tmp)
            loc = np.argsort(mi)
            mi[loc] = tmp

        if args.debug:
            print(f'mi (sorted) is \n{np.round(sorted(mi),2)}')


    if use_pca:
        
        tmp = np.dot(X, eig_vec)
        norm_para = np.sqrt(np.mean(tmp**2,axis=0))

        if args.mi_type != 'None':
            tmp = tmp/norm_para

        dataset_extracted['feature'] = torch.tensor(tmp * mi, device=args.device)  
        dataset_extracted['feature'][torch.isnan(dataset_extracted['feature'])] = 0.0
    else:

        dataset_extracted['feature'] = torch.tensor(dataset_extracted['feature'] * mi, dtype=torch.float64, device=args.device)
    T, p = get_T_global_min_new(dataset_extracted, max_step=1501, T0=None, p0=None, lr=0.1, NumTest=max(50, num_train//all_point_cnt * 3), all_point_cnt=all_point_cnt, KINDS = KINDS, mi = None) # must set mi=None since mi is already used above

    return T, p


def func(KINDS, p_estimate, T_out, P_out, N,step, LOCAL, _device):
    eps = 1e-2
    eps2 = 1e-8
    eps3 = 1e-5
    loss = torch.tensor(0.0).to(_device)       # define the loss

    P = smp(P_out)
    T = smt(T_out)

    mode = random.randint(0, KINDS-1)
    mode = -1
    # Borrow p_ The calculation method of real is to calculate the temporary values of T and P at this time: N, N*N, N*N*N
    p_temp = count_real(KINDS, T.to(torch.device("cpu")), P.to(torch.device("cpu")), mode, _device)

    weight = [1.0,1.0,1.0]
    # weight = [2.0,1.0,1.0]

    for j in range(3):  # || P1 || + || P2 || + || P3 ||
        p_temp[j] = p_temp[j].to(_device)
        loss += weight[j] * torch.norm(p_estimate[j] - p_temp[j]) #/ np.sqrt(N**j)
    for i in range(KINDS): # guarantee diagonal dominant
        if T[i][i] < torch.max(T[i]):
            loss += (T[i][i] - torch.max(T[i]))**2
    if step > 100 and LOCAL and KINDS != 100:
        loss += torch.mean(torch.log(P+eps))/10

    return loss


def calc_func(KINDS, p_estimate, LOCAL, _device = torch.device("cpu"), max_step = 501, T0=None, p0 = None, lr = 0.1):
    # init
    # _device =  torch.device("cpu")
    N = KINDS
    eps = 1e-8
    if T0 is None:
        # T = 1 * torch.eye(N) - torch.ones((N,N))
         T = 3 * torch.eye(N) - 2*torch.rand((N,N))
    else:
        T = T0

    if p0 is None:
        P = torch.ones((N, 1), device = None) / N + torch.rand((N,1), device = None)*0.1     # P：0-9 distribution
    else:
        P = p0

    T = T.to(_device)
    P = P.to(_device)
    p_estimate = [item.to(_device) for item in p_estimate]
    # print(f'using {_device} to solve equations')

    T.requires_grad = True
    P.requires_grad = True

    optimizer = torch.optim.Adam([T, P], lr = lr)

    # train
    loss_min = 100.0
    T_rec = T.detach()
    P_rec = P.detach()

    time1 = time.time()
    for step in range(max_step):
        if step:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss = func(KINDS, p_estimate, T, P, N,step, LOCAL, _device)
        if loss < loss_min and step > 5:
            loss_min = loss.detach()
            T_rec = T.detach()
            P_rec = P.detach()
        if step % 100 == 0:
            print('loss {}'.format(loss))
        #     print(f'step: {step}  time_cost: {time.time() - time1}')
            # print(f'T {np.round(smt(T.cpu()).detach().numpy()*100,1)}', flush=True)
            # print(f'P {np.round(smp(P.cpu().view(-1)).detach().numpy()*100,1)}', flush=True)
        #     time1 = time.time()
    # if global_var.get_value('T_init') is None:
    # global_var.set_value('T_init', T_rec.detach())
        # tmp = global_var.get_value('T_init')
        # print(f'set T_init to {tmp}')
    # if global_var.get_value('p_init') is None:
    # global_var.set_value('p_init', P_rec.detach())
    # print(f'T_init and p_init are updated')
    return loss_min, smt(T_rec).detach(), smp(P_rec).detach(), T_rec.detach()


def count_y(KINDS, feat_cord, label, cluster_sum, mi = None):
    # feat_cord = torch.tensor(final_feat)
    cnt = [[] for _ in range(3)]
    cnt[0] = torch.zeros(KINDS)
    cnt[1] = torch.zeros(KINDS, KINDS)
    cnt[2] = torch.zeros(KINDS, KINDS, KINDS)
    # if isinstance(feat_cord, torch.Tensor):
    #     feat_cord = feat_cord.cpu().numpy()
    # dist = distCosine(feat_cord, feat_cord)
    if isinstance(feat_cord, np.ndarray):
        feat_cord = torch.tensor(feat_cord, dtype=torch.float64)
    dist = cosDistance(feat_cord, mi = mi).cpu().numpy()
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

def count_2nn_acc(KINDS, feat_cord, label, cluster_sum):
    # feat_cord = torch.tensor(final_feat)
    cnt = [[] for _ in range(3)]
    cnt[0] = torch.zeros(KINDS)
    cnt[1] = torch.zeros(KINDS, KINDS)
    cnt[2] = torch.zeros(KINDS, KINDS, KINDS)
    feat_cord = feat_cord.cpu().numpy()
    dist = distCosine(feat_cord, feat_cord)
    # print(dist.shape)
    # print(f'Use Euclidean distance')
    # dist = distEuclidean(feat_cord, feat_cord)
    
    max_val = np.max(dist)
    am = np.argmin(dist,axis=1)
    # TODO: speedup this part
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







    