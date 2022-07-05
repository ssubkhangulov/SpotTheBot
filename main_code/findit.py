import numpy as np
from scipy.stats import binom
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

true_array = 0

def normalize_data(data):
    data = np.array(data)
    data -= np.min(data, axis=0)[None, :]
    data /= (np.max(data, axis=0) + 1e-5)[None, :]
    return data

def dod_directed(p, q, d_p, d_q, eps):
    dist = np.sum(d_p, axis=1) - np.sum((np.abs(p - q) <= eps) & d_p & d_q, axis=1)
    return dist.astype(int)

def dod(p, q, d_p, d_q, eps):
    return np.maximum(dod_directed(p, q, d_p, d_q, eps), dod_directed(q, p, d_q, d_p, eps))

def dod_medoids_simplified(mc1, mc2, mc1_members, mc2_members, mc1_KD, mc2_KD, eps):
    dod_value = dod(mc1, mc2, mc1_KD, mc2_KD, eps) * mc1_members[0] * mc2_members
    dod_value /= np.maximum(mc1_members[0] * mc2_members, 1e-5)
    return dod_value

def update_dod_medoids(dod_value_13, dod_value_23, mc1_members, mc2_members, mc3_members):
    dod_value_13 *= max(np.sum(mc1_members) * np.sum(mc3_members), 1e-5)
    dod_value_23 *= max(np.sum(mc2_members) * np.sum(mc3_members), 1e-5)
    dod_value = dod_value_13 + dod_value_23
    dod_value /= max((np.sum(mc1_members) + np.sum(mc2_members)) * np.sum(mc3_members), 1e-5)
    return dod_value

def dod_medoids(mc1, mc2, mc1_members, mc2_members, mc1_KD, mc2_KD, eps):
    dod_value = 0
    for m1, m1_members in zip(mc1, mc1_members):
        dod_value += np.sum(dod(m1[None, :], mc2, mc1_KD[None, :], mc2_KD[None, :], eps) * m1_members * mc2_members)
    dod_value /= np.sum(mc1_members) * np.sum(mc2_members)
    return dod_value
        
        
def chernoff_bounds(N, c_minsize, s_minsize):
    #--------------#
    delta = 0.01
    p = 1
    k = N/c_minsize
    #--------------#
    ch_b = lambda xsi: xsi*k*p + k*p*np.log(1/delta) + k*p*np.sqrt(np.log(1/delta)**2 + 2*xsi*np.log(1/delta))
    S = min(N, int(ch_b(s_minsize)))
    M = min(S, int(ch_b(1)))
    return S, M


def sampling_phase(data, c_minsize, s_minsize):
    S, M = chernoff_bounds(len(data), c_minsize, s_minsize)
    S_set = data[np.random.choice(len(data), S, replace=False)]
    M_set = S_set[np.random.choice(len(S_set), M, replace=False)]
    return M_set, S_set


def determine_KD(Voters, M_set, eps, V):
    #-------------#
    hundred_persent = 0.9995
    #-------------#
    votes = np.array([np.sum(np.abs(Voters[ind_m] - m[None, :]) <= eps, axis=0) for ind_m, m in enumerate(M_set)])
    threshold = int(binom.ppf(hundred_persent, V, eps*2))
    KD = votes >= threshold
    return KD


def member_assignment(M_set, S_set, KD, eps):
    medoid_members = np.zeros(len(M_set))
    for s in S_set:
        dist_directed = dod_directed(M_set, s[None, :], KD, true_array[None, :], eps)
        dist = dod(M_set, s[None, :], KD, true_array[None, :], eps)
        
        if (dist_directed == 0).any():
            dist[dist_directed != 0] = len(M_set[0]) + 1 #np.inf
            medoid_members[np.argmin(dist)] += 1
            
    return medoid_members


def merge_medoids(M_set, KD, medoid_members, d_mindist, eps):
    MC_ind = list(range(len(M_set)))
    MC = [[i] for i in MC_ind]
    dist_clusters = np.ones((len(M_set), len(M_set)))
    for i in MC:
        dist_clusters[i[0], :] = dod_medoids_simplified(M_set[i], M_set, medoid_members[i], medoid_members, KD[i], KD, eps)
    dist_clusters[np.arange(len(M_set)), np.arange(len(M_set))] = np.inf

    while True:
        if np.min(dist_clusters) > d_mindist:
            break

        i, j = np.unravel_index(np.argmin(dist_clusters), (len(M_set), len(M_set)))

        MC_ind.remove(j)
        
        dist_clusters[i, MC_ind] = np.array([update_dod_medoids(dist_clusters[i, k], dist_clusters[j, k], medoid_members[MC[i]], medoid_members[MC[j]], medoid_members[MC[k]]) for k in MC_ind])
        dist_clusters[MC_ind, i] = np.array(dist_clusters[i, MC_ind])
        
        MC[i].extend(MC[j])
        MC[j] = []
        
        dist_clusters[i, i] = dist_clusters[j, :] = dist_clusters[:, j] = np.inf
    
    MC_members = [medoid_members[MC[j]] for j in MC_ind]
    MC_KD = [KD[MC[j]] for j in MC_ind]
    MC_M = [M_set[MC[j]] for j in MC_ind]
    
    return MC_M, MC_KD, MC_members


def medoid_cluster_tuning(MC_KD, MC_members):
    #--------------#
    avg_threshold = 0.90
    #--------------#

    MC_KD_new = []
    for mc_KD, mc_members in zip(MC_KD, MC_members):
        avg = np.sum(mc_KD * np.array(mc_members)[:, None], axis=0) / max(int(np.sum(mc_members)), 1e-5)
        MC_KD_new.append(avg > avg_threshold)
    
    return MC_KD_new

def medoid_cluster_refining(MC_M, MC_KD, MC_members, d_mindist, eps, s_minsize):
    #--------------#
    avg_threshold = 0.90
    #--------------#
    MC_ind = list(range(len(MC_M)))
    dist_clusters = np.zeros((len(MC_M), len(MC_M)))
    for i, (mc_a, mc_a_KD, mc_a_members) in enumerate(zip(MC_M, MC_KD, MC_members)):
        for j, (mc_b, mc_b_KD, mc_b_members) in enumerate(zip(MC_M, MC_KD, MC_members)):
            dist_clusters[i, j] = dod_medoids(mc_a, mc_b, mc_a_members, mc_b_members, mc_a_KD, mc_b_KD, eps)
    dist_clusters[np.arange(len(MC_M)), np.arange(len(MC_M))] = np.inf
    
    while True:
        if np.min(dist_clusters) > d_mindist:
            break
        i, j = np.unravel_index(np.argmin(dist_clusters), (len(MC_M), len(MC_M)))
        MC_members[i] = np.append(MC_members[i], MC_members[j])
        MC_M[i] = np.concatenate((MC_M[i], MC_M[j]), axis=0)
        
        
        MC_KD[i] = ((MC_KD[i] * np.sum(MC_members[i]) + MC_KD[j] * np.sum(MC_members[j])) / \
            (np.sum(MC_members[i]) + np.sum(MC_members[j]))) > avg_threshold
        
        MC_members[j] = []
        MC_M[j] = []
        MC_KD[j] = []
        MC_ind.remove(j)
        
        for k in MC_ind:
            mc_a, mc_a_KD, mc_a_members = MC_M[i], MC_KD[i], MC_members[i]
            mc_b, mc_b_KD, mc_b_members = MC_M[k], MC_KD[k], MC_members[k]
            dist_clusters[i, k] = dod_medoids(mc_a, mc_b, mc_a_members, mc_b_members, mc_a_KD, mc_b_KD, eps)
        
        dist_clusters[i, i] = dist_clusters[j, :] = dist_clusters[:, j] = np.inf
        
    MC_KD = [MC_KD[i] for i in MC_ind if np.sum(MC_members[i]) > s_minsize]
    MC_M = [MC_M[i] for i in MC_ind if np.sum(MC_members[i]) > s_minsize]
    MC_members = [MC_members[i] for i in MC_ind if np.sum(MC_members[i]) > s_minsize]
    
    return MC_M, MC_KD, MC_members

def soundness(MC_KD, MC_members):
    soundness_value = 0
    for mc_kd, mc_members in zip(MC_KD, MC_members):
        soundness_value += int(np.sum(mc_kd) * np.sum(mc_members))
    return soundness_value

def dimension_voting(E_set, M_set, S_set, s_minsize, V, d_mindist):
    for eps in (E_set):
        # Voters - voters for dimensions of each cluster
        Voters = []
        for m in M_set:
            dist = np.sum(np.abs(S_set - m[None, :]) >= eps, axis=1)
            Voters.append(S_set[np.argsort(dist)[:V]])
        
        # KD - clusters' dimensions - vectors of bools
        KD = determine_KD(Voters, M_set, eps, V)
        
        # medoid_members - amount of points belongs to each medoid
        medoid_members = member_assignment(M_set, S_set, KD, eps)
        
        # remove unnecessary
        M_set_not_empty = M_set[medoid_members != 0]
        KD = KD[medoid_members != 0]
        medoid_members = medoid_members[medoid_members != 0]
        
        MC_M, MC_KD, MC_members = merge_medoids(M_set_not_empty, KD, medoid_members, d_mindist, eps)
        MC_KD = medoid_cluster_tuning(MC_KD, MC_members)
        
        MC_M, MC_KD, MC_members = medoid_cluster_refining(MC_M, MC_KD, MC_members, d_mindist, eps, s_minsize)

        if eps == E_set[0]:
            MC_M_best, MC_KD_best, MC_members_best = MC_M, MC_KD, MC_members
            best_eps = eps
        elif soundness(MC_KD_best, MC_members_best) < soundness(MC_KD, MC_members):
            MC_M_best, MC_KD_best, MC_members_best = MC_M, MC_KD, MC_members
            best_eps = eps

    return  MC_M_best, MC_KD_best, MC_members_best, best_eps


def data_assingning(data, MC_M, MC_KD, eps):
    medoids = []
    medoids_idx = []
    medoids_kd = []
    for mc_idx, (mc_m, mc_kd) in enumerate(zip(MC_M, MC_KD)):
        medoids.extend(list(mc_m))
        medoids_idx.extend([mc_idx] * len(mc_m))
        medoids_kd.extend(list(mc_kd[None, :]) * len(mc_m))
        
    medoids = np.array(medoids)
    medoids_idx = np.array(medoids_idx)
    medoids_kd = np.array(medoids_kd)

    if len(medoids) == 0:
        return np.array([-1] * len(data))
    
    data_clusters = []
    for p in (data):
        dist_directed = dod_directed(medoids, p[None, :], medoids_kd, true_array[None, :], eps)
        dist_directed_r = dod_directed(p[None, :], medoids, true_array[None, :], medoids_kd, eps)
        dist = np.maximum(dist_directed, dist_directed_r)
        
        if (dist_directed == 0).any():
            p_cluster = medoids_idx[dist_directed == 0][np.argmin(dist[dist_directed == 0])]
            data_clusters.append(p_cluster)
        else:
            data_clusters.append(-1)
            
    return np.array(data_clusters)


def FINDIT(data, c_minsize, d_mindist, eps=None, normalize_data_=False):
#     print(f'START\nc_minsize = {c_minsize}\nd_mindist = {d_mindist}')
    if normalize_data_:
        data = normalize_data(data)
    
    if c_minsize < 1:
        c_minsize = len(data) * c_minsize
   
    global true_array
    true_array = np.ones(len(data[0])).astype(bool)
    #--------------#
    # s_minsize(xsi) - minimum number of points in a cluster for chernoff_bounds sampling
    s_minsize = min(30, c_minsize)
    # V - number of voters for dimension_voting of each cluster
    V = min(20, s_minsize)
    #--------------#

    # E_set - set of eps - np.linspace(1/100, 25/100, 25)
    # M_set - set of medoids
    # S_set - set of dots
    M_set, S_set = sampling_phase(data, c_minsize, s_minsize)
    E_set = np.linspace(1/100, 25/100, 25) if eps is None else np.array([eps])
    
    S_set = np.array(S_set)
    M_set = np.array(M_set)
    
    MC_M, MC_KD, MC_members, eps = dimension_voting(E_set, M_set, S_set, s_minsize, V, d_mindist)
    data_clusters = data_assingning(data, MC_M, MC_KD, eps)
    cluster_dimensions = np.array(MC_KD)
    return data_clusters, cluster_dimensions