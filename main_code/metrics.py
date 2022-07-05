from __future__ import annotations

import numpy as np
import random
from scipy.sparse import issparse
from s_dbw import S_Dbw, SD
from sklearn.metrics import f1_score, confusion_matrix, silhouette_score, davies_bouldin_score, calinski_harabasz_score

import time
from itertools import product
from math import log
import pandas as pd

# Scikit-Learn
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import pairwise_distances

from scipy.spatial.distance import cdist


def precision_recall_f1(y_true, y_pred):
    if (y_pred == -1).all():
        return 1, 0, 0

    matrix = np.zeros((len(np.unique(y_pred)), len(np.unique(y_true))))
    for i, c1 in enumerate(np.unique(y_pred)):
        for j, c2 in enumerate(np.unique(y_true)):
            matrix[i, j] = np.sum((y_pred == c1) & (y_true == c2))
            
    precision_j, recall_j = [], []
    f1_value_j = []

    for j, c in enumerate(np.unique(y_true)):
        if c == -1:
            continue
        
        add_i = 1 if -1 in y_pred else 0
        i = np.argmax(matrix[add_i:, j]) + add_i

        if matrix[i, j] == 0:
            precision = 1
            recall = 0
        else :
            precision = matrix[i, j] / np.sum(matrix[i, :])
            recall = matrix[i, j] / np.sum(matrix[:, j])
        f1_value_j.append((2*precision*recall) / (precision + recall))
        precision_j.append(precision)
        recall_j.append(recall)
    return np.mean(precision_j), np.mean(recall), np.mean(f1_value_j)

def mean_intracluster_distance(data, labels):
    s = []
    for c in set(labels):
        median = np.mean(data[labels==c], axis=0)[None, :]
        s.extend(list(cdist(data[labels==c], median, metric='euclidean').flatten()))
    return np.mean(s)


def divide(data, labels):
    clusters = set(labels)
    clusters_data = []
    for cluster in clusters:
        clusters_data.append(data[labels == cluster, :])
    return clusters_data

def get_centroids(clusters):
    centroids = []
    for cluster_data in clusters:
        centroids.append(cluster_data.mean(axis=0))
    return centroids

def cohesion(data, labels):
    clusters = sorted(set(labels))
    sse = 0
    for cluster in clusters:
        cluster_data = data[labels == cluster, :]
        centroid = cluster_data.mean(axis = 0)
        sse += ((cluster_data - centroid)**2).sum()
    return sse

def separation(data, labels, cohesion_score):
    # calculate separation as SST - SSE
    return cohesion(data, np.zeros(data.shape[0])) - cohesion_score

def SST(data):
    c = get_centroids([data])
    return ((data - c) ** 2).sum()

def SSE(clusters, centroids):
    result = 0
    for cluster, centroid in zip(clusters, centroids):
        result += ((cluster - centroid) ** 2).sum()
    return result

# Clear the store before running each time
within_cluster_dist_sum_store = {}
def within_cluster_dist_sum(cluster, centroid, cluster_id):
    if cluster_id in within_cluster_dist_sum_store:
        return within_cluster_dist_sum_store[cluster_id]
    else:
        result = (((cluster - centroid) ** 2).sum(axis=1)**.5).sum()
        within_cluster_dist_sum_store[cluster_id] = result
    return result

def RMSSTD(data, clusters, centroids):
    df = data.shape[0] - len(clusters)
    attribute_num = data.shape[1]
    return (SSE(clusters, centroids) / (attribute_num * df)) ** .5

# equal to separation / (cohesion + separation)
def RS(data, clusters, centroids):
    sst = SST(data)
    sse = SSE(clusters, centroids)
    return (sst - sse) / sst

def DB_find_max_j(clusters, centroids, i):
    max_val = 0
    max_j = 0
    for j in range(len(clusters)):
        if j == i:
            continue
        cluster_i_stat = within_cluster_dist_sum(clusters[i], centroids[i], i) / clusters[i].shape[0]
        cluster_j_stat = within_cluster_dist_sum(clusters[j], centroids[j], j) / clusters[j].shape[0]
        val = (cluster_i_stat + cluster_j_stat) / (((centroids[i] - centroids[j]) ** 2).sum() ** .5)
        if val > max_val:
            max_val = val
            max_j = j
    return max_val

def DB(data, clusters, centroids):
    result = 0
    for i in range(len(clusters)):
        result += DB_find_max_j(clusters, centroids, i)
    return result / len(clusters)

def XB(data, clusters, centroids):
    sse = SSE(clusters, centroids)
    min_dist = ((centroids[0] - centroids[1]) ** 2).sum()
    for centroid_i, centroid_j in list(product(centroids, centroids)):
        if (centroid_i - centroid_j).sum() == 0:
            continue
        dist = ((centroid_i - centroid_j) ** 2).sum()
        if dist < min_dist:
            min_dist = dist
    return sse / (data.shape[0] * min_dist)

def Sep(labels, k, dk, dist):
    clusters = sorted(set(labels))
    max_sep = None
    for cluster in clusters:
        cluster_data = dist[labels == cluster]
        cluster_data = cluster_data[:, labels != cluster]
        cluster_dk = dk[labels == cluster]
        sep = len(cluster_data[cluster_data <= np.c_[cluster_dk]]) / (k * cluster_data.shape[0])
        if max_sep is None or max_sep < sep:
            max_sep = sep
    return max_sep

def Com(labels, dist):
    clusters = sorted(set(labels))
    com = 0
    max_com = 0
    for cluster in clusters:
        cluster_data = dist[labels == cluster]
        cluster_data = cluster_data[:, labels == cluster]
        n_i = cluster_data.shape[0]
#        print(n_i, cluster_data.sum())
        if n_i > 1:
            cur_sum = 2 * cluster_data.sum() / (n_i * (n_i - 1))
            com += cur_sum
            if max_com < cur_sum:
                max_com = cur_sum
    return com, max_com

def CVNN(labels, k, dk, dist):
    com, max_com = Com(labels, dist)
    return Sep(labels, k, dk, dist) + com / max_com 

class InternalMetrics():
    def __init__(self, data=None, labels=None, k=None, dk=None, dist=None):
        self.data = data
        self.labels = labels
        self.k = k
        self.dk = dk
        self.dist = dist
        
    def get_scores(self):
        within_cluster_dist_sum_store.clear()
    
        clusters = divide(self.data, self.labels)
        centroids = get_centroids(clusters)

        scores = {}

        scores['cohesion'] = cohesion(self.data, self.labels)
        scores['separation'] = separation(self.data, self.labels, scores['cohesion'])
        
        scores['RMSSTD'] = RMSSTD(self.data, clusters, centroids)
        scores['RS'] = RS(self.data, clusters, centroids)
        
        #scores['calinski_harabasz_score'] = calinski_harabasz_score(self.data, self.labels)
        scores['silhouette_score'] = silhouette_score(self.data, self.labels)
        scores['DB'] = davies_bouldin_score(self.data, self.labels)
        
        scores['DB'] = DB(self.data, clusters, centroids)
        scores['XB'] = XB(self.data, clusters, centroids)
        scores['SD'] = SD(self.data, self.labels)
        scores['S_Dbw'] = S_Dbw(self.data, self.labels)
        # scores['CVNN'] = CVNN(self.labels, self.k, self.dk, self.dist)

        scores['mean_intracluster_distance'] = mean_intracluster_distance(self.data, self.labels)

        return scores
    
    def get_metrics_names(self):
        return ['cohesion', 
                'separation', 
                'RMSSTD',
                'RS', 
            #'calinski_harabasz_score', 
                'silhouette_score', 
                'DB', 'XB', 'SD', 'S_Dbw', #'CVNN',
                'mean_intracluster_distance',
                ]
