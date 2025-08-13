import numpy as np
import cv2
import seaborn as sns
from tqdm import tqdm
from addons.eda import *
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from sklearn.cluster import SpectralClustering


def spectral_segmentation(img_rgb, n_clusters=4, gamma=0.04): # wip, one image learning solution below
    
    im = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    height, width, _ = im.shape
    X, Y = np.meshgrid(np.arange(width), np.arange(height)) 
    coords = np.stack((X, Y), axis=2)

    features = np.concatenate((im, coords), axis=2).reshape(-1, 5)
    features = features.astype(float)

    sc = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=gamma, assign_labels='kmeans', random_state=42)

    labels = sc.fit_predict(features)

    segmented = np.zeros_like(img_rgb)
    for k in range(n_clusters):
        mask = (labels == k)
        cluster_color = np.mean(img_rgb.reshape(-1, 3)[mask], axis=0)
        segmented.reshape(-1, 3)[mask] = cluster_color

    segmented = segmented.astype(np.uint8)
    return segmented