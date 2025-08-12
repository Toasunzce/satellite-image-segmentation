from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns

import pandas as pd
import numpy as np
import cv2
import scipy as sp

from sklearn.preprocessing import StandardScaler

import os
from collections import Counter
from tqdm import tqdm
import warnings



# --------- EDA ---------
def load_images(path: str, mode: str='RGB') -> np.ndarray:
    
    conversion_attr = f"COLOR_BGR2{mode}"
    conversion_code = getattr(cv2, conversion_attr)
    
    images = []
    for file in os.listdir(path):
        if file.lower().endswith('.jpg'):
            img_path = os.path.join(path, file)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img = cv2.cvtColor(img_bgr, conversion_code)
            images.append(img)
            
    return np.array(images)


# --------- KMeans & Mean-Shift ---------
from sklearn.cluster import KMeans, MeanShift

def plot_multiple_Kmeans_analysis(data, reps: int=20, k_values=range(3, 10), mode: str='RGB'):

    all_pixels = data.reshape(-1, 3)
    kmeans_models = dict()

    for k in tqdm(k_values):
        kmeans_models[k] = KMeans(n_clusters=k).fit(all_pixels)

    if mode != 'RGB':
        conversion_code = getattr(cv2, f"COLOR_{mode}2RGB")

    used = list()

    dist_x = int((len(kmeans_models) + 1) * 1.8) # some fucking magic numbers there
    dist_y = int(reps * 2.2)
    fig, ax = plt.subplots(reps, len(kmeans_models) + 1, figsize=(dist_x, dist_y), dpi=100)

    for i in range(reps):

        num = np.random.randint(0, len(data))
        while num in used:
            num = np.random.randint(0, len(data))
        used.append(num)

        im = data[num]
        if mode != 'RGB':
            im = cv2.cvtColor(im, conversion_code)

        ax[i, 0].imshow(im)
        ax[i, 0].axis('off')
        ax[i, 0].set_title(f'Original {num}')

        pixels = data[num].reshape(-1, 3)

        for idx, k in enumerate(k_values, start=1):
            model = kmeans_models[k]
            labels = model.predict(pixels)
            segmented_img = model.cluster_centers_[labels]
            segmented_img = segmented_img.reshape(data[num].shape)
            segmented_img = segmented_img.astype(np.uint8)

            if mode != 'RGB':
                segmented_img = cv2.cvtColor(segmented_img, conversion_code)

            ax[i, idx].imshow(segmented_img)
            ax[i, idx].axis('off')
            ax[i, idx].set_title(f'{k=} {mode=}')

    plt.tight_layout()


def plot_multiple_MeanShift_analysis(data, reps: int = 20, bandwidth_values=range(18, 24), mode: str = 'RGB'):

    all_features = []

    # adding extra features for images - coords for every pixel
    for im in data:
        h, w, _ = im.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        coords = np.stack((X, Y), axis=2) 
        features = np.concatenate((im, coords), axis=2) 
        all_features.append(features.reshape(-1, 5))
    all_features = np.vstack(all_features).astype(float) # list -> ndarray

    meanshift_models = dict()
    for k in tqdm(bandwidth_values):
        meanshift_models[k] = MeanShift(bandwidth=k, n_jobs=-1, bin_seeding=True).fit(all_features)

    if mode != 'RGB':
        conversion_code = getattr(cv2, f"COLOR_{mode}2RGB")

    used = []
    dist_x = int((len(meanshift_models) + 1) * 1.8)  # more magic numbers
    dist_y = int(reps * 2.2)
    fig, ax = plt.subplots(reps, len(meanshift_models) + 1, figsize=(dist_x, dist_y), dpi=100)

    for i in range(reps):
        num = np.random.randint(0, len(data))
        while num in used:
            num = np.random.randint(0, len(data))
        used.append(num)

        im = data[num]
        if mode != 'RGB':
            im = cv2.cvtColor(im, conversion_code)

        ax[i, 0].imshow(im)
        ax[i, 0].axis('off')
        ax[i, 0].set_title(f'Original {num}')

        h, w, _ = im.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        coords = np.stack((X, Y), axis=2)
        features = np.concatenate((data[num], coords), axis=2).reshape(-1, 5).astype(float)
        for idx, k in enumerate(bandwidth_values, start=1):
            model = meanshift_models[k]
            labels = model.predict(features)
            segmented_img = model.cluster_centers_[:, :3][labels]
            segmented_img = segmented_img.reshape(h, w, 3).astype(np.uint8)

            if mode != 'RGB':
                segmented_img = cv2.cvtColor(segmented_img, conversion_code)

            ax[i, idx].imshow(segmented_img)
            ax[i, idx].axis('off')
            ax[i, idx].set_title(f'bw{k} {mode=}')

    plt.tight_layout()


def convert_image(im: np.ndarray, mode: str='RGB') -> np.ndarray:
    if mode != 'RGB':
        conversion_code = getattr(cv2, f"COLOR_{mode}2RGB")
        im = cv2.cvtColor(im, conversion_code)
    return im


def get_segmented_image(data, model, im_num, mode):
    assert model.__class__ in [KMeans, MeanShift], "wrong model"

    h, w, _ = data[im_num].shape
    
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack((X, Y), axis=2)
    features = np.concatenate((data[im_num], coords), axis=2).reshape(-1, 5).astype(float)
    
    labels = model.predict(features)

    segmented_im = model.cluster_centers_[:, :3][labels]
    segmented_im = segmented_im.reshape(data[im_num].shape)
    segmented_im = segmented_im.astype(np.uint8)

    if mode != 'RGB':
        segmented_im = convert_image(segmented_im, mode)

    return segmented_im, labels


def get_pyplot_palette(model, mode) -> dict[int, list]:
    assert model.__class__ in [KMeans, MeanShift], "wrong model"

    cluster_colors_rgb = []

    for color in model.cluster_centers_[:, :3]:
        color_img = color.reshape(1, 1, 3).astype(np.uint8)
        if mode != 'RGB':
            color_img = convert_image(color_img, mode)
        cluster_colors_rgb.append(color_img[0, 0])

    cluster_colors_rgb = np.array(cluster_colors_rgb)

    if cluster_colors_rgb.max() <= 1.0:
        cluster_colors_rgb = (cluster_colors_rgb * 255)
    cluster_colors_rgb = cluster_colors_rgb.astype(np.uint8)

    hex_colors = [mcolors.to_hex(c / 255) for c in cluster_colors_rgb]
    return {str(i): hex_colors[i] for i in range(len(hex_colors))}


def plot_cluster_analysis(data, mode: str, im_num: int=None, model_class=KMeans, **kwargs):
    """Can be used with models that implement cluster centers"""
    if im_num is None:
        im_num = np.random.randint(len(data))

    im = data[im_num]
    im = convert_image(im, mode)

    all_features = []
    for img in data:
        h, w, _ = img.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        coords = np.stack((X, Y), axis=2)
        features = np.concatenate((img, coords), axis=2)
        all_features.append(features.reshape(-1, 5))
    all_features = np.vstack(all_features).astype(float)

    model = model_class(**kwargs)
    model.fit(all_features)

    segmented_im, labels = get_segmented_image(data, model, im_num, mode)
    palette = get_pyplot_palette(model, mode=mode)

    fig, ax = plt.subplots(1, 3, figsize=(6,2))
    sns.countplot(x=labels, palette=palette, ax=ax[2],
                  order=[str(i) for i in range(len(model.cluster_centers_))])
    box = ax[2].get_position()
    ticks = ax[2].get_yticks()

    ax[0].imshow(im)
    ax[0].axis('off')
    ax[0].set_title(f'Original {im_num}')

    ax[1].imshow(segmented_im)
    ax[1].axis('off')
    ax[1].set_title(f'{mode=}  k={len(model.cluster_centers_)}')

    ax[2].set_title('Clusters size')
    ax[2].set_ylabel('')
    ax[2].tick_params(axis='y', labelsize=8)  
    ax[2].set_position([box.x0, box.y0 + 0.047, box.width, box.height * 0.88])
    ax[2].yaxis.tick_right()
    ax[2].set_yticks([ticks[-1]])
    ax[2].set_yticklabels([f'{int(ticks[-1])}'], fontsize=10)
    ax[2].set_xticks([])


# --------- Spectral Clustering ---------
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

if __name__ == "main":
    print("You're running library file...")