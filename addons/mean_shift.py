import numpy as np
import cv2
import seaborn as sns
from tqdm import tqdm
from addons.eda import *
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from sklearn.cluster import MeanShift


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


def get_segmented_image(data, model, im_num, mode):

    assert model.__class__ is MeanShift, "wrong model"

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
    assert model.__class__ is MeanShift, "wrong model"

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


def plot_MeanShift_analysis(data, mode: str, im_num: int=None, bandwidth=15):
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

    model = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
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


if __name__ == "main":
    print("You're running library file...")