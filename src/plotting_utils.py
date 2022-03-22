import numpy as np
import pandas as pd
from scipy.spatial.distance import (directed_hausdorff, cdist)
from sklearn.preprocessing import normalize
import os
import matplotlib.pyplot as plt


def boot_strap(data, size=None):

    len_data = len(data)
    if size:
        mask = np.random.choice(len_data, size, replace=True)
    else:
        mask = np.random.choice(len_data, len_data, replace=True)
    return data[mask]


def compare_point_clouds(u, v, method='dorff'):
    if method == 'dorff':
        score = max(directed_hausdorff(u, v)[0],
                    directed_hausdorff(v, u)[0])

    elif method == 'cdist':
        score = cdist(u, v).sum()

    elif method == 'mtopdiv':
        barc = mtd.calc_cross_barcodes(u,
                                       v,
                                       batch_size1=100,
                                       batch_size2=1000,
                                       is_plot=False)
        #mtd_score = mtd.mtopdiv(u,v)
        score = mtd.get_score(barc, 1, 'sum_length')

    else:
        score = None

    return score


def get_scores_for_classes(path_to_train_norm_class,
                           path_to_test_classes_folder, method='cdist'):

    train = path_to_train_norm_class

    try:
        P = pd.read_csv(train, index_col=0).values
    except Exception as e:
        print(f'Some trubles due to {e}')

    folder = path_to_test_classes_folder

    scores = dict()
    for name in os.listdir(folder):
        try:
            Q = pd.read_csv(f'{folder}/{name}', index_col=0).values
            name = name.split('.')[0]
            score = compare_point_clouds(P, Q, method=method)
            score = round(score, 2)
            scores.update({name: score})
        except Exception as e:
            print(f'Some trubles due to {e}')

    return scores


def get_boot_strapped_hist(path_to_train_norm_class='',
                           path_to_test_norm_class='',
                           path_to_test_classes_folder='',
                           method='cdist',
                           N_strapps : int =1000,
                           N_strapped_samples=100):

    train = path_to_train_norm_class
    test = path_to_test_norm_class
    folder = path_to_test_classes_folder

    P = pd.read_csv(train, index_col=0).values
    P_test = pd.read_csv(test, index_col=0).values
    hists_abnorm = dict()
    hists_norm = dict()
    for name in os.listdir(folder):
        try:
            Q = pd.read_csv(f'{folder}/{name}', index_col=0).values
            name = name.split('.')[0]
            scores = list()
            norm_class_scores = list()

            for i in range(N_strapps):

                P_i = boot_strap(P, size=N_strapped_samples)
                P_test_i = boot_strap(P_test, size=N_strapped_samples)
                Q_i = boot_strap(Q, size=N_strapped_samples)
                scores.append(compare_point_clouds(P_i, Q_i, method=method))
                norm_class_scores.append(compare_point_clouds(P_test_i, P_i, method=method))

            hists_abnorm.update({name: scores})
            hists_norm.update({name: norm_class_scores})
        except Exception as e:
            print(f'Some trubles due to {e}')

    return hists_abnorm, hists_norm


def plot_hists(abnorm_hists, norm_hists, save=False, train_class_name='cat', **kwargs):

    plt.rcParams.update({'font.size': 6})
    fig, axarr = plt.subplots(ncols=3, nrows=3, figsize=(6.75, 6.5), dpi=600)

    for key, ax in zip(abnorm_hists.keys(), axarr.flatten()):

        ax.hist(abnorm_hists[key],
                bins=25,
                ec='k',
                alpha=0.8,
                label=f'Norm class vs. {key}',
                **kwargs)

        ax.hist(norm_hists[key],
                bins=25,
                ec='k',
                alpha=0.5,
                label=f'Norm train vs. norm test',
                **kwargs)
        #ax.set_title('Comparison of embeddings for cats and automobiles')
        ax.set_xlabel('Scores')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.tight_layout()
    if save:
        if type(save) == str:
            filename = save
        else:
            filename = 'hists_one_column_sized'
        plt.savefig(filename)


def generate_embeddings_of_one_image(model, dataloader, repeats=50):
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for _ in range(repeats):
            for (img1, img2), _, _ in dataloader:
                img = img1
                img = img.to(model.device)
                emb = model.backbone(img).flatten(start_dim=1)
                embeddings.append(emb)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)

    return embeddings


#embeddings = generate_embeddings_of_one_image(model, dataloader_test, repeats = 2)


if 0:
    # tests

    cats_train = '/Users/artemdembitskiy/Downloads/Telegram Desktop/train_embed/train_cat.csv'
    cats_test = '/Users/artemdembitskiy/Downloads/Telegram Desktop/test_embed/test_cat.csv'
    folder = '/Users/artemdembitskiy/Downloads/Telegram Desktop/test_embed'

    get_scores_for_classes(cats_train,
                           folder, method='dorff')

    abnorm, norm = get_boot_strapped_hist(path_to_train_norm_class=cats_train,
                                          path_to_test_norm_class=cats_test,
                                          path_to_test_classes_folder=folder,
                                          method='dorff',
                                          N_strapps=1000,
                                          N_strapped_samples=50)

    abnorm.pop('test_cat', None)
    norm.pop('test_cat', None)

    plot_hists(abnorm, norm)
