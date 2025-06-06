import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from lifelines.statistics import multivariate_logrank_test
from lifelines import KaplanMeierFitter
import torch
import pandas as pd

from typing import Sequence, Tuple
import matplotlib.patches as patches
import math


def plot_Weibull_cdf(t_horizon, shape, scale, data_name='sim', num_inst=1000, num_feat=200, seed=42):
    step = 100
    for i in range(len(shape)):
        k = shape[i]
        b = scale[i]
        s = np.zeros(step)
        t_space = np.linspace(0, t_horizon, step)
        for j in range(step):
            s[j] = np.exp(-(np.power(np.exp(b) * t_space[j], np.exp(k))))
        plt.plot(t_space, s, label='Expert Distribution {}'.format(i))
    plt.legend()
    # plt.title('Weibull CDF, Data: {}, Seed: {}'.format(data_name, seed))
    plt.title('Weibull CDF, Data: {}'.format(data_name), fontsize=16)
    if data_name == 'sim':
        plt.savefig('./Figures/Weibull_cdf_#clusters{}_{}_{}x{}_seed{}.png'.
                    format(len(shape), data_name, num_inst, num_feat, seed))
    else:
        plt.savefig('./Figures/Weibull_cdf_#clusters{}_{}_seed{}.png'.
                    format(len(shape), data_name, seed))
    plt.show()
    plt.close()


def plot_loss_c_index(results_all, lr, epoch, bs):
    # plot the loss and the C Index
    fig, ax = plt.subplots()
    ax.plot(results_all[:, 0], color='tab:red', label='train loss')
    ax.plot(results_all[:, 1], color='tab:blue', label='test loss')
    ax.set_xlabel("epoch", fontsize=14)
    ax.set_ylabel("loss", fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(results_all[:, 2], color='tab:green', label='C Index Test')
    ax2.plot(results_all[:, 3], color='tab:orange', label='C Index Train')
    ax2.set_ylabel("C Index", fontsize=14)
    ax2.plot(np.nan, color='tab:red', label='train loss')  # print an empty line to represent loss
    ax2.plot(np.nan, color='tab:blue', label='test loss')

    ax2.legend(loc=0)
    ax.grid()
    plt.title('lr: {:.2e}, epoch: {}, batch_size: {}'.format(lr, epoch, bs))
    plt.savefig('./cindex_plots/loss_c_index_lr{}_epoch{}_bs{}.png'.format(lr, epoch, bs))
    plt.show()
    plt.close()


def visualize(X_train_list, X_test_list, data_name, is_normalize=0, is_TSNE=1):
    """This function is to visualize the scatter plot with clustering information"""

    X_train = np.concatenate(X_train_list)
    X_test = np.concatenate(X_test_list)
    X = np.concatenate((X_train, X_test), axis=0)
    # normalize
    if is_normalize == 1:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    if is_TSNE == 1:
        # embed using TSNE
        embeddings = TSNE(random_state=42).fit_transform(X)
    else:
        # embed using UMAP
        trans = umap.UMAP(random_state=42).fit(X)
        embeddings = trans.embedding_
        # embeddings = []


    xlim = [-100, 95]
    ylim = [-90, 90]

    # show each cluster separately on all train data
    len_train = 0
    for idx, f in enumerate(X_train_list):
        plt.scatter(embeddings[len_train:(len_train + len(f)), 0],
                    embeddings[len_train:(len_train + len(f)), 1],
                    s=5, label='Train Cluster {}'.format(idx))

        len_train += len(f)
        plt.xlim(xlim)
        plt.ylim(ylim)
    plt.title(data_name)
    # plt.legend()
    plt.xlabel('Train Data with #Clusters {}'.format(len(X_train_list)))
    plt.show()
    plt.close()

    # show each cluster separately on all test data
    len_test = len(X_train)
    for idx, f in enumerate(X_test_list):
        plt.scatter(embeddings[len_test:(len_test + len(f)), 0],
                    embeddings[len_test:(len_test + len(f)), 1],
                    s=5, label='Test Cluster {}'.format(idx))
        len_test += len(f)
        plt.xlim(xlim)
        plt.ylim(ylim)
    plt.title(data_name)
    # plt.legend()
    plt.xlabel('Train Data with #Clusters {}'.format(len(X_test_list)))
    plt.show()
    plt.close()


def plot_KM(y_list, cluster_method, data_name,
            is_train=True, is_lifelines=True,
            seed=42, num_inst=1000, num_feat=200,
            is_expert=False, shape=[], scale=[], t_horizon=10):
    """This function is to plot the Kaplan-Meier curve regarding different clusters"""

    if is_train:
        stage = 'train'
    else:
        stage = 'test'

    group_indicator = []
    for idx, cluster in enumerate(y_list):
        group_indicator.append([idx] * len(cluster))
    group_indicator = np.concatenate(group_indicator)

    if is_lifelines:
        results = multivariate_logrank_test([item[1] for item in np.concatenate(y_list)], # item 1 is the survival time
                                            group_indicator,
                                            [int(item[0]) for item in np.concatenate(y_list)]) # item 0 is the event
        chisq, pval = results.test_statistic, results.p_value
    else:
        chisq, pval = compare_survival(np.concatenate(y_list), group_indicator)

    print('Test statistic of {}: {:.4e}'.format(stage, chisq))
    print('P value of {}: {:.4e}'.format(stage, pval))
    figure(figsize=(8, 6), dpi=80)
    for idx, cluster in enumerate(y_list):  # each element in the y_list is a cluster
        # use lifelines' KM tool to estimate and plot KM
        # this will provide confidence interval
        if len(cluster) == 0:
            continue
        if is_lifelines:
            kmf = KaplanMeierFitter()
            kmf.fit([item[1] for item in cluster], event_observed=[item[0] for item in cluster],
                    label='Cluster {}, #{}'.format(idx, len(cluster)))
            kmf.plot_survival_function(ci_show=False, show_censors=True)
        else:
            # use scikit-survival's KM tool to estimate and plot KM
            # this does not provide confidence interval
            x, y = kaplan_meier_estimator([item[0] for item in cluster], [item[1] for item in cluster])
            plt.step(x, y, where="post", label='Cluster {}, #{}'.format(idx, len(cluster)))

    if is_expert:
        step = 100
        for i in range(len(shape)):
            k = shape[i]
            b = scale[i]
            s = np.zeros(step)
            t_space = np.linspace(0, t_horizon, step)
            for j in range(step):
                s[j] = -(np.power(np.exp(b) * t_space[j], np.exp(k)))
            plt.plot(t_space, s, label='Expert Distribution {}'.format(i))

    plt.title("LogRank: {:.2f}".format(chisq), fontsize=18)
    plt.xlabel("Time", fontsize=18)
    plt.ylabel("Survival Probability", fontsize=18)
    plt.legend(fontsize=18)

    if data_name == 'sim':
        plt.savefig('./Figures/{}_{}_KM_plot_#clusters{}_{}_{}x{}_seed{}.png'.
                    format(cluster_method, stage, len(y_list), data_name, num_inst, num_feat, seed))
    else:
        plt.savefig('./Figures/{}_{}_KM_plot_#clusters{}_{}_seed{}.png'.
                    format(cluster_method, stage, len(y_list), data_name, seed))
    plt.show()
    plt.close()
    return pval, chisq


def get_feature_contributions(model, X, feature_names):
    feature_contributions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for r in range(model.torch_model.risks): 
        risk = str(r+1)
        for k in range(model.k):
            feature_contributions_k = []
            for i, _ in enumerate(feature_names):
                unique_feature = np.unique(X[:, i])
                feature = torch.tensor(unique_feature, dtype=torch.float32).to(device).reshape(-1, 1) 
                feature_nns = model.torch_model.alpha[risk][k]
                feat_contribution = feature_nns.feature_nns[i](feature).cpu().detach().numpy().squeeze()
                feature_contributions_k.append(feat_contribution)

            feature_contributions.append(feature_contributions_k)

    return feature_contributions


def calc_mean_prediction(model, X, feature_names):

    feature_contributions = get_feature_contributions(model, X, feature_names)
    all_indices, mean_pred = [], []
    avg_hist_data = []
    for k in range(model.k):
        feature_contributions_k = feature_contributions[k]
        
        avg_hist_data_k = {col: contributions for col, contributions in zip(feature_names, feature_contributions_k)}
        avg_hist_data.append(avg_hist_data_k)
        
        all_indices_k, mean_pred_k = {}, {}

        for col in feature_names:
            mean_pred_k[col] = np.mean([avg_hist_data_k[col]])  
        
        mean_pred.append(mean_pred_k)

    return mean_pred, avg_hist_data

def compute_mean_feature_importance(mean_pred, avg_hist_data):
    mean_abs_score = {}
    for k in avg_hist_data:
        try:
            mean_abs_score[k] = np.mean(np.abs(avg_hist_data[k] - mean_pred[k]))
        except:
            continue
    x1, x2 = zip(*mean_abs_score.items())
    return x1, x2


def plot_mean_feature_importance(model, X, feature_names, dataset_name, method, seed, width=0.5):

    mean_pred_all_k, avg_hist_data_all_k = calc_mean_prediction(model, X, feature_names)
    
    x1_all_k, x2_all_k = [], []
    importance_df_dict = {}

    for k in range(model.k):
        mean_pred = mean_pred_all_k[k]
        avg_hist_data = avg_hist_data_all_k[k]

        x1, x2 = compute_mean_feature_importance(mean_pred, avg_hist_data)
        x1_all_k.append(x1)
        x2_all_k.append(x2)

        importance_df = pd.DataFrame({
            "Feature": x1,
            "Importance": x2
        })
        
        importance_df = importance_df.sort_values(by="Importance", ascending=False)
        importance_df = importance_df.reset_index(drop=True)

        importance_df["Normalized Importance"] = importance_df["Importance"] / importance_df["Importance"].sum()
        importance_df.drop(columns=["Importance"], inplace=True)

        importance_df_dict[k] = importance_df
    
        importance_df.to_csv('./fim_feature_importance/feature_importance_seed{}_k{}_{}_{}.csv'.format(seed, k, dataset_name, 'IRIS'), index=False)
        
        cols = feature_names
        fig = plt.figure(figsize=(5, 5))
        ind = np.arange(len(x1))
        x1_indices = np.argsort(x2)

        cols_here = [cols[i] for i in x1_indices]
        x2_here = [x2[i] for i in x1_indices]

        plt.bar(ind, x2_here, width, label='FIMs')
        plt.xticks(ind + width / 2, cols_here, rotation=90, fontsize='large')
        plt.ylabel('Mean Absolute Score', fontsize='x-large')
        plt.legend(loc='upper right', fontsize='large')
        plt.title(f'Overall Importance', fontsize='x-large')
        plt.savefig('./fim_feature_importance_plots/mean_feature_importance_seed_{}_k{}_{}_{}.png'.format(seed, k, dataset_name, 'IRIS'))
        plt.show()
        plt.close()

    return fig, importance_df_dict

def plot_fims(model, X, orig_X, feature_names, dataset_name, method, seed, num_cols = 2, n_blocks = 20, color = [0.4, 0.5, 0.9], linewidth = 7.0, alpha = 1.0, feature_to_use_k = None):
    dataset = pd.DataFrame(orig_X, columns=feature_names)
    unique_feature = np.unique(X[:, 0])
    mean_pred_all_k, feat_data_contrib_all_k = calc_mean_prediction(model, X, feature_names)

    for k in range(model.k):
        print('k:', k)
        
        if dataset_name == 'ehr':
            feature_to_use = list(feature_to_use_k[k]["Feature"].head(20)) # list(feature_to_use_k[k]["Feature"].head(20))
        else:
            feature_to_use = None

        mean_pred = mean_pred_all_k[k]
        feat_data_contrib = feat_data_contrib_all_k[k]

        num_rows = math.ceil(len(feature_names) / num_cols)
        fig = plt.figure(num=None, figsize=(num_cols * 10, num_rows * 10), facecolor='w', edgecolor='k')
        fig.tight_layout(pad=7.0)

        feat_data_contrib_pairs = list(feat_data_contrib.items())
        feat_data_contrib_pairs.sort(key=lambda x: x[0])

        mean_pred_pairs = list(mean_pred.items())
        mean_pred_pairs.sort(key=lambda x: x[0])

        if feature_to_use:
            feat_data_contrib_pairs = [v for v in feat_data_contrib_pairs if v[0] in feature_to_use]
            mean_pred_pairs = [v for v in mean_pred_pairs if v[0] in feature_to_use]

        min_y = np.min([np.min(a[1] - mean_pred_pairs[i][1]) for i, a in enumerate(feat_data_contrib_pairs)])
        max_y = np.max([np.max(a[1] - mean_pred_pairs[i][1]) for i, a in enumerate(feat_data_contrib_pairs)])

        min_max_dif = max_y - min_y
        min_y = min_y - 0.05 * min_max_dif
        max_y = max_y + 0.05 * min_max_dif

        total_mean_bias = 0

        def shade_by_density_blocks(color: list = [0.9, 0.5, 0.5]):
            single_feature_data = np.array(dataset[name])
            
            x_n_blocks = min(n_blocks, len(unique_feat_data))

            segments = (max_x - min_x) / x_n_blocks
            density = np.histogram(single_feature_data, bins=x_n_blocks)
            normed_density = density[0] / np.max(density[0])
            rect_params = []

            for p in range(x_n_blocks):
                start_x = min_x + segments * p
                end_x = min_x + segments * (p + 1)
                d = min(1.0, 0.01 + normed_density[p])
                rect_params.append((d, start_x, end_x))

            for param in rect_params:
                alpha, start_x, end_x = param
                rect = patches.Rectangle(
                    (start_x, min_y - 1),
                    end_x - start_x,
                    max_y - min_y + 1,
                    linewidth=0.01,
                    edgecolor=color,
                    facecolor=color,
                    alpha=alpha,
                )
                ax.add_patch(rect)

        for i, (name, feat_contrib) in enumerate(feat_data_contrib_pairs):
            mean_pred = mean_pred_pairs[i][1]
            total_mean_bias += mean_pred    

            unique_feat_data = np.unique(np.array(dataset[name]))
            ax = plt.subplot(num_rows, num_cols, i + 1)
            
            plt.plot(unique_feat_data, feat_contrib, color=color, linewidth=linewidth, alpha=alpha)
        
            plt.xticks(fontsize='x-large')

            max_y = np.max(feat_contrib)
            min_y = np.min(feat_contrib)
            min_max_dif = max_y - min_y
            min_y = min_y - 0.05 * min_max_dif
            max_y = max_y + 0.05 * min_max_dif

            plt.ylim(min_y, max_y)
            plt.yticks(fontsize='x-large')

            min_x = np.min(unique_feat_data)  # - 0.5  ## for categorical
            max_x = np.max(unique_feat_data)  # + 0.5
            plt.xlim(min_x, max_x)

            shade_by_density_blocks()

            if i % num_cols == 0:
                plt.ylabel('Features Contribution', fontsize='x-large')

            plt.xlabel(name, fontsize='x-large')
        
        plt.savefig('./fim_plots/fim_plot_seed_{}_k{}_{}_{}.png'.format(seed, k, dataset_name, 'IRIS'))
        plt.show()
        plt.close()

