import numpy as np
from utils.plotting import plot_KM, plot_Weibull_cdf, plot_mean_feature_importance, plot_fims
from utils.data_utils import load_data
from utils.general_utils import train_test_IRIS, test_IRIS, sample_weibull, calibration
import time
import argparse
import scipy.stats as st
import torch
from sklearn.preprocessing import StandardScaler
import pickle as pkl

def init_config():
    parser = argparse.ArgumentParser(description='Interpretable Risk Clustering Intelligence for Survival Analysis')
    # model hyper-parameters
    parser.add_argument('--dataset', type=str, default='breastcancer',
                        help='dataset in [breastcancer, support, flchain, AV45, ehr]') 
    parser.add_argument('--is_normalize', type=bool, default=True, help='whether to normalize data')
    parser.add_argument('--is_cluster', type=bool, default=True, help='whether to use IRIS to do clustering')
    parser.add_argument('--is_generate_sim', type=bool, default=True, help='whether we generate simulation data')
    parser.add_argument('--is_save_sim', type=bool, default=True, help='whether we save simulation data')
    parser.add_argument('--num_inst', default=200, type=int,
                        help='specifies the number of instances for simulation data')
    parser.add_argument('--num_feat', default=10, type=int,
                        help='specifies the number of features for simulation data')
    parser.add_argument('--cuda_device', default=0, type=int,
                        help='specifies the index of the cuda device')
    parser.add_argument('--discount', default=0.5, type=float, help='specifies number of discount parameter')
    parser.add_argument('--weibull_shape', default=2, type=int, help='specifies the Weibull shape')
    parser.add_argument('--num_cluster', default=2, type=int, help='specifies the number of clusters')
    parser.add_argument('--train_IRIS', default=True, type=bool, help='whether to train IRIS')

    args = parser.parse_args()
    parser.print_help()
    return args

start_time = time.perf_counter()
print('start time is: ', start_time)

args = init_config()  # input params from command
torch.cuda.set_device(args.cuda_device)  # set cuda device
data_name = args.dataset

result_IRIS = []
logrank_IRIS = []
rae_nc_IRIS_list = []
rae_c_IRIS_list = []
cal_IRIS_list = []
brier_score_IRIS_list = []

# normalization
is_normalized = args.is_normalize

########################################
#      Train and Test Models
########################################

# this may not be optimal 

if args.dataset == 'breastcancer':
    param = {'learning_rate': 0.00001, 'layers': [50, 50], 'k': 2, # 0.0008303256995946264
            'iters': 1000, 'distribution': 'Weibull', 'discount': 0.5}
elif args.dataset == 'support':
    param = {'learning_rate': 0.0001, 'layers': [100], 'k': 2, # 0.006592097947267537
            'iters': 1000, 'distribution': 'Weibull', 'discount': 0.5}
elif args.dataset == 'flchain':
    param = {'learning_rate': 0.00010147923472896853, 'layers': [50], 'k': 2, # 0.00010147923472896853
            'iters': 1000, 'distribution': 'Weibull', 'discount': 0.5}
elif args.dataset == 'AV45':
    param = {'learning_rate': 0.00001, 'layers': [100], 'k': 2, # 9.045214001002122e-05
            'iters': 1000, 'distribution': 'Weibull', 'discount': 0.5}
elif args.dataset == 'ehr':
    param = {'learning_rate': 0.00001, 'layers': [50, 50], 'k': 2, # 2.1239642407639025e-05
            'iters': 1000, 'distribution': 'Weibull', 'discount': 0.5}

# hold out testing with different splitting using the same parameters set
for seed in [42, 73, 666, 777, 1009]:
# for seed in [73]:
    X_train, X_test, y_train, y_test, column_names = load_data(args, random_state=seed, task='prediction')

    print('-------------------------dataset: {}, train shape: {}, seed {}-----------------'
          .format(data_name, X_train.shape, seed))
    e_train = np.array([[item[0] * 1 for item in y_train]]).T
    t_train = np.array([[item[1] for item in y_train]]).T
    e_test = np.array([[item[0] * 1 for item in y_test]]).T
    t_test = np.array([[item[1] for item in y_test]]).T

    orig_X_train = X_train
    orig_X_test = X_test

    if is_normalized:
        print('Data are normalized')
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
    else:
        print('Data are not normalized')

    if args.train_IRIS:
        print('-----------------------------train and test IRIS-------------------------------')

        fix = True
        if fix:
            method = 'IRIS'
        else:
            method = 'DSM'
        model, c_index, pred_IRIS, pred_time_IRIS, rae_nc_IRIS, rae_c_IRIS, brier_score_test \
            = train_test_IRIS(param, X_train, X_test, orig_X_test, y_train, y_test, column_names, data_name, optuna=False, seed=seed, fix=fix, method=method)
        
        with open('models/IRIS_{}_seed{}.pkl'.format(data_name, seed), 'wb') as file:
            pkl.dump(model, file)
    else:
        print('-----------------------------just test IRIS-------------------------------')
        with open('models/IRIS_{}_seed{}.pkl'.format(data_name, seed), 'rb') as file:
            model = pkl.load(file)
        c_index, pred_IRIS, pred_time_IRIS, rae_nc_IRIS, rae_c_IRIS = test_IRIS(model, X_test, y_test)

    print('IRIS_c_index on all test data: {:.4f}'.format(c_index))
    result_IRIS.append(c_index)
    print('IRIS_brier_score on test data: {:.4f}'.format(brier_score_test))
    brier_score_IRIS_list.append(brier_score_test)

    is_cluster = args.is_cluster
    if is_cluster:
        print('----------------------cluster data with IRIS------------------------')

        X_train, X_test, y_train, y_test, column_names = load_data(args, random_state=seed, task='clustering')

        print('-------------------------dataset: {}, train shape: {}, seed {}-----------------'
            .format(data_name, X_train.shape, seed))
        e_train = np.array([[item[0] * 1 for item in y_train]]).T
        t_train = np.array([[item[1] for item in y_train]]).T
        e_test = np.array([[item[0] * 1 for item in y_test]]).T
        t_test = np.array([[item[1] for item in y_test]]).T

        orig_X_train = X_train
        orig_X_test = X_test

        if is_normalized:
            print('Data are normalized')
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
        else:
            print('Data are not normalized')

        fix = True
        if fix:
            method = 'IRIS'
        else:
            method = 'DSM'
        model, c_index, pred_IRIS, pred_time_IRIS, rae_nc_IRIS, rae_c_IRIS, brier_score_test \
            = train_test_IRIS(param, X_train, X_test, orig_X_test, y_train, y_test, column_names, data_name, optuna=False, seed=seed, fix=fix, method=method)

        fig, importance_df_dict = plot_mean_feature_importance(model, X_test, column_names, data_name, method, seed)
        
        if data_name == 'ehr':
            feature_to_use = importance_df_dict
        else:
            feature_to_use = None
        
        plot_fims(model, X_test, orig_X_test, column_names, data_name, method, seed, num_cols=3, feature_to_use_k=feature_to_use)
        
        cluster_tags_IRIS, shape, scale = model.predict_phenotype(np.float64(X_test))
        shape = shape.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()
        X_test_list = []
        y_test_list = []
        for i in range(model.k):  # go through all the classes
            idx_i = np.where(cluster_tags_IRIS == i)[0]
            print('num in cluster {} is {}'.format(i, len(idx_i)))
            X_test_list.append(X_test[idx_i])
            y_test_list.append(y_test[idx_i])
        pval, logrank = plot_KM(y_test_list, 'IRIS', data_name, is_train=False,
                seed=seed, num_inst=args.num_inst, num_feat=args.num_feat,
                is_expert=False, shape=shape, scale=scale, t_horizon=t_test.max())
        
        print('logrank p-value: {:.4f}'.format(logrank))
        
        logrank_IRIS.append(logrank)
        plot_Weibull_cdf(t_test.max(), shape, scale, data_name=data_name, seed=seed)

low_IRIS, high_IRIS = st.t.interval(confidence=0.95, df=len(result_IRIS)-1, loc=np.mean(result_IRIS), scale=st.sem(result_IRIS))
print('-----------------C Index results-----------------')
print('IRIS:{:.4f}±{:.4f} from {:.4f} to {:.4f}'.format(np.mean(result_IRIS), np.std(result_IRIS), low_IRIS, high_IRIS))
low_IRIS, high_IRIS = st.t.interval(confidence=0.95, df=len(logrank_IRIS)-1, loc=np.mean(logrank_IRIS), scale=st.sem(logrank_IRIS))
print('---------------logrank results-----------------')
print('IRIS:{:.4f}±{:.4f} from {:.4f} to {:.4f}'.format(np.mean(logrank_IRIS), np.std(logrank_IRIS), low_IRIS, high_IRIS))
print('---------------brier_score results---------------------')
print('IRIS:{:.4f}±{:.4f}'.format(np.mean(brier_score_IRIS_list), np.std(brier_score_IRIS_list)))

e = int(time.perf_counter() - start_time)
print('Elapsed Time: {:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
