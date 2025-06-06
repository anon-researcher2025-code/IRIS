"""This module is implemented to load data.
We provide four real-world datasets: support, flchain, PBC and FRAMINGHAM.
We also provide the code for generating simulation data.
"""

from .general_utils import combine_t_e
import pandas as pd
from scipy.stats import weibull_min
import pickle as pkl
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_spd_matrix, make_low_rank_matrix
import numpy as np
from numpy.random import multivariate_normal, uniform, choice
import io
import pkgutil
from imblearn.over_sampling import RandomOverSampler, SMOTE


def load_data(args, random_state=42, task='prediction'):
    name = args.dataset
    if name == 'support':
        return load_support(random_state)
    elif name == 'flchain':
        return load_flchain_data(random_state)
    elif name == 'breastcancer':
        x, t, e, column_names = load_coxnam_dataset()
        y = combine_t_e(t, e)

        if task == 'prediction':
            X_train, X_test, y_train, y_test = \
                train_test_split(x, y, test_size=0.3, random_state=random_state, stratify=e)
        else:
            X_train, y_train = x, y
            X_temp, X_test, y_temp, y_test = \
                train_test_split(x, y, test_size=0.95, random_state=random_state, stratify=e)
            print('in clustering')
            print('x', x.shape)
            print('X_train', X_train.shape)
            print('X_test', X_test.shape)
        return X_train, X_test, y_train, y_test, column_names
    elif name == 'ehr':
        x, t, e, column_names = load_ehr_dataset()
        y = combine_t_e(t, e)
        X_train, X_test, y_train, y_test = \
            train_test_split(x, y, test_size=0.3, random_state=random_state, stratify=e)
        return X_train, X_test, y_train, y_test, column_names
    elif name == 'AV45':
        x, t, e, column_names = load_AV45_dataset()
        y = combine_t_e(t, e)
        if task == 'prediction':
            X_train, X_test, y_train, y_test = \
                train_test_split(x, y, test_size=0.3, random_state=random_state, stratify=e)
        else:
            X_train, y_train = x, y
            X_temp, X_test, y_temp, y_test = \
                train_test_split(x, y, test_size=0.95, random_state=random_state, stratify=e)
            print('in clustering')
            print('x', x.shape)
            print('X_train', X_train.shape)
            print('X_test', X_test.shape)
    
def load_coxnam_dataset():
    print('loading coxnam dataset')
    rotterdam_file = './datasets/<dataset_name>.csv.csv' # <dataset_name>.csv
    rotterdam_df = pd.read_csv(rotterdam_file)
    print('rotterdam', rotterdam_df.shape)
    print(rotterdam_df.columns)
    rotterdam_df.rename(columns={'rtime': 'time'}, inplace=True)
    rotterdam_df['status'] = rotterdam_df['nodes'].apply(lambda x: 1 if x > 0 else 0)

    gbsg_file = './datasets/<dataset_name>.csv.csv'
    gbsg_df = pd.read_csv(gbsg_file)
    print('gbsg', gbsg_df.shape)
    gbsg_df.rename(columns={'rfstime': 'time'}, inplace=True)
    
    column_names = list(gbsg_df.columns)
    print(column_names)
    rotterdam_df = rotterdam_df[column_names]

    data = rotterdam_df.copy()
    data.drop(columns=['size', 'pid'], inplace=True)
    print('data', data.shape)

    data = data[data['time'] > 0]
    time = data['time'].values
    event = data['status'].values
    data = data.drop(['time', 'status'], axis=1)
    
    column_names = data.columns
    x = data.values

    x_ = x

    return x_, time, event, column_names

def load_ehr_dataset():
    print('loading ehr dataset')

    filename = './datasets/<dataset_name>.csv'
    data = pd.read_csv(filename)
    print('data', data.shape)
    
    data = data[data['time'] > 0]
    print('data after discarding time < 0', data.shape)

    time = data['time'].values
    event = data['outcome_fall'].values
    data = data.drop(['time', 'outcome_fall'], axis=1)

    column_names = data.columns
    x = data.values

    x_ = x

    return x_, time, event, column_names


def simulate_surv(p: int, n: int, k: int, latent_dim: int, p_cens: float, seed: int, p_c=None,
                 balanced=False, clust_mean=True, clust_cov=True, isotropic=False, clust_coeffs=True,
                 clust_intercepts=True, weibull_k=1, xrange=[-5.0, 5.0], brange=[-1.0, 1.0], is_save_sim=False):
    """
    Simulates data with heterogeneous survival profiles and nonlinear (!) relationships
    (covariates are generated from latent features using an MLP decoder).
    """
    # Replicability
    np.random.seed(seed)

    # Sanity checks
    assert p > 0 and latent_dim > 0 and n > 0 and k > 0
    assert 1 < k < n
    while latent_dim > p:
        latent_dim = latent_dim // 2
    assert latent_dim < p
    assert len(xrange) == 2 and xrange[0] < xrange[1]
    assert len(brange) == 2 and brange[0] < brange[1]
    assert weibull_k > 0

    # Cluster prior prob-s
    if p_c is not None:
        assert len(p_c) == k and sum(p_c) == 1
    else:
        if balanced:
            p_c = np.ones((k,)) / k
        else:
            p_c = uniform(0, 1, (k,))
            p_c = p_c / np.sum(p_c)

    # Cluster assignments
    c = choice(a=np.arange(k), size=(n,), replace=True, p=p_c)

    # Cluster-specific means
    means = np.zeros((k, latent_dim))
    mu = uniform(xrange[0], xrange[1], (1, latent_dim))
    for l in range(k):
        if clust_mean:
            mu_l = uniform(xrange[0], xrange[1], (1, latent_dim))
            means[l, :] = mu_l
        else:
            means[l, :] = mu

    # Cluster-specific covariances
    cov_mats = []
    sigma = make_spd_matrix(latent_dim, random_state=seed)
    if isotropic:
        sigma = sigma * np.eye(latent_dim)
    for l in range(k):
        if clust_cov:
            sigma_l = make_spd_matrix(latent_dim, random_state=(seed + l))
            if isotropic:
                sigma_l = sigma_l * np.eye(latent_dim)
            cov_mats.append(sigma_l)
        else:
            cov_mats.append(sigma)

    # Latent features mutli-uniform
    low = uniform(low=xrange[0], high=0, size=k)
    high = uniform(low=0, high=xrange[1], size=k)
    Z = np.zeros((n, latent_dim))
    for l in range(k):
        n_l = np.sum(c == l)
        Z_l = uniform(low=low[l], high=high[l], size=(n_l, latent_dim))
        Z[c == l, :] = Z_l

    # Predictors
    mlp_dec = random_nonlin_map(n_in=latent_dim, n_out=p, n_hidden=int((latent_dim + p) / 2))
    X = mlp_dec(Z)
    # add noise to X
    mu, sigma = 0, 0.1
    noise = np.random.normal(mu, sigma, X.shape)
    X = X + noise

    # Cluster-specific coefficients for the survival model
    coeffs = np.zeros((k, latent_dim))
    intercepts = np.zeros((k,))
    beta = uniform(brange[0], brange[1], (1, latent_dim))
    beta0 = uniform(brange[0], brange[1], (1, 1))
    for l in range(k):
        if clust_coeffs:
            beta_l = uniform(brange[0], brange[1], (1, latent_dim))
            coeffs[l, :] = beta_l
        else:
            coeffs[l, :] = beta
        if clust_intercepts:
            beta0_l = uniform(brange[0], brange[1], (1, 1))
            intercepts[l] = beta0_l
        else:
            intercepts[l] = beta0

    # Survival times
    t = np.zeros((n,))
    for l in range(k):
        n_l = np.sum(c == l)
        Z_l = Z[c == l, :]
        coeffs_l = np.expand_dims(coeffs[l, :], 1)
        intercept_l = intercepts[l]
        logexps_l = np.log(1 + np.exp(intercept_l + np.squeeze(np.matmul(Z_l, coeffs_l))))

        t_l = weibull_min.rvs(weibull_k, loc=0, scale=logexps_l, size=n_l)

        t[c == l] = t_l

    # add noise to t
    mu, sigma = 0, 0.1
    noise = np.random.normal(mu, sigma, t.shape)
    t = t + np.abs(noise)
    # Censoring
    # NB: d == 1 if failure; 0 if censored
    d = (uniform(0, 1, (n,)) >= p_cens) * 1.0
    t_cens = uniform(0, t, (n,))
    t[d == 0] = t_cens[d == 0]

    sim_data = {}
    sim_data['X'] = X
    sim_data['t'] = t
    sim_data['e'] = d
    sim_data['c'] = c
    sim_data['Z'] = Z

    if is_save_sim:
        with open('datasets/sim_data_({},{}).pkl'.format(n, p), 'wb') as f:
            pkl.dump(sim_data, f)

    return X, t, d, c, Z, mlp_dec, means, cov_mats, coeffs, intercepts


def ReLU(x):
    return x * (x > 0)


def random_nonlin_map(n_in, n_out, n_hidden, rank=1000):
    # Random MLP mapping
    W_0 = make_low_rank_matrix(n_in, n_hidden, effective_rank=rank)
    W_1 = make_low_rank_matrix(n_hidden, n_hidden, effective_rank=rank)
    W_2 = make_low_rank_matrix(n_hidden, n_out, effective_rank=rank)
    # Disabled biases for now...
    b_0 = np.random.uniform(0, 0, (1, n_hidden))
    b_1 = np.random.uniform(0, 0, (1, n_hidden))
    b_2 = np.random.uniform(0, 0, (1, n_out))

    nlin_map = lambda x: np.matmul(ReLU(np.matmul(ReLU(np.matmul(x, W_0) + np.tile(b_0, (x.shape[0], 1))),
                                                       W_1) + np.tile(b_1, (x.shape[0], 1))), W_2) + \
                         np.tile(b_2, (x.shape[0], 1))

    return nlin_map


def load_support(random_state):
    x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, column_names = \
        generate_support(seed=random_state)
    y_train = combine_t_e(t_train, d_train)
    y_test = combine_t_e(t_test, d_test)
    return x_train, x_test, y_train, y_test, column_names


def generate_data(seed=42):
    np.random.seed(seed)
    data = pkgutil.get_data(__name__, '../datasets/support2.csv')
    data_frame = pd.read_csv(io.BytesIO(data))

    print("data shape:{}".format(data_frame.shape))

    to_drop = ['hospdead', 'death', 'prg2m', 'prg6m', 'dnr', 'dnrday', 'd.time', 'aps', 'sps', 'surv2m', 'surv6m',
               'totmcst']
    
    # Preprocess
    one_hot_encoder_list = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'sfdm2']
    data_frame = one_hot_encoder(data=data_frame, encode=one_hot_encoder_list)

    print("data shape after one_hot_encoding:{}".format(data_frame.shape))

    data_frame = log_transform(data_frame, transform_ls=['totmcst', 'totcst', 'charges', 'pafi', 'sod'])
    t_data = data_frame[['d.time']]
    e_data = data_frame[['death']]

    x_data = data_frame.drop(labels=to_drop, axis=1)

    column_names = list(x_data.columns)

    encoded_indices = one_hot_indices(x_data, one_hot_encoder_list)
    include_idx = set(np.array(sum(encoded_indices, [])))
    mask = np.array([(i in include_idx) for i in np.arange(x_data.shape[1])])
    
    x = np.array(x_data).reshape(x_data.shape)
    t = np.array(t_data).reshape(len(t_data))
    e = np.array(e_data).reshape(len(e_data))
    
    idx = np.arange(0, x.shape[0])

    np.random.shuffle(idx)
    x = x[idx]
    t = t[idx]
    e = e[idx]

    # Normalization
    t = t / np.max(t) + 0.001
    scaler = StandardScaler()
    scaler.fit(x[:, ~mask])
    x[:, ~mask] = scaler.transform(x[:, ~mask])
    
    num_examples = int(0.80 * len(e))
    train_idx = idx[0: num_examples]
    split = int((len(t) - num_examples) / 2)

    test_idx = idx[num_examples: num_examples + split]
    valid_idx = idx[num_examples + split: len(t)]

    imputation_values = get_train_median_mode(x=x[train_idx], categorial=encoded_indices)

    preprocessed = {
        'train': formatted_data(x=x, t=t, e=e, idx=train_idx, imputation_values=imputation_values),
        'test': formatted_data(x=x, t=t, e=e, idx=test_idx, imputation_values=imputation_values),
        'valid': formatted_data(x=x, t=t, e=e, idx=valid_idx, imputation_values=imputation_values)
    }

    return preprocessed, column_names


def generate_support(seed=42):
    preproc, column_names = generate_data(seed)

    x_train = preproc['train']['x']
    x_valid = preproc['valid']['x']
    x_test = preproc['test']['x']

    t_train = preproc['train']['t']
    t_valid = preproc['valid']['t']
    t_test = preproc['test']['t']

    d_train = preproc['train']['e']
    d_valid = preproc['valid']['e']
    d_test = preproc['test']['e']

    return x_train, x_valid, x_test, t_train, t_valid, t_test, d_train, d_valid, d_test, column_names


def load_flchain_data(random_state):
    data = pkgutil.get_data(__name__, '../datasets/flchain.csv')
    data = pd.read_csv(io.BytesIO(data))
    feats = ['age', 'sex', 'sample.yr', 'kappa', 'lambda', 'flc.grp', 'creatinine', 'mgus']
    prot = 'sex'
    feats = set(feats)
    feats = list(feats)  # - set([prot]))
    t = data['futime'].values + 1
    d = data['death'].values
    x = data[feats].values
    c = data[prot].values
    
    X = x
    t = t / np.max(t) + 0.001
    x_train, x_test, t_train, t_test, d_train, d_test, c_train, c_test = train_test_split(X, t, d, c, test_size=.3,
                                                                                          random_state=random_state)
    y_train = combine_t_e(t_train, d_train)
    y_test = combine_t_e(t_test, d_test)
    return x_train, x_test, y_train, y_test, feats


def one_hot_encoder(data, encode):
    # print("Encoding data:{}".format(data.shape))
    data_encoded = data.copy()
    encoded = pd.get_dummies(data_encoded, prefix=encode, columns=encode)
    # print("head of data:{}, data shape:{}".format(data_encoded.head(), data_encoded.shape))
    # print("Encoded:{}, one_hot:{}{}".format(encode, encoded.shape, encoded[0:5]))
    return encoded


def log_transform(data, transform_ls):
    dataframe_update = data

    def transform(x):
        constant = 1e-8
        transformed_data = np.log(x + constant)
        # print("max:{}, min:{}".format(np.max(transformed_data), np.min(transformed_data)))
        return np.abs(transformed_data)

    for column in transform_ls:
        df_column = dataframe_update[column]
        # print(" before log transform: column:{}{}".format(column, df_column.head()))
        # print("stats:max: {}, min:{}".format(df_column.max(), df_column.min()))
        dataframe_update[column] = dataframe_update[column].apply(transform)
        # print(" after log transform: column:{}{}".format(column, dataframe_update[column].head()))
    return dataframe_update


def formatted_data(x, t, e, idx, imputation_values=None):
    death_time = np.array(t[idx], dtype=float)
    censoring = np.array(e[idx], dtype=float)
    covariates = np.array(x[idx])

    # for i, value in enumerate(x[:, 0]): 
    #     if not isinstance(value, (int, float, np.number)):
    #         print(f"Non-numeric value at index {i}: {value}")
    #     else:
    #         print('i in formatted_data', i)

    if imputation_values is not None:
        impute_covariates = impute_missing(data=covariates, imputation_values=imputation_values)
    else:
        impute_covariates = x
    survival_data = {'x': impute_covariates, 't': death_time, 'e': censoring}
    assert np.sum(np.isnan(impute_covariates)) == 0
    return survival_data


def get_train_median_mode(x, categorial):
    categorical_flat = flatten_nested(categorial)
    # print("categorical_flat:{}".format(categorical_flat))
    imputation_values = []
    # print("len covariates:{}, categorical:{}".format(x.shape[1], len(categorical_flat)))
    median = np.nanmedian(x, axis=0)
    mode = []
    for idx in np.arange(x.shape[1]):
        a = x[:, idx]
        (_, idx, counts) = np.unique(a, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode_idx = a[index]
        mode.append(mode_idx)
    for i in np.arange(x.shape[1]):
        if i in categorical_flat:
            imputation_values.append(mode[i])
        else:
            imputation_values.append(median[i])
    # print("imputation_values:{}".format(imputation_values))
    return imputation_values


def missing_proportion(dataset):
    missing = 0
    columns = np.array(dataset.columns.values)
    for column in columns:
        missing += dataset[column].isnull().sum()
    return 100 * (missing / (dataset.shape[0] * dataset.shape[1]))


def one_hot_indices(dataset, one_hot_encoder_list):
    indices_by_category = []
    for colunm in one_hot_encoder_list:
        values = dataset.filter(regex="{}_.*".format(colunm)).columns.values
        # print("values:{}".format(values, len(values)))
        indices_one_hot = []
        for value in values:
            indice = dataset.columns.get_loc(value)
            # print("column:{}, indice:{}".format(colunm, indice))
            indices_one_hot.append(indice)
        indices_by_category.append(indices_one_hot)
    # print("one_hot_indices:{}".format(indices_by_category))
    return indices_by_category


def flatten_nested(list_of_lists):
    flattened = [val for sublist in list_of_lists for val in sublist]
    return flattened


def print_missing_prop(covariates):
    missing = np.array(np.isnan(covariates), dtype=float)
    shape = np.shape(covariates)
    proportion = np.sum(missing) / (shape[0] * shape[1])
    # print("missing_proportion:{}".format(proportion))


def impute_missing(data, imputation_values):
    data = np.array(data, dtype=float)
    copy = data
    for i in np.arange(len(data)):
        # print('i', i)
        row = data[i]
        # print('type(row)', type(row), type(row[0]))
        indices = np.isnan(row)

        for idx in np.arange(len(indices)):
            if indices[idx]:
                # print("idx:{}, imputation_values:{}".format(idx, np.array(imputation_values)[idx]))
                copy[i][idx] = imputation_values[idx]
    # print("copy;{}".format(copy))
    return copy

def load_AV45_dataset():
    print('loading AV45 dataset')

    filename = './datasets/<dataset_name>.csv'  # Replace with the actual dataset file path
    data = pd.read_csv(filename)
    print('data', data.shape)
    data = data.drop(['Unnamed: 0', 'ID'], axis=1)
    
    data = data[data['time'] > 0]
    print('data after discarding time < 0', data.shape)

    time = data['time'].values
    event = data['label'].values
    data = data.drop(['time', 'label'], axis=1)
    
    column_names = data.columns
    x = data.values

    x_ = x

    return x_, time, event, column_names