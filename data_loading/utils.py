import datetime
from logging import root
# from models.SupervisedAutoEncoder import create_hidden_rep
from operator import mod
import os
import random
import re
from typing import List, Tuple
import dotenv
import datatable as dt
from dotenv.main import load_dotenv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from torch.utils.data import Dataset, Subset, BatchSampler, SequentialSampler, DataLoader, Sampler
import time
from joblib import Parallel, delayed
import gc
import xgboost as xgb
from numerapi import NumerAPI
import numerai_utils as n_utils
import json


class FinData(Dataset):
    def __init__(self, data, target, era, hidden=None, mode='train', transform=None, cache_dir=None, ordinal=False):
        self.data = data
        self.target = target
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir
        self.era = era
        self.hidden = hidden
        self.ordinal = ordinal

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index.to_list()
        if self.transform:
            return self.transform(self.data.iloc[index].values)
        else:
            if type(self.data) is not np.ndarray:
                if self.ordinal:
                    sample = {
                        'target': torch.Tensor(self.target[index].values),
                        'data':   torch.Tensor(self.data.loc[index].values),
                        'era':    torch.Tensor(self.era[index].values),
                    }
                else:
                    sample = {
                        'target': torch.LongTensor(self.target[index].values),
                        'data':   torch.Tensor(self.data.loc[index].values),
                        'era':    torch.Tensor(self.era[index].values),
                    }

            else:
                if self.ordinal:
                    sample = {
                        'target': torch.Tensor(self.target[index].values),
                        'data':   torch.Tensor(self.data[index]),
                        'era':    torch.Tensor(self.era[index].values),
                    }
                else:
                    sample = {
                        'target': torch.LongTensor(self.target[index].values),
                        'data':   torch.Tensor(self.data[index]),
                        'era':    torch.Tensor(self.era[index].values),
                    }
            if self.hidden is not None:
                sample['hidden'] = torch.Tensor(self.hidden[index])
        return sample

    def __len__(self):
        return len(self.data)


def get_data_path(root_dir, old=False):
    dotenv_path = 'num_config.env'
    load_dotenv(dotenv_path=dotenv_path)
    curr_round = os.getenv('LATEST_ROUND')
    if old:
        data_path = root_dir + '/numerai_dataset_' + str(curr_round)
    else:
        data_path = root_dir + '/round_' + str(curr_round)
    return data_path


def load_data(root_dir, mode, overide=None, old=False):
    print(f'loading {mode} data')
    data_path = get_data_path(root_dir=root_dir)
    # for legacy purposes
    if old:
        if overide:
            data = dt.fread(overide).to_pandas()
        elif mode == 'train':
            data = dt.fread(
                data_path + '/numerai_training_data.csv').to_pandas()
        elif mode == 'test':
            data = dt.fread(
                data_path + '/numerai_tournament_data.csv').to_pandas()
        return data
    else:
        if overide:
            data = pd.read_parquet(overide)
        elif mode == 'train':
            data = pd.read_parquet(
                root_dir + '/datasets/numerai_training_data.parquet')
        elif mode == 'test':
            data = pd.read_parquet(
                data_path + '/numerai_tournament_data.parquet')
        return data


def reduce_mem(df):
    """ iterate through all the columns of a dataframe and modify the data type
            to reduce memory usage.
        https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
        """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(
        100 * (start_mem - end_mem) / start_mem))

    return df


def preprocess_data(data: pd.DataFrame, scale: bool = False, nn: bool = False, test=False, ordinal=False):
    """
    Preprocess the data.

    Parameters
    ----------
    data
        Pandas DataFrame
    scale
        scale data with unit std and 0 mean
    nn
        return data as np.array
    missing
        options to replace missing data with - mean, median, 0
    action
        options to create action value  - weight = (weight * resp) > 0
                                        - combined = (resp_cols) > 0
                                        - multi = each resp cols >0

    Returns
    -------
    """
    # data = reduce_mem(data)
    features = [col for col in data.columns if 'feature' in col]
    era = data['era']
    era = era.transform(lambda x: re.sub('[a-z]', '', x))
    if not test:
        era = era.astype('int')
    if ordinal:
        oe = LabelEncoder()
        data['target'] = oe.fit_transform(data.target.values)
    target = data['target']
    data = data[features]
    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)

    # data = data.values
    if nn:
        data = data.values
    return data, target, features, era


def calc_data_mean(array, cache_dir=None, fold=None, train=True, mode='mean'):
    if train:
        if mode == 'mean':
            f_mean = np.nanmean(array, axis=0)
            if cache_dir and fold:
                np.save(f'{cache_dir}/f_{fold}_mean.npy', f_mean)
            elif cache_dir:
                np.save(f'{cache_dir}/f_mean.npy', f_mean)
            array = np.nan_to_num(array) + np.isnan(array) * f_mean
        if mode == 'median':
            f_med = np.nanmedian(array, axis=0)
            if cache_dir and fold:
                np.save(f'{cache_dir}/f_{fold}_median.npy', f_med)
            elif cache_dir:
                np.save(f'{cache_dir}/f_median.npy', f_med)
            array = np.nan_to_num(array) + np.isnan(array) * f_med
        if mode == 'zero':
            array = np.nan_to_num(array) + np.isnan(array) * 0
    if not train:
        if mode == 'mean':
            f_mean = np.load(f'{cache_dir}/f_mean.npy')
            array = np.nan_to_num(array) + np.isnan(array) * f_mean
        if mode == 'median':
            f_med = np.load(f'{cache_dir}/f_med.npy')
            array = np.nan_to_num(array) + np.isnan(array) * f_med
        if mode == 'zero':
            array = np.nan_to_num(array) + np.isnan(array) * 0
    return array


def weighted_mean(scores, sizes=0):
    """largest = np.max(sizes)
    weights = [size / largest for size in sizes]
    return np.average(scores, weights=weights)"""
    w = []
    n = len(scores)
    for j in range(1, n + 1):
        j = 2 if j == 1 else j
        w.append(1 / (2 ** (n + 1 - j)))
    # return np.average([score.reshape(-1) for score in scores], weights=w)
    for i in range(len(scores)):
        scores[i] = scores[i] * w[i]
    return np.mean([np.sum(scores, axis=0)], 0)


def create_dataloaders(dataset: Dataset, indexes: dict, batch_size):
    train_idx = indexes.get('train', None)
    val_idx = indexes.get('val', None)
    test_idx = indexes.get('test', None)
    dataloaders = {}
    if type(train_idx) != list:
        train_idx = train_idx.tolist()
    if type(val_idx) != list:
        val_idx = val_idx.tolist()
    if type(test_idx) != list and test_idx is not None:
        test_idx = test_idx.tolist()
    if train_idx:
        # train_set = Subset(dataset, train_idx)
        train_sampler = EraSampler(
            data_source=dataset, indices=train_idx, shuffle=False)
        dataloaders['train'] = DataLoader(
            dataset, sampler=train_sampler, num_workers=10, pin_memory=True, shuffle=False)
    if val_idx:
        # val_set = Subset(dataset, val_idx)
        val_sampler = EraSampler(
            data_source=dataset, indices=val_idx, shuffle=False)
        dataloaders['val'] = DataLoader(
            dataset, sampler=val_sampler, num_workers=10, pin_memory=True, shuffle=False)
    if test_idx:
        # test_set = Subset(dataset, test_idx)
        test_sampler = EraSampler(data_source=dataset[test_idx], shuffle=False)
        dataloaders['test'] = DataLoader(
            dataset, sampler=test_sampler, num_workers=10, pin_memory=True, shuffle=False)
    return dataloaders


def seed_everything(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model(path, p, pl_lightning, model):
    Classifier = model
    if os.path.isdir(path):
        models = []
        for file in os.listdir(path):
            if pl_lightning:
                model = Classifier.load_from_checkpoint(
                    checkpoint_path=path + '/' + file, params=p)
            else:
                model = Classifier(params=p)
                model.load_state_dict(torch.load(path + '/' + file))
            models.append(model)
        return models
    elif os.path.isfile(path):
        if pl_lightning:
            return Classifier.load_from_checkpoint(checkpoint_path=path, params=p)
        else:
            model = Classifier(params=p)
            model.load_state_dict(torch.load(path))
            return model


def init_weights(m, func):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, nn.init.calculate_gain(func))


def create_predictions(root_dir: str = './data', models: dict = {}, hidden=True, ae=True, val=False):
    """test_files_path, test_files_exist = check_test_files(root_dir)
    if test_files_exist:
        test_files = os.listdir(test_files_path)
    for file in test_files:
        df = load_data(root_dir='./data', mode='test',
                       overide=f'{test_files_path}/{file}')
        df['target'] = 0
        data, target, features, era = preprocess_data(data=df, nn=True)
        t_idx = np.arange(start=0, stop=len(era), step=1).tolist()
        data_dict = data_dict = {'data':     data, 'target': target,
                                 'features': features, 'era': era}
        preds = {k: [] for k in models.keys()}
        for key, val in models.items():
            for i, model in enumerate(val):
                if key == 'xgb':
                    d = xgb.DMatrix(data)
                    preds[key].append(model.predict(d).squeeze())
                    del d
                elif key == 'lgb':
                    preds[key].append(model.predict(data).squeeze())
                elif key == 'ae':
                    model.eval()
                    d = torch.Tensor(data)
                    _, _, preds_ae, _, _ = model(d)
                    preds[key].append(preds_ae.detach().numpy().squeeze())
                if key == 'resnet':
                    model.eval()
                    d = torch.Tensor(data)
                    pred = model(d)
                    preds[key].append(pred.detach().numpy().squeeze())
                if key == 'cat':
                    preds[key].append(model.predict(data).squeeze())
        for key in preds:
            temp = []
            temp.append(weighted_mean(preds[key][:10]))
            temp.append(weighted_mean(preds[key][10:20]))
            temp.append(weighted_mean(preds[key][20:30]))
            preds[key] = np.mean([temp], axis=0)
        preds_stacked = np.vstack([pred for pred in preds.values()])

        df['prediction'] = np.mean(preds_stacked, 0)
        df = df[['id', 'prediction']]
        if not os.path.isdir(f'{get_data_path(root_dir)}/predictions/'):
            os.makedirs(f'{get_data_path(root_dir)}/predictions')
        pred_path = f'{get_data_path(root_dir)}/predictions/{era[0]}'
        df.to_csv(f'{pred_path}.csv')
    """

    if val:
        df = pd.read_parquet(
            root_dir + '/numerai_dataset/numerai_validation_data.parquet')
        example_pred = pd.read_parquet(
            root_dir + '/numerai_dataset/example_validation_predictions.parquet')
        df['example_preds'] = example_pred['prediction']
    else:
        data_path = check_tournament_data(root_dir)
        df = pd.read_parquet(data_path)
        df['target'] = 0
        df['era'][df['era'] == 'eraX'] = 'era999'
    data, target, features, era = preprocess_data(data=df, nn=True)
    t_idx = np.arange(start=0, stop=len(era), step=1).tolist()
    data_dict = data_dict = {'data':     data, 'target': target,
                             'features': features, 'era': era}
    pred_cols = set()
    for model_type, models_dict in models.items():
        for model in models_dict.keys():
            if model_type == 'xgb':
                d = xgb.DMatrix(data)
                df[model] = models_dict[model].predict(d).squeeze()
                del d
            elif model_type == 'lgb':
                df[model] = models_dict[model].predict(data).squeeze()
            elif model_type == 'ae':
                ae = models_dict[model]
                ae.eval()
                d = torch.Tensor(data)
                _, _, preds_ae, _, _ = ae(d)
                df[model] = preds_ae.detach().numpy().squeeze()
            if model_type == 'resnet':
                resnet = models_dict[model]
                resnet.eval()
                d = torch.Tensor(data)
                pred = resnet(d)
                df[model] = pred.detach().numpy().squeeze()
            if model_type == 'cat':
                df[model] = models_dict[model].predict(data).squeeze()
            pred_cols.add(model)

    if val:
        df['ensemble_all'] = sum(df[pred_col]
                                 for pred_col in pred_cols).rank(pct=True)
        val_stats = n_utils.validation_metrics(validation_data=df, pred_cols=['ensemble_all'],
                                               example_col='example_preds')
        print(val_stats.to_markdown())
        val_stats.to_csv(
            f'./data/numerai_dataset/val_stats{datetime.date.today()}.csv')


def check_test_files(root_dir='./data'):
    """
    Deprecated but may be useful for low memory implementations
    """
    data_path = get_data_path(root_dir)
    print(data_path)
    test_files_path = f'{data_path}/test_files'
    # TODO Check to make sure all era's present
    if os.path.isdir(test_files_path):
        return test_files_path, True
    else:
        os.makedirs(test_files_path)
        df = load_data(root_dir=root_dir, mode='test')
        df['era'][df['era'] == 'eraX'] = 'era999'
        for era in df['era'].unique():
            path = f'{test_files_path}/{era}'
            df[df['era'] == era].to_csv(f'{path}.csv')
        return test_files_path, True


def check_tournament_data(root_dir='./data'):
    napi = NumerAPI()
    current_round = napi.get_current_round()
    tournament_data_path = root_dir + \
                           f'/numerai_dataset/numerai_tournament_data_{current_round}.parquet'
    if not os.path.isfile(path=tournament_data_path):
        n_utils.download_data(napi, filename='numerai_torunament_data.parquet',
                              dest_path=tournament_data_path)
    return tournament_data_path


def create_prediction_file(root_dir='./data', eras=None):
    pred_path = f'{get_data_path(root_dir)}/predictions/'
    files = os.listdir(pred_path)
    files.sort()
    if eras:
        dfs = [pd.read_csv(f'{pred_path}{file}')
               for file in files if file != 'predictions.csv' and file in eras]
    else:
        dfs = [pd.read_csv(f'{pred_path}{file}')
               for file in files if file != 'predictions.csv']
    df = pd.concat(dfs)
    df = df[['id', 'prediction']]
    df.columns = ['id', 'prediction']
    df_test = load_data(root_dir=root_dir, mode='test')
    df = df_test.merge(df, on='id')
    df = df[['id', 'prediction']]
    df.to_csv(f'{pred_path}predictions.csv', index=False)
    return pred_path


def read_api_token(path: str = 'neptune_api_token.txt'):
    with open(path) as f:
        token = f.readline()
    return token


def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]


# convenience method for scoring
def scoring(df):
    return correlation(df['prediction'], df['target'])


# Payout is just the score cliped at +/-25%
def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)


def calculate_sharpe_ratio(preds, target, era):
    df = pd.DataFrame({'pred': preds, 'target': target, 'era': era})
    score_per_era = df.groupby('era')[['pred', 'target']].apply(
        lambda x: correlation(x[0], x[1]))
    sharpe_ratio = score_per_era.mean() / score_per_era.std()
    return sharpe_ratio


def calculate_max_drawdown(preds, target, era):
    df = pd.DataFrame({'pred': preds, 'target': target, 'era': era})
    score_per_era = df.groupby('era')[['pred', 'target']].apply(
        lambda x: correlation(x[0], x[1]))
    rolling_max = (score_per_era +
                   1).cumprod().rolling(window=100, min_periods=1).max()
    daily_values = (score_per_era + 1).cumprod()
    max_drawdown = (rolling_max - daily_values).max()
    return max_drawdown


"""
Author: Kirgsn, 2018
https://www.kaggle.com/wkirgsn/fail-safe-parallel-memory-reduction/comments
"""


def measure_time_mem(func):
    def wrapped_reduce(self, df, *args, **kwargs):
        # pre
        mem_usage_orig = df.memory_usage().sum() / self.memory_scale_factor
        start_time = time.time()
        # exec
        ret = func(self, df, *args, **kwargs)
        # post
        mem_usage_new = ret.memory_usage().sum() / self.memory_scale_factor
        end_time = time.time()
        print(f'reduced df from {mem_usage_orig:.4f} MB '
              f'to {mem_usage_new:.4f} MB '
              f'in {(end_time - start_time):.2f} seconds')
        gc.collect()
        return ret

    return wrapped_reduce


class Reducer:
    """
    Class that takes a dict of increasingly big numpy datatypes to transform
    the data of a pandas dataframe into, in order to save memory usage.
    """
    memory_scale_factor = 1024 ** 2  # memory in MB

    def __init__(self, conv_table=None, use_categoricals=True, n_jobs=-1):
        """
        :param conv_table: dict with np.dtypes-strings as keys
        :param use_categoricals: Whether the new pandas dtype "Categoricals"
                shall be used
        :param n_jobs: Parallelization rate
        """

        self.conversion_table = \
            conv_table or {'int':   [np.int8, np.int16, np.int32, np.int64],
                           'uint':  [np.uint8, np.uint16, np.uint32, np.uint64],
                           'float': [np.float32, ]}
        self.use_categoricals = use_categoricals
        self.n_jobs = n_jobs

    def _type_candidates(self, k):
        for c in self.conversion_table[k]:
            i = np.iinfo(c) if 'int' in k else np.finfo(c)
            yield c, i

    @measure_time_mem
    def reduce(self, df, verbose=False):
        """Takes a dataframe and returns it with all data transformed to the
        smallest necessary types.

        :param df: pandas dataframe
        :param verbose: If True, outputs more information
        :return: pandas dataframe with reduced data types
        """
        ret_list = Parallel(n_jobs=self.n_jobs)(delayed(self._reduce)
                                                (df[c], c, verbose) for c in
                                                df.columns)

        del df
        gc.collect()
        return pd.concat(ret_list, axis=1)

    def _reduce(self, s, colname, verbose):
        # skip NaNs
        if s.isnull().any():
            if verbose:
                print(f'{colname} has NaNs - Skip..')
            return s
        # detect kind of type
        coltype = s.dtype
        if np.issubdtype(coltype, np.integer):
            conv_key = 'int' if s.min() < 0 else 'uint'
        elif np.issubdtype(coltype, np.floating):
            conv_key = 'float'
        else:
            if isinstance(coltype, object) and self.use_categoricals:
                # check for all-strings series
                if s.apply(lambda x: isinstance(x, str)).all():
                    if verbose:
                        print(f'convert {colname} to categorical')
                    return s.astype('category')
            if verbose:
                print(f'{colname} is {coltype} - Skip..')
            return s
        # find right candidate
        for cand, cand_info in self._type_candidates(conv_key):
            if s.max() <= cand_info.max and s.min() >= cand_info.min:
                if verbose:
                    print(f'convert {colname} to {cand}')
                return s.astype(cand)

        # reaching this code is bad. Probably there are inf, or other high numbs
        print(f"WARNING: {colname} doesn't fit the grid with \nmax: {s.max()} "
              f"and \nmin: {s.min()}")
        print('Dropping it..')


class EraSampler(Sampler):
    r"""Takes a dataset with era indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Arguments:
        data_source (Dataset): a Dataset to sample from. Should have a era property
        batch_size (int): a batch size that you would like to use later with Dataloader class
        shuffle (bool): whether to shuffle the data or not
    """

    def __init__(self, data_source, indices, batch_size=None, shuffle=True):
        super(EraSampler, self).__init__(data_source)
        self.data_source = data_source
        self.era = self.data_source.era[indices]
        """
        if batch_size is not None:
            assert self.data_source.batch_sizes is None, "do not declare batch size in sampler " \
                                                         "if data source already got one"
            self.batch_sizes = [len(np.where(self.data_source.era == i)) for i in np.unique(self.data_source.era)]
        else:
            self.batch_sizes = self.data_source.batch_sizes
        """

        self.shuffle = shuffle

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):
        batch_lists = [self.era.index[np.where(
            self.era == i)] for i in np.unique(self.era)]
        # flatten lists and shuffle the batches if necessary
        # this works on batch level
        # batch_lists = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(batch_lists)
        return iter(batch_lists)

    def __len__(self):
        return len(np.unique(self.era))


def data_sampler(tr_idx: list, val_idx: list, data_dict: dict = None, type='train', downsampling=1, count=0):
    if not data_dict:
        data = load_data('./data', mode=type)
        data = data.iloc[tr_idx[0]:val_idx[-1] + 1]
        data, target, features, era = preprocess_data(data, nn=True)
    #data, target, features, era = data_dict['data'], data_dict['target'], data_dict['features'], data_dict['era']
    tr_idx, val_idx = np.array(tr_idx), np.array(val_idx)
    tr_idx, val_idx = tr_idx[count::downsampling], val_idx[count::downsampling]
    x_tr, x_val = data_dict['data'][tr_idx], data_dict['data'][val_idx]
    y_tr, y_val = data_dict['target'][tr_idx], data_dict['target'][val_idx]
    era_tr,era_val= data_dict['era'][tr_idx],data_dict['era'][val_idx]
    return x_tr, y_tr, x_val, y_val, era_tr,era_val


def get_eras(type='train'):
    data = load_data('./data', mode=type)
    return data[['era', 'target']]


def create_data_dict(mode, nn=False):
    data = load_data('data/', mode=mode)
    data, target, features, era = preprocess_data(data, nn=nn)
    data_dict = {'data':     data, 'target': target,
                 'features': features, 'era': era}
    return data_dict


def get_feature_sets(root_dir):
    with open(f'{root_dir}/datasets/features.json') as f:
        feature_metadata = json.load(f)
    leg_feat = feature_metadata['feature_sets']['legacy']
    med_feat = feature_metadata['feature_sets']['medium']
    small_feat = feature_metadata['feature_sets']['small']
    meta_data = ['era', 'data_type']
    features = {'legacy': leg_feat, 'medium': med_feat, 'small': small_feat, 'meta_data': meta_data}
    return features
