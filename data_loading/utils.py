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
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from torch.utils.data import Dataset, Subset, BatchSampler, SequentialSampler, DataLoader
import time
from joblib import Parallel, delayed
import gc
import xgboost as xgb


class FinData(Dataset):
    def __init__(self, data, target, era, hidden=None, mode='train', transform=None, cache_dir=None):
        self.data = data
        self.target = target
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir
        self.era = era
        self.hidden = hidden

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index.to_list()
        if self.transform:
            return self.transform(self.data.iloc[index].values)
        else:
            if type(self.target) is not np.ndarray:
                sample = {
                    'target': torch.Tensor(self.target[index].values),
                    'data':   torch.Tensor(self.data[index]),
                    'era':    torch.Tensor(self.era[index].values),
                }
            else:
                sample = {
                    'target': torch.Tensor(self.target[index]),
                    'data':   torch.Tensor(self.data[index]),
                    'era':    torch.Tensor(self.era[index]),
                }
            if self.hidden is not None:
                sample['hidden'] = torch.Tensor(self.hidden[index])
        return sample

    def __len__(self):
        return len(self.data)


def get_data_path(root_dir):
    dotenv_path = 'num_config.env'
    load_dotenv(dotenv_path=dotenv_path)
    curr_round = os.getenv('LATEST_ROUND')
    data_path = root_dir + '/numerai_dataset_' + str(curr_round)
    return data_path


def load_data(root_dir, mode, overide=None):
    data_path = get_data_path(root_dir=root_dir)
    if overide:
        data = dt.fread(overide).to_pandas()
    elif mode == 'train':
        data = dt.fread(data_path + '/numerai_training_data.csv').to_pandas()
    elif mode == 'test':
        data = dt.fread(data_path + '/numerai_tournament_data.csv').to_pandas()
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
    features = [col for col in data.columns if 'feature' in col]
    era = data['era']
    era = era.transform(lambda x: re.sub('[a-z]', '', x))
    if not test:
        era = era.astype('int')
    target = data['target']
    data = data[features]
    if scale:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    if ordinal:
        oe = OrdinalEncoder()
        data = oe.fit_transform(data)
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


def weighted_mean(scores, sizes):
    largest = np.max(sizes)
    weights = [size / largest for size in sizes]
    return np.average(scores, weights=weights)


def create_dataloaders(dataset: Dataset, indexes: dict, batch_size):
    train_idx = indexes.get('train', None)
    val_idx = indexes.get('val', None)
    test_idx = indexes.get('test', None)
    dataloaders = {}
    if train_idx:
        train_set = Subset(dataset, train_idx)
        train_sampler = BatchSampler(
            train_set.indices, batch_size=batch_size, drop_last=False)
        dataloaders['train'] = DataLoader(
            dataset, sampler=train_sampler, num_workers=10, pin_memory=True, shuffle=False)
    if val_idx:
        val_set = Subset(dataset, val_idx)
        val_sampler = BatchSampler(
            val_set.indices, batch_size=batch_size, drop_last=False)
        dataloaders['val'] = DataLoader(
            dataset, sampler=val_sampler, num_workers=10, pin_memory=True, shuffle=False)
    if test_idx:
        test_set = Subset(dataset, test_idx)
        test_sampler = BatchSampler(
            test_set.indices, batch_size=batch_size, drop_last=False)
        dataloaders['test'] = DataLoader(
            dataset, sampler=test_sampler, num_workers=10, pin_memory=True, shuffle=False)
    return dataloaders


def seed_everything(seed):
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


def create_predictions(root_dir: str = './data', models: dict = {}, hidden=True, ae=True):
    test_files_path, test_files_exist = check_test_files(root_dir)
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
        if models.get('ae', None):
            p_ae = models['ae'][1]
            p_ae['input_size'] = len(features)
            p_ae['output_size'] = 1
            model = models['ae'][0]
            model.eval()
        if ae:
            hidden_pred = create_hidden_rep(
                model=model, data_dict=data_dict)
            data_dict['hidden_true'] = True
            df['prediction_ae'] = hidden_pred['preds']
        if models.get('ResNet', None):
            p_res = models['ResNet'][1]
            p_res['input_size'] = len(features)
            p_res['output_size'] = 1
            p_res['hidden_len'] = data_dict['hidden'].shape[-1]
            dataset = FinData(
                data=data_dict['data'], target=data_dict['target'], era=data_dict['era'],
                hidden=data_dict.get('hidden', None))
            dataloaders = create_dataloaders(
                dataset, indexes={'train': t_idx}, batch_size=p_res['batch_size'])
            model = models['ResNet'][0]
            model.eval()
            predictions = []
            for batch in dataloaders['train']:
                pred = model(batch['data'].view(
                    batch['data'].shape[1], -1), hidden=batch['hidden'].view(batch['hidden'].shape[1], -1))
                predictions.append(pred.cpu().detach().numpy().tolist())
            predictions = np.array([predictions[i][j] for i in range(
                len(predictions)) for j in range(len(predictions[i]))])
            df['prediction_resnet'] = predictions
        if models.get('xgb', None):
            model_xgboost_1 = models['xgb'][0]
            model_xgboost_2 = models['xgb'][1]
            x_val = data_dict['data']
            preds_1 = model_xgboost_1.predict(xgb.DMatrix(x_val))
            preds_2 = model_xgboost_2.predict(xgb.DMatrix(x_val))
            df['prediction_xgb'] = np.mean([0.55*preds_1 + 0.45 * preds_2], 0)
        if models.get('lgb', None):
            model_lgb = models['lgb'][0]
            model_lgb_2 = models['lgb'][1]
            x_val = data_dict['data']
            pred_1 = model_lgb.predict(x_val)
            preds_2 = model_lgb_2.predict(x_val)
            df['prediction_lgb'] = np.mean([0.55*preds_1 + 0.45 * preds_2], 0)
        df['prediction'] = df['prediction_xgb']
        df = df[['id', 'prediction']]
        pred_path = f'{get_data_path(root_dir)}/predictions/{era[0]}'
        df.to_csv(f'{pred_path}.csv')


def check_test_files(root_dir='./data'):
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
    return df


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
