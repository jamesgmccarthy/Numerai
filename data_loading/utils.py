from logging import root
from models.SupervisedAutoEncoder import create_hidden_rep
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


# from lightning_nn import Classifier


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
            if type(index) is list:
                sample = {
                    'target': torch.Tensor(self.target[index].values),
                    'data':   torch.LongTensor(self.data[index]),
                    'era':    torch.Tensor(self.era[index].values),
                }
            else:
                sample = {
                    'target': torch.Tensor([self.target[index]]),
                    'data':   torch.LongTensor([self.data[index]]),
                    'era':    torch.Tensor([self.era[index]]),
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
        data, target, features, era = preprocess_data(data=df, ordinal=True)
        t_idx = np.arange(start=0, stop=len(era), step=1).tolist()
        data_dict = data_dict = {'data': data, 'target': target,
                                 'features': features, 'era': era}
        if models.get('ae', None):
            p_ae = models['ae'][1]
            p_ae['input_size'] = len(features)
            p_ae['output_size'] = 1
            model = models['ae'][0]
            model.eval()
        if not ae:
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
                data=data_dict['data'], target=data_dict['target'], era=data_dict['era'], hidden=data_dict.get('hidden', None))
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
        if models.get('xgboost', None):
            model_xgboost = models['xgboost'][0]
            p_xgboost = models['xgboost'][1]
            x_val = data_dict['data']
            df['prediction_xgb'] = model_xgboost.predict(x_val)
        if models.get('lgb', None):
            model_lgb = models['lgb'][0]
            p_lgb = models['lgb'][1]
            x_val = data_dict['data']
            df['prediction_lgb'] = model_lgb.predict(x_val)
        df = df[['id', 'prediction_lgb']]
        pred_path = f'{get_data_path(root_dir)}/predictions/{era[0]}'
        df.to_csv(f'{pred_path}.csv')


def check_test_files(root_dir='./data'):
    data_path = get_data_path(root_dir)
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
    df = df[['id', 'prediction_lgb']]
    df.columns = ['id', 'prediction']
    df.to_csv(f'{pred_path}predictions.csv')

    return df
