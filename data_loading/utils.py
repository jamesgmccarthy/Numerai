import os
import random
import re
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
    def __init__(self, data, target, era, mode='train', transform=None, cache_dir=None):
        self.data = data
        self.target = target
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir
        self.era = era

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
                    'era':    torch.Tensor(self.era[index].values)
                }
            else:
                sample = {
                    'target': torch.Tensor([self.target[index]]),
                    'data':   torch.LongTensor([self.data[index]]),
                    'era':    torch.Tensor([self.era[index]])
                }
        return sample

    def __len__(self):
        return len(self.data)


def load_data(root_dir, mode, overide=None):
    dotenv_path = 'num_config.env'
    load_dotenv(dotenv_path=dotenv_path)
    curr_round = os.getenv('LATEST_ROUND')
    data_path = root_dir + '/numerai_dataset_' + str(curr_round)
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
        train_set = Subset(
            dataset, train_idx)
        train_sampler = BatchSampler(SequentialSampler(
            train_set), batch_size=batch_size, drop_last=False)
        dataloaders['train'] = DataLoader(
            dataset, sampler=train_sampler, num_workers=10, pin_memory=True)
    if val_idx:
        val_set = Subset(dataset, val_idx)
        val_sampler = BatchSampler(SequentialSampler(
            val_set), batch_size=batch_size, drop_last=False)
        dataloaders['val'] = DataLoader(
            dataset, sampler=val_sampler, num_workers=10, pin_memory=True)
    if test_idx:
        test_set = Subset(dataset, test_idx)
        test_sampler = BatchSampler(SequentialSampler(
            test_set), batch_size=batch_size, drop_last=False)
        dataloaders['test'] = DataLoader(
            dataset, sampler=test_sampler, num_workers=10, pin_memory=True)
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
