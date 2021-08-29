import datetime
import os.path
import pickle

import catboost as cat
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import xgboost as xgb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import mean_squared_error

import data_loading.utils as utils
from data_loading.purged_group_time_series import PurgedGroupTimeSeriesSplit
from models.resnet import ResNet, init_weights


class Trainer():
    def __init__(self, data_dict, model_dict):
        self.data_dict = data_dict
        self.features = data_dict['features']
        self.save_loc = './saved_models/trained/cross_val/'
        self.model_dict = model_dict
        self.get_hyp_params()
        self.add_const_params()
        self.scores = {}

    def get_hyp_params(self):
        for model, path in self.model_dict.items():
            if type(path) == str:
                study = joblib.load(path)
                self.model_dict[model] = study.best_params

    def add_const_params(self):
        if 'cat' in self.model_dict:
            self.model_dict['cat'].update({'loss_function':  'RMSE',
                                           'eval_metric':    'RMSE',
                                           'bootstrap_type': 'Bayesian',
                                           'use_best_model': True})
        if 'lgb' in self.model_dict:
            self.model_dict['lgb'].update({'boosting':  'gbdt',
                                           'objective': 'regression',
                                           'verbose':   1,
                                           'n_jobs':    10,
                                           'metric':    'mse'})
        if 'xgb' in self.model_dict:
            self.model_dict['xgb'].update({'objective':   'reg:squarederror',
                                           'booster':     'gbtree',
                                           'tree_method': 'gpu_hist',
                                           'verbosity':   1,
                                           'n_jobs':      10,
                                           'eval_metric': 'rmse'})

    def cross_val_train(self, splits=10, state=0):

        gts = PurgedGroupTimeSeriesSplit(n_splits=splits, group_gap=2)
        for i, (tr_idx, val_idx) in enumerate(gts.split(self.data_dict['data'], groups=self.data_dict['era'])):
            x_tr, x_val = self.data_dict['data'][tr_idx], self.data_dict['data'][val_idx]
            y_tr, y_val = self.data_dict['target'][tr_idx], self.data_dict['target'][val_idx]
            if 'xgb' in self.model_dict:
                dir = f'./saved_models/xgb/cross_val/{datetime.date.today()}'
                if not os.path.isdir(dir):
                    os.makedirs(dir)
                path = f'./saved_models/xgb/cross_val/{datetime.date.today()}/xgb_{i}_model_seed_{state}.pkl'
                self.train_xgb(x_tr=x_tr, x_val=x_val,
                               y_tr=y_tr, y_val=y_val, path=path)
            if 'lgb' in self.model_dict:
                dir = f'./saved_models/lgb/cross_val/{datetime.date.today()}'
                if not os.path.isdir(dir):
                    os.makedirs(dir)
                path = f'./saved_models/lgb/cross_val/{datetime.date.today()}/lgb_{i}_model_seed_{state}.txt'
                self.model_dict['lgb'].update({'seed': state})
                self.train_lgb(x_tr=x_tr, x_val=x_val,
                               y_tr=y_tr, y_val=y_val, path=path)
            if 'cat' in self.model_dict:
                dir = f'./saved_models/cat/cross_val/{datetime.date.today()}'
                if not os.path.isdir(dir):
                    os.makedirs(dir)
                path = f'./saved_models/cat/cross_val/{datetime.date.today()}/cat_{i}_model_seed_{state}.dump'
                self.model_dict['cat'].update({'random_state': state})
                self.train_cat(x_tr=x_tr, x_val=x_val,
                               y_tr=y_tr, y_val=y_val, path=path)
            if 'resnet' in self.model_dict:
                dataset = utils.FinData(
                    data=self.data_dict['data'], target=self.data_dict['target'], era=self.data_dict['era'])
                data_loaders = utils.create_dataloaders(
                    dataset, indexes={
                        'train': tr_idx.tolist(), 'val': val_idx.tolist()},
                    batch_size=self.model_dict['resnet']['batch_size'])
                self.model_dict['resnet'].update(
                    {'features': [feat for feat in range(len(self.data_dict['features']))]})

                path = f'./saved_models/ResNet/cross_val/{datetime.date.today()}/resnet_{i}.pkl'
                self.train_resnet(data_loaders=data_loaders, path=path)

    def train_resnet(self, data_loaders, path):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='./lightning_logs',
                                                           monitor='val_loss',
                                                           save_top_k=1, period=10,
                                                           mode='min')
        model = ResNet(self.model_dict['resnet'])
        if self.model_dict['resnet']['activation'] == nn.ReLU:
            model.apply(lambda m: init_weights(m, 'relu'))
        elif self.model_dict['resnet']['activation'] == nn.LeakyReLU:
            model.apply(lambda m: init_weights(m, 'leaky_relu'))
        es = EarlyStopping(monitor='val_loss', patience=10,
                           min_delta=0.000005, mode='min')
        trainer = pl.Trainer(max_epochs=1000,
                             gpus=1,
                             callbacks=[checkpoint_callback, es])
        trainer.fit(
            model, train_dataloader=data_loaders['train'], val_dataloaders=data_loaders['val'])
        best_model = ResNet.load_from_checkpoint(
            checkpoint_callback.best_model_path, params=self.model_dict['resnet'])
        torch.save(best_model.state_dict(), path)

    def train_xgb(self, x_tr, y_tr, x_val, y_val, path):
        d_tr = xgb.DMatrix(x_tr, label=y_tr)
        d_val = xgb.DMatrix(x_val, label=y_val)
        clf = xgb.train(self.model_dict['xgb'], d_tr, 1000, [
            (d_val, 'eval')], early_stopping_rounds=50, verbose_eval=True)
        val_pred = clf.predict(d_val)
        score = mean_squared_error(y_val, val_pred)
        if self.scores.get('xgb', None):
            self.scores['xgb'].append(score)
        else:
            self.scores['xgb'] = [score]
        pickle.dump(
            clf, open(path, 'wb'))
        del clf, d_tr, d_val, val_pred

    def train_lgb(self, x_tr, x_val, y_tr, y_val, path):
        train = lgb.Dataset(x_tr, label=y_tr)
        val = lgb.Dataset(x_val, label=y_val)
        clf = lgb.train(self.model_dict['lgb'], train, 1000, valid_sets=[
            val], early_stopping_rounds=50, verbose_eval=True)
        preds = clf.predict(x_val)
        score = mean_squared_error(y_val, preds)
        if self.scores.get('lgb', None):
            self.scores['lgb'].append(score)
        else:
            self.scores['lgb'] = [score]
        clf.save_model(path)
        del clf, train, val, preds

    def train_cat(self, x_tr, x_val, y_tr, y_val, path):
        model = cat.CatBoostRegressor(
            iterations=1000, task_type='GPU', **self.model_dict['cat'])
        model.fit(X=x_tr, y=y_tr, eval_set=(x_val, y_val),
                  early_stopping_rounds=50, verbose_eval=10)
        preds = model.predict(x_val)
        score = mean_squared_error(y_val, preds)
        if self.scores.get('cat', None):
            self.scores['cat'].append(score)
        else:
            self.scores['cat'] = [score]
        model.save_model(path)

        del model

    def full_train(self):
        tr_idx = np.where(self.data_dict['era'] > 200)[0].tolist()
        val_idx = np.where(self.data_dict['era'] <= 200)[0].tolist()
        x_tr, x_val = self.data_dict['data'][tr_idx], self.data_dict['data'][val_idx]
        y_tr, y_val = self.data_dict['target'][tr_idx], self.data_dict['target'][val_idx]

        if self.model_dict['xgb']:
            path_xgb = f'./saved_models/xgb/full_train/{datetime.date.today()}/full_train.pkl'
            self.train_xgb(x_tr=x_tr, x_val=x_val, y_tr=y_tr,
                           y_val=y_val, path=path_xgb)
        if self.model_dict['lgb']:
            path_lgb = f'./saved_models/lgb/full_train/{datetime.date.today()}/full_train.txt'
            self.train_lgb(x_tr=x_tr, x_val=x_val, y_tr=y_tr,
                           y_val=y_val, path=path_lgb)
        if self.model_dict['cat']:
            path_cat = f'./saved_models/cat/full_train/{datetime.date.today()}/full_train.dump'
            self.train_cat(x_tr=x_tr, x_val=x_val, y_tr=y_tr,
                           y_val=y_val, path=path_cat)
        try:
            del x_tr, x_val, y_tr, y_val
        except:
            pass
        if self.model_dict['resnet']:
            dataset = utils.FinData(
                data=self.data_dict['data'], target=self.data_dict['target'], era=self.data_dict['era'])
            data_loaders = utils.create_dataloaders(
                dataset, indexes={
                    'train': tr_idx, 'val': val_idx}, batch_size=self.model_dict['resnet']['batch_size'])
            path_resnet = './saved_models/ResNet/full_train.pkl'
            self.train_resnet(data_loaders=data_loaders, path=path_resnet)


def main():
    data = utils.load_data(root_dir='./data', mode='train')
    data, target, features, era = utils.preprocess_data(data, nn=True)
    random_seed = 0
    data_dict = {'data':     data, 'target': target,
                 'features': features, 'era': era}
    data = utils.load_data(root_dir='./data', mode='test')
    data = data[data['data_type'] == 'validation']
    data, target, features, era = utils.preprocess_data(data, nn=True)
    data_dict['data'] = np.concatenate(
        [data_dict['data'], data], 0)
    data_dict['target'] = np.concatenate(
        [data_dict['target'], target], 0)
    data_dict['era'] = pd.Series(
        np.concatenate([data_dict['era'], era], 0))
    model_dict = {'xgb': './hpo/params/xgb_hpo_2021-05-15.pkl',
                  'lgb': './hpo/params/lgb_hpo_ae_False_2021-05-16.pkl',
                  'cat': {'learning_rate': 0.013520865420108316, 'min_data_in_leaf':
                                           599, 'l2_leaf_reg': 0.04498050323217781, 'bagging_temperature':
                                           0.17428787707251533, 'depth': 10}
                  }
    trainer = Trainer(data_dict=data_dict, model_dict=model_dict)
    trainer.cross_val_train(splits=10, state=random_seed)
    trainer.cross_val_train(splits=10, state=random_seed + 1)
    trainer.cross_val_train(splits=10, state=random_seed + 2)
    trainer.full_train()
