import datetime
import os
import copy
import joblib
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import optuna
import pytorch_lightning as pl
import torch.nn as nn
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import Subset, BatchSampler, SequentialSampler, DataLoader
import torch
import numpy as np
from models.resnet import ResNet as Classifier
from data_loading.purged_group_time_series import PurgedGroupTimeSeriesSplit
from data_loading.utils import load_data, preprocess_data, FinData, weighted_mean, seed_everything, calc_data_mean, \
    create_dataloaders


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('leaky_relu'))
        m.bias.data.fill_(1)


def create_param_dict(trial, trial_file=None):
    if trial and not trial_file:
        dim_1 = trial.suggest_int('dim_1', 500, 2000)
        dim_2 = trial.suggest_int('dim_2', 1000, 3000)
        dim_3 = trial.suggest_int('dim_3', 1000, 3000)
        dim_4 = trial.suggest_int('dim_4', 500, 1000)
        dim_5 = trial.suggest_int('dim_5', 100, 250)
        act_func = trial.suggest_categorical(
            'activation', ['relu', 'leaky_relu', 'gelu', 'silu'])
        act_dict = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU,
                    'gelu': nn.GELU, 'silu': nn.SiLU}
        act_func = act_dict[act_func]
        dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
        lr = trial.suggest_uniform('lr', 0.00005, 0.05)
        p = {'dim_1': dim_1, 'dim_2': dim_2, 'dim_3': dim_3,
             'dim_4': dim_4, 'dim_5': dim_5, 'activation': act_func, 'dropout': dropout,
             'lr': lr, 'loss': nn.MSELoss, 'embedding': True}
    elif trial and trial_file:
        p = joblib.load(trial_file).best_params
        if not p.get('dim_5', None):
            p['dim_5'] = 75
        if not p.get('label_smoothing', None):
            p['label_smoothing'] = 0.094
        act_dict = {'relu': nn.ReLU,
                    'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU}
        act_func = trial.suggest_categorical(
            'activation', ['leaky_relu', 'gelu'])
        p['activation'] = act_dict[p['activation']]
    return p


def optimize(trial: optuna.Trial, data_dict):
    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=5)
    input_size = data_dict['data'].shape[-1]
    output_size = 1
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join('models/', "trial_resnet_{}".format(trial.number)), monitor="val_mse", mode='min')
    logger = MetricsCallback()
    metrics = []
    sizes = []
    # trial_file = 'HPO/nn_hpo_2021-01-05.pkl'
    trial_file = None
    p = create_param_dict(trial, trial_file)
    p['batch_size'] = trial.suggest_int('batch_size', 8000, 15000)
    for i, (train_idx, val_idx) in enumerate(gts.split(data_dict['data'], groups=data_dict['era'])):
        model = Classifier(input_size, output_size, params=p)
        # model.apply(init_weights)
        dataset = FinData(
            data=data_dict['data'], target=data_dict['target'], era=data_dict['era'])
        dataloaders = create_dataloaders(
            dataset, indexes={'train': train_idx, 'val': val_idx}, batch_size=p['batch_size'])
        es = EarlyStopping(monitor='val_mse', patience=10,
                           min_delta=0.0005, mode='min')
        trainer = pl.Trainer(logger=False,
                             max_epochs=500,
                             gpus=1,
                             callbacks=[checkpoint_callback, logger, PyTorchLightningPruningCallback(
                                 trial, monitor='val_mse'), es],
                             precision=16)
        trainer.fit(
            model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
        val_loss = logger.metrics[-1]['val_loss'].item()
        metrics.append(val_loss)
        sizes.append(len(train_idx))
    metrics_mean = weighted_mean(metrics, sizes)
    return metrics_mean


def main():
    seed_everything(0)
    data = load_data(root_dir='./data/', mode='train')
    data, target, features, era = preprocess_data(
        data, ordinal=True)
    api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYWQxMjg3OGEtMGI1NC00NzFmLTg0YmMtZmIxZjcxZDM2NTAxIn0='
    neptune.init(api_token=api_token,
                 project_qualified_name='jamesmccarthy65/Numerai')
    nn_exp = neptune.create_experiment('Resnet_HPO')
    nn_neptune_callback = opt_utils.NeptuneCallback(experiment=nn_exp)
    study = optuna.create_study(direction='minimize')
    data_dict = {'data': data, 'target': target,
                 'features': features, 'era': era}
    study.optimize(lambda trial: optimize(trial, data_dict=data_dict), n_trials=10,
                   callbacks=[nn_neptune_callback])
    joblib.dump(
        study, f'hpo/params/nn_hpo_{str(datetime.datetime.now().date())}.pkl')


if __name__ == '__main__':
    main()
