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
from models.SupervisedAutoEncoder import SupAE
from data_loading.purged_group_time_series import PurgedGroupTimeSeriesSplit
from data_loading.utils import load_data, preprocess_data, FinData, weighted_mean, seed_everything, calc_data_mean, \
    create_dataloaders
import data_loading.utils as utils


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def create_param_dict(trial, trial_file=None):
    if trial and not trial_file:
        dim_1 = trial.suggest_int('dim_1', 900, 1500)
        dim_2 = trial.suggest_int('dim_2', 400, 1000)
        dim_3 = trial.suggest_int('dim_3', 200, 500)
        hidden = trial.suggest_int('hidden', 100, 300)
        act_func = trial.suggest_categorical(
            'activation', ['relu', 'leaky_relu', 'gelu', 'silu'])
        act_dict = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU,
                    'gelu': nn.GELU, 'silu': nn.SiLU}
        act_func = act_dict[act_func]
        dropout = trial.suggest_uniform('dropout', 0.05, 0.5)
        dropout_ae = trial.suggest_uniform('dropout_ae', 0.05, 0.25)
        lr = trial.suggest_uniform('lr', 0.00005, 0.05)
        recon_loss_factor = trial.suggest_uniform('recon_loss_factor', 0.1, 1)
        p = {'dim_1':      dim_1, 'dim_2': dim_2, 'dim_3': dim_3, 'hidden': hidden,
             'activation': act_func, 'dropout': dropout, 'dropout_ae': dropout_ae,
             'lr':         lr, 'recon_loss_factor': recon_loss_factor, 'loss_sup_ae': nn.MSELoss,
             'loss_recon': nn.MSELoss, 'loss_reg': nn.MSELoss}
    elif not trial and trial_file:
        p = joblib.load(trial_file).best_params
        act_dict = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU,
                    'gelu': nn.GELU, 'silu': nn.SiLU}
        p['activation'] = act_dict[p['activation']]
    return p


def optimize(trial: optuna.Trial, data_dict, embedding=False):
    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=5)
    input_size = data_dict['data'].shape[-1]
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        os.path.join('hpo/checkpoints/', "trial_ae_{}".format(trial.number)), monitor="val_reg_loss", mode='min')
    logger = MetricsCallback()
    metrics = []
    sizes = []
    # trial_file = 'HPO/nn_hpo_2021-01-05.pkl'
    trial_file = None
    p = create_param_dict(trial, trial_file)
    p['batch_size'] = trial.suggest_int('batch_size', 500, 5000)
    p['input_size'] = input_size
    p['output_size'] = 1
    p['emb'] = embedding
    print(f'Running Trail with params: {p}')
    for i, (train_idx, val_idx) in enumerate(gts.split(data_dict['data'], groups=data_dict['era'])):
        model = SupAE(params=p)
        # model.apply(init_weights)
        dataset = FinData(
            data=data_dict['data'], target=data_dict['target'], era=data_dict['era'])
        dataloaders = create_dataloaders(
            dataset, indexes={'train': train_idx, 'val': val_idx}, batch_size=p['batch_size'])
        es = EarlyStopping(monitor='val_reg_loss', patience=10,
                           min_delta=0.0005, mode='min')
        trainer = pl.Trainer(logger=False,
                             max_epochs=100,
                             gpus=1,
                             callbacks=[checkpoint_callback, logger, PyTorchLightningPruningCallback(
                                 trial, monitor='val_reg_loss'), es],
                             precision=16)
        trainer.fit(
            model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
        val_loss = logger.metrics[-1]['val_reg_loss'].item()
        metrics.append(val_loss)
        sizes.append(len(train_idx))
    metrics_mean = weighted_mean(metrics, sizes)
    return metrics_mean


def main(embedding=False):
    seed_everything(0)
    data = utils.load_data(root_dir='./data/', mode='train')
    data, target, features, era = preprocess_data(
        data, ordinal=embedding, nn=embedding)
    api_token = utils.read_api_token()
    neptune.init(api_token=api_token,
                 project_qualified_name='jamesmccarthy65/NumeraiV2')

    nn_exp = neptune.create_experiment('SupAE_HPO')
    nn_neptune_callback = opt_utils.NeptuneCallback(experiment=nn_exp)
    study = optuna.create_study(direction='minimize')
    data_dict = {'data':     data, 'target': target,
                 'features': features, 'era': era}
    study.optimize(lambda trial: optimize(trial, data_dict=data_dict, embedding=embedding), n_trials=200,
                   callbacks=[nn_neptune_callback])
    best_params = study.best_params
    best_params['emb'] = embedding
    best_params['input_size'] = data_dict['data'].shape[-1]
    best_params['output_size'] = 1
    joblib.dump(
        best_params, f'./hpo/params/ae_sup_params_{datetime.date.today()}.pkl')


if __name__ == '__main__':
    main()
