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
            'activation', ['swish'])  # 'relu', 'leaky_relu', 'gelu', 'silu',])
        act_dict = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU,
                    'gelu': nn.GELU, 'silu': nn.SiLU, 'swish': nn.Hardswish}
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


def optimize(trial: optuna.Trial, downsample, embedding=False):
    data_dict = utils.create_data_dict(mode='train', nn=True)
    input_size = data_dict['data'].shape[-1]
    trial_file = None
    p = create_param_dict(trial, trial_file)
    p['batch_size'] = trial.suggest_int('batch_size', 500, 5000)
    p['input_size'] = input_size
    p['output_size'] = 1
    p['emb'] = embedding
    metrics_downsampled = []
    for i in range(downsample):
        gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=5)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join('hpo/checkpoints/', f"trial_ae_{trial.number}_{i}"), monitor="val_reg_loss", mode='min')
        logger = MetricsCallback()

        # trial_file = 'HPO/nn_hpo_2021-01-05.pkl'
        metrics = []
        sizes = []
        print(f'Running Trail with params: {p}')
        for i, (train_idx, val_idx) in enumerate(gts.split(data_dict['data'], groups=data_dict['era'])):

            model = SupAE(params=p)
            # model.apply(init_weights)
            x_tr, y_tr, x_val, y_val, era_tr, era_val = utils.data_sampler(
                train_idx, val_idx, data_dict=data_dict, downsampling=downsample, count=i)
            x_tr = np.append(x_tr, x_val, 0)
            y_tr = np.append(y_tr, y_val, 0)
            era_tr = np.append(era_tr, era_val, 0)
            del x_val, y_val, era_val
            dataset = FinData(
                data=x_tr, target=y_tr, era=era_tr)
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
        metrics_downsampled.append(metrics_mean)
    return np.mean(metrics_downsampled)


def main(downsample=4, embedding=False):
    seed_everything(0)
    api_token = utils.read_api_token()
    neptune.init(api_token=api_token,
                 project_qualified_name='jamesmccarthy65/NumeraiV2')
    nn_exp = neptune.create_experiment(f'SupAE_HPO_downsample_{downsample}')
    nn_neptune_callback = opt_utils.NeptuneCallback(experiment=nn_exp)
    study = optuna.create_study(direction='minimize')

    study.optimize(lambda trial: optimize(trial, downsample=downsample, embedding=embedding), n_trials=200,
                   callbacks=[nn_neptune_callback])
    best_params = study.best_params
    best_params['emb'] = embedding
    best_params['input_size'] = data_dict['data'].shape[-1]
    best_params['output_size'] = 1
    joblib.dump(
        best_params, f'./hpo/params/ae_sup_params_{datetime.date.today()}.pkl')


if __name__ == '__main__':
    main()
