from numpy.core.fromnumeric import mean
import pandas as pd
import torch

import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import OrderedDict
from itertools import combinations
import numpy as np
from torch.nn.modules.batchnorm import BatchNorm1d
from data_loading import utils
from data_loading import purged_group_time_series as pgs
from sklearn.metrics import mean_squared_error
from metrics.corr_loss_function import CorrLoss


class FE_NN(pl.LightningModule):
    def __init__(self, data_dict, combos, p):
        super(FE_NN, self).__init__()
        self.data = data_dict['data']
        self.n_features = self.data.shape[-1]
        self.features = p['features']
        self.n_feature_combos = p['n_feature_combos']
        self.group_size = p['group_size']
        self.loss = nn.MSELoss()
        self.corr_loss = CorrLoss()
        self.feature_loss = nn.MSELoss()
        self.activation = nn.LeakyReLU
        self.drop_rate = 0.5
        self.lr = 1e-4
        self.f_models = nn.ModuleDict(self.feature_models())
        self.regressor = self.final_model()
        self.feature_combos = combos
        self.embedded_features = {}
        self.val_embedded_features = {}
        self.create_forward_hooks()

    def feature_models(self):
        models = {}
        for combo in range(self.n_feature_combos):
            models[f'feature_combo_{combo}'] = nn.Sequential(
                OrderedDict(
                    [
                        (f"batch_norm_inp_{combo}",
                         nn.BatchNorm1d(self.group_size - 1)),
                        (f"input_mod_{combo}", nn.Linear(
                            self.group_size - 1, 10, bias=False)),
                        (f"act_1_mod_{combo}", self.activation()),
                        (f"batch_nn_1_mod_{combo}",
                         nn.BatchNorm1d(10)),
                        (f"drop_1_mod_{combo}", nn.Dropout(self.drop_rate)),
                        (f"layer_2_mod_{combo}", nn.Linear(
                            10, 2)),
                        (f"act_2_mod_{combo}", self.activation()),
                        (f"batch_nn_2_mod_{combo}", nn.BatchNorm1d(2)),
                        (f"drop_2_mod_{combo}", nn.Dropout(self.drop_rate)),
                        (f"out_mod_{combo}", nn.Linear(2, 1))
                    ]
                )
            )
        return models

    def create_forward_hooks(self):
        for i, key in enumerate(self.f_models.keys()):
            self.f_models[key].__getattr__(f'act_2_mod_{i}').register_forward_hook(
                self.hook_embedded_feature(key))

    def get_embedded_features(self):
        features = torch.ones(
            (self.n_feature_combos, self.batch_size, 2)).to(self.device)
        if self.training:
            for i, key in enumerate(self.embedded_features.keys()):
                features[i] = self.embedded_features[key]
        else:
            for i, key in enumerate(self.val_embedded_features.keys()):
                features[i] = self.val_embedded_features[key]
        return torch.hstack([feat for feat in features])

    def final_model(self):
        model = nn.Sequential(
            nn.BatchNorm1d(self.n_feature_combos * 2 + self.n_features),
            nn.Linear(self.n_feature_combos * 2 +
                      self.n_features, 500, bias=False),
            nn.BatchNorm1d(500),
            self.activation(),
            nn.Dropout(self.drop_rate),
            nn.Linear(500, 250, bias=False),
            nn.BatchNorm1d(250),
            self.activation(),
            nn.Dropout(self.drop_rate),
            nn.Linear(250, 100, bias=False),
            nn.BatchNorm1d(100),
            self.activation(),
            nn.Dropout(self.drop_rate),
            nn.Linear(100, 25, bias=False),
            nn.BatchNorm1d(25),
            self.activation(),
            nn.Dropout(self.drop_rate),
            nn.Linear(25, 1, bias=False)
        )
        return model

    def forward(self, x):
        targets = {}
        outs = {}
        for i, key in enumerate(self.f_models.keys()):
            inputs = x[:, self.feature_combos[i][0]]
            targets[key] = x[:, self.feature_combos[i][1]]
            outs[key] = self.f_models[key](inputs)
        embedded_features = self.get_embedded_features()
        reg_input = torch.cat([x, embedded_features], 1)
        reg_out = self.regressor(reg_input)
        return (outs, targets), reg_out

    def training_step(self, batch, batch_idx):
        x, y = batch['data'].view(
            batch['data'].shape[1], -1), batch['target'].T
        self.batch_size = x.shape[0]
        (outs, targets), reg_out = self(x)
        feature_losses = 0.0
        feature_loss_weight = 0.5 / self.n_feature_combos
        for key in outs.keys():
            out = outs[key]
            target = targets[key]
            feature_losses += self.feature_loss(out,
                                                target) * feature_loss_weight
        reg_loss = self.loss(reg_out, y)
        corr_loss = self.corr_loss(reg_out, y)
        loss = (reg_loss * 0.5) + feature_losses + (1 - corr_loss) * 0.1
        self.log('reg_loss', reg_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('feature_losses', feature_losses,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('corr_loss', corr_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch['data'].view(
            batch['data'].shape[1], -1), batch['target'].T
        self.batch_size = x.shape[0]
        (outs, targets), reg_out = self(x)
        feature_losses = 0.0
        feature_loss_weight = 0.5 / self.n_feature_combos
        for key in outs.keys():
            out = outs[key]
            target = targets[key]
            feature_losses += self.feature_loss(out,
                                                target) * feature_loss_weight
        reg_loss = self.loss(reg_out, y)
        corr_loss = self.corr_loss(reg_out, y)
        loss = (reg_loss * 0.5) + feature_losses + (1 - corr_loss) * 0.1
        return {'val_loss':      loss, 'val_reg_loss': reg_loss, 'val_feature_loss': feature_losses,
                'val_corr_loss': corr_loss}

    def validation_epoch_end(self, outputs):
        epoch_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        epoch_feature_loss = torch.tensor(
            [x['val_feature_loss'] for x in outputs]).mean()
        epoch_reg_loss = torch.tensor(
            [x['val_reg_loss'] for x in outputs]).mean()
        epoch_corr_loss = torch.tensor(
            [x['val_corr_loss'] for x in outputs]).mean()
        self.log('val_loss', epoch_loss, prog_bar=True)
        self.log('val_reg_loss', epoch_reg_loss, prog_bar=True)
        self.log('val_corr_loss', epoch_corr_loss,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_feature_loss', epoch_feature_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, mode='min', factor=0.1, min_lr=1e-7, eps=1e-08
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def hook_embedded_feature(self, name):
        def hook(model, input, output):
            if self.training:
                self.embedded_features[name] = output.detach()
            else:
                self.val_embedded_features[name] = output.detach()

        return hook


def create_combos(features, group_size, n_feature_combos):
    top_features = ['feature_intelligence4',
                    'feature_intelligence8',
                    'feature_dexterity11',
                    'feature_intelligence1',
                    'feature_dexterity6',
                    'feature_dexterity7',
                    'feature_strength34',
                    'feature_charisma63',
                    'feature_dexterity4',
                    'feature_charisma85',
                    'feature_dexterity14',
                    'feature_intelligence5',
                    'feature_wisdom23',
                    'feature_strength14',
                    'feature_intelligence11']
    # indices = [features.index(feat) for feat in top_features]
    # features = indices
    combos = []
    targets = []
    for i in range(n_feature_combos):
        feature_subset = np.random.choice(
            features, size=group_size, replace=False)
        combo = np.random.choice(
            feature_subset, size=group_size - 1, replace=False)
        combos.append(combo)
        targets.append([(feat,)
                        for feat in feature_subset if feat not in combo])
    return [combo for combo in zip(combos, targets)]


def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]


# convenience method for scoring
def scoring(df):
    return correlation(df['preds'], df['target'])


# Payout is just the score cliped at +/-25%
def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)


def main():
    utils.seed_everything(0)
    data = utils.load_data(root_dir='./data', mode='test')
    data = data[data['data_type'] == 'validation']
    data_test, target_test, features, era_test = utils.preprocess_data(
        data, nn=True)
    data = utils.load_data(root_dir='./data', mode='train')
    data, target, features, era = utils.preprocess_data(data, nn=True)
    data = np.concatenate([data, data_test], 0)
    target = np.concatenate([target, target_test], 0)
    era = pd.Series(np.concatenate([era, era_test], 0))
    tr_idx = np.where(era < 121)[0].tolist()
    val_idx = np.where(era >= 121)[0].tolist()
    del data_test, target_test, era_test
    data_dict = {'data':     data, 'target': target,
                 'features': features, 'era': era}
    dataset = utils.FinData(data, target, era)
    p = {'features':         [i for i in range(len(features))],
         'n_feature_combos': 1,
         'group_size':       4}
    combos = create_combos(
        p['features'], p['group_size'], p['n_feature_combos'])

    model = FE_NN(data_dict, combos, p)

    data_loaders = utils.create_dataloaders(
        dataset, {'train': tr_idx, 'val': val_idx}, 5000)

    """
    es = EarlyStopping(monitor='val_reg_loss', patience=10,
                       min_delta=0.000001, mode='min')
    mod_check = ModelCheckpoint(
        dirpath='./lightning_logs', monitor='val_reg_loss', mode='min', save_top_k=1)
    trainer = pl.Trainer(max_epochs=1000, gpus=1,
                         callbacks=[es, mod_check])
    trainer.fit(
        model, train_dataloader=data_loaders['train'], val_dataloaders=data_loaders['val'])
    data = utils.load_data(root_dir='./data', mode='test')
    data = data[data['data_type'] == 'validation']
    data, target, features, era = utils.preprocess_data(data, nn=True)
    data = torch.Tensor(data)
    model = FE_NN.load_from_checkpoint(
        mod_check.best_model_path, data_dict=data_dict, combos=combos, p=p)
    model.eval()
    model.batch_size = data.shape[0]
    _, preds = model(data)
    preds = preds.detach().cpu().numpy().reshape(-1)
    df_preds = pd.DataFrame.from_dict(
        {'era': era, 'pred': preds, 'target': target})
    corr_per_era = df_preds.groupby('era')[['pred', 'target']].apply(
        lambda x: correlation(x['pred'], x['target']))
    sharpe_ratio = corr_per_era.mean() / corr_per_era.std()
    rolling_max = (corr_per_era +
                   1).cumprod().rolling(window=100, min_periods=1).max()
    daily_values = (corr_per_era + 1).cumprod()
    max_drawdown = (rolling_max - daily_values).max()
    mean_squared_error_ = mean_squared_error(
        df_preds['target'], df_preds['pred'])
    print("preds", preds)
    print("Corr_mean", corr_per_era.mean())
    print("Sharpe", sharpe_ratio)
    print("Max Drawdown", max_drawdown)
    print('mean_squared_error', mean_squared_error_).
    """
