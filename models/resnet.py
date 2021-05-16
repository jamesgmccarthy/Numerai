import torch
import copy
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data_loading.purged_group_time_series import PurgedGroupTimeSeriesSplit
from data_loading.utils import load_data, preprocess_data, FinData, create_dataloaders, calc_data_mean, init_weights
from metrics.corr_loss_function import CorrLoss, SpearmanLoss
from torchinfo import summary
from sklearn.model_selection import GroupKFold


class ResNet(pl.LightningModule):
    def __init__(self, params):
        super(ResNet, self).__init__()
        dim_1 = params['dim_1']
        dim_2 = params['dim_2']
        dim_3 = params['dim_3']
        dim_4 = params['dim_4']
        dim_5 = params['dim_5']
        self.drop_prob = params['dropout']
        self.drop = nn.Dropout(self.drop_prob)
        self.lr = params['lr']
        self.activation = params['activation']()
        self.input_size = len(params['features'])
        self.output_size = 1
        self.loss = params['loss']()
        self.corr_loss = params['corr_loss']()
        if params['embedding']:
            cat_dims = [5 for i in range(self.input_size)]
            emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
            self.embedding_layers = nn.ModuleList(
                [nn.Embedding(x, y) for x, y in emb_dims]).to(self.device)
            self.num_embeddings = sum([y for x, y in emb_dims])
            if params.get('hidden_len', None):
                self.input_size = self.num_embeddings + params['hidden_len']
                self.d0 = nn.Linear(self.input_size, dim_1)
                self.d1 = nn.Linear(dim_1 + self.input_size, dim_2)
            else:
                self.d0 = nn.Linear(self.num_embeddings, dim_1)
                self.d1 = nn.Linear(dim_1 + self.num_embeddings, dim_2)

        else:
            self.d0 = nn.Linear(self.input_size, dim_1)
            self.d1 = nn.Linear(dim_1 + self.input_size, dim_2)

        self.d2 = nn.Linear(dim_2 + dim_1, dim_3)
        self.d3 = nn.Linear(dim_3 + dim_2, dim_4)
        self.d4 = nn.Linear(dim_4 + dim_3, dim_5)
        self.out = nn.Linear(dim_4 + dim_5, self.output_size)

        # Batch Norm
        if params['embedding']:
            if params['hidden_len']:
                self.bn_hidden = nn.BatchNorm1d(params['hidden_len'])
            self.bn0 = nn.BatchNorm1d(self.num_embeddings)
        else:
            self.bn0 = nn.BatchNorm1d(self.input_size)
        self.bn1 = nn.BatchNorm1d(dim_1)
        self.bn2 = nn.BatchNorm1d(dim_2)
        self.bn3 = nn.BatchNorm1d(dim_3)
        self.bn4 = nn.BatchNorm1d(dim_4)
        self.bn5 = nn.BatchNorm1d(dim_5)

    def forward(self, x, hidden=None):
        x = self.bn0(x)
        if getattr(self, 'num_embeddings', None):
            x = [emb_lay(x[:, i])
                 for i, emb_lay in enumerate(self.embedding_layers)]
            x = torch.cat(x, 1)
        if hidden is not None:
            hidden = self.bn_hidden(hidden)
            x = torch.cat([x, hidden], 1)

        # block 0
        x1 = self.d0(x)
        x1 = self.bn1(x1)
        x1 = self.activation(x1)
        x1 = self.drop(x1)

        x = torch.cat([x, x1], 1)

        # block 1
        x2 = self.d1(x)
        x2 = self.bn2(x2)
        x2 = self.activation(x2)
        x2 = self.drop(x2)

        x = torch.cat([x1, x2], 1)

        # block 2
        x3 = self.d2(x)
        x3 = self.bn3(x3)
        x3 = self.activation(x3)
        x3 = self.drop(x3)

        x = torch.cat([x2, x3], 1)

        # block 3
        x4 = self.d3(x)
        x4 = self.bn4(x4)
        x4 = self.activation(x4)
        x4 = self.drop(x4)

        x = torch.cat([x3, x4], 1)

        # block 4
        x5 = self.d4(x)
        x5 = self.bn5(x5)
        x5 = self.activation(x5)
        x5 = self.drop(x5)

        x = torch.cat([x4, x5], 1)

        out = self.out(x)
        return out

    def training_step(self, batch, batch_idx):
        x, hidden, y = batch['data'], batch.get(
            'hidden', None), batch['target']
        x = x.view(x.size(1), -1)
        y = y.T
        if hidden is not None:
            hidden = hidden.view(hidden.size(1), -1)
        logits = self(x, hidden)
        loss = self.loss(input=logits, target=y)
        corr = self.corr_loss(input=logits, target=y).cuda()
        loss += (1 - corr) * 0.05
        self.log('train_loss', loss, prog_bar=True,
                 on_epoch=True, on_step=False)
        self.log('train_corr', corr, prog_bar=True,
                 on_epoch=True, on_step=False)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, hidden, y = batch['data'], batch.get(
            'hidden', None), batch['target']
        x = x.view(x.size(1), -1)
        y = y.T
        if hidden is not None:
            hidden = hidden.view(hidden.size(1), -1)
        logits = self(x, hidden)
        loss = self.loss(input=logits, target=y)
        corr = self.corr_loss(input=logits, target=y).cuda()
        loss += (1 - corr) * 0.05
        return {'val_loss': loss, 'val_corr': corr}

    def validation_epoch_end(self, val_step_outputs):
        epoch_loss = torch.tensor([x['val_loss']
                                   for x in val_step_outputs]).mean()
        epoch_corr = torch.tensor([x['val_corr']
                                   for x in val_step_outputs]).mean()
        self.log('val_loss', epoch_loss, prog_bar=True)
        self.log('val_corr', epoch_corr, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        epoch_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        self.log('test_loss', epoch_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.1, min_lr=1e-7, eps=1e-08
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


def correlation(predictions, targets):
    ranked_preds = predictions.rank(pct=True, method="first")
    return np.corrcoef(ranked_preds, targets)[0, 1]


# convenience method for scoring
def scoring(df):
    return correlation(df['preds'], df['target'])


# Payout is just the score cliped at +/-25%
def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)


def cross_val(p) -> dict:
    data = load_data(root_dir='./data/', mode='train')
    data, target, features, era = preprocess_data(
        data, nn=True)
    data_dict = {'data':     data, 'target': target,
                 'features': features, 'era': era}
    p['features'] = [feat for feat in range(len(data_dict['features']))]
    gts = GroupKFold(n_splits=10)
    models = {}
    for i, (train_idx, val_idx) in enumerate(gts.split(data, groups=era)):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath='./lightning_logs', monitor='val_loss',
                                                           save_top_k=1, period=10,
                                                           mode='min'
                                                           )
        model = ResNet(p)
        if p['activation'] == nn.ReLU:
            model.apply(lambda m: init_weights(m, 'relu'))
        elif p['activation'] == nn.LeakyReLU:
            model.apply(lambda m: init_weights(m, 'leaky_relu'))
        dataset = FinData(data=data, target=target, era=era)
        data_loaders = create_dataloaders(
            dataset, indexes={'train': train_idx.tolist(), 'val': val_idx.tolist()}, batch_size=p['batch_size'])
        es = EarlyStopping(monitor='val_loss', patience=10,
                           min_delta=0.000005, mode='min')
        trainer = pl.Trainer(max_epochs=100,
                             gpus=1,
                             callbacks=[checkpoint_callback, es])
        trainer.fit(
            model, train_dataloader=data_loaders['train'], val_dataloaders=data_loaders['val'])
        best_model = ResNet.load_from_checkpoint(
            checkpoint_callback.best_model_path, params=p)
        models[f'fold_{i}'] = best_model
        torch.save(model.state_dict(), f'./saved_models/ResNet/cross_val_{i}.pkl')
    return models


def main():
    p = {'dim_1':           500,
         'dim_2':           600,
         'dim_3':           500,
         'dim_4':           250,
         'dim_5':           50,
         'activation':      nn.LeakyReLU,
         'dropout':         0.21062362698532755,
         'lr':              0.0022252024054478523,
         'label_smoothing': 0.05564974140461841,
         'weight_decay':    0.04106097088288333,
         'amsgrad':         True,
         'batch_size':      10072,
         'loss':            nn.MSELoss,
         'corr_loss':       CorrLoss,
         'embedding':       False}
    models = cross_val(p)
    data = load_data(root_dir='./data', mode='test')
    data = data[data['data_type'] == 'validation']
    data, target, features, era = preprocess_data(data, nn=True)
    data = torch.Tensor(data)
    preds = []
    for i, (key, model) in enumerate(models.items()):
        model.eval()
        pred = model(data)
        preds.append(pred.detach().cpu().numpy().reshape(-1))

    preds = np.mean(preds, axis=1)
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
    print('mean_squared_error', mean_squared_error_)

    if __name__ == '__main__':
        main()
