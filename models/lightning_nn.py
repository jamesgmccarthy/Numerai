# %%
import copy
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit
from pytorch_lightning import Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from sklearn.metrics import roc_auc_score
from torch.nn.modules import linear
from torch.nn.modules.batchnorm import BatchNorm1d
from tqdm import tqdm
from sklearn.metrics import mean_squared_error


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class Classifier(pl.LightningModule):
    def __init__(self, input_size, output_size, params=None,
                 model_path='models/'):
        super(Classifier, self).__init__()
        dim_1 = params['dim_1']
        dim_2 = params['dim_2']
        dim_3 = params['dim_3']
        dim_4 = params['dim_4']
        dim_5 = params['dim_5']
        self.dropout_prob = params['dropout']
        self.lr = params['lr']
        self.activation = params['activation']
        self.input_size = input_size
        self.output_size = output_size
        self.loss = params['loss']
        self.weight_decay = params['weight_decay']
        self.amsgrad = params['amsgrad']
        self.label_smoothing = params['label_smoothing']
        self.train_log = pd.DataFrame({'auc': [0], 'loss': [0]})
        self.val_log = pd.DataFrame({'auc': [0], 'loss': [0]})
        self.model_path = model_path
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, dim_1, bias=False),
            nn.BatchNorm1d(dim_1),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(dim_1, dim_2, bias=False),
            nn.BatchNorm1d(dim_2),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(dim_2, dim_3, bias=False),
            nn.BatchNorm1d(dim_3),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(dim_3, dim_4, bias=False),
            nn.BatchNorm1d(dim_4),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(dim_4, dim_5, bias=False),
            nn.BatchNorm1d(dim_5),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(dim_5, self.output_size, bias=False)
        )
        self.encoder_1l = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, dim_1, bias=False),
            nn.BatchNorm1d(dim_1),
            self.activation(),
            nn.Dropout(p=self.dropout_prob),
            nn.Linear(dim_1, self.output_size, bias=False)
        )

    def forward(self, x):
        out = self.encoder_1l(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        x = x.view(x.size(1), -1)
        y = y.T
        logits = self(x)
        loss = self.loss(input=logits, target=y)
        mse = mean_squared_error(y_true=y.cpu().numpy(),
                                 y_pred=logits.cpu().detach().numpy())
        self.log('train_mse', mse, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        x = x.view(x.size(1), -1)
        y = y.T
        logits = self(x)
        loss = self.loss(input=logits,
                         target=y)
        mse = mean_squared_error(y_true=y.cpu().numpy(),
                                 y_pred=logits.cpu().detach().numpy())
        return {'loss': loss, 'y': y, 'logits': logits, 'mse': mse}

    def validation_epoch_end(self, val_step_outputs):
        epoch_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        epoch_mse = torch.tensor([x['mse'] for x in val_step_outputs]).mean()
        self.log('val_loss', epoch_loss, prog_bar=True)
        self.log('val_mse', epoch_mse, prog_bar=True)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        epoch_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        epoch_auc = torch.tensor([x['auc'] for x in outputs]).mean()
        self.log('test_loss', epoch_loss)
        self.log('test_auc', epoch_auc)

    def predict(self, batch):
        self.eval()
        x, y = batch['data'], batch['target']
        x = x.view(x.size(1), -1)
        x = self(x)
        return torch.sigmoid(x.view(-1))

    def prediction_loop(self, dataloader, return_tensor=True):
        bar = tqdm(dataloader)
        preds = []
        for batch in bar:
            preds.append(self.predict(batch))
        if return_tensor:
            return torch.cat(preds, dim=0)
        else:
            return preds

    def configure_optimizers(self):
        # weight_decay = self.weight_decay,
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr,
                                     amsgrad=self.amsgrad)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.1, min_lr=1e-7, eps=1e-08
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_mse'}


def init_weights(m, func):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, nn.init.calculate_gain(func))
        # m.bias.data.fill_(1)


def train_cross_val(p):
    data_ = load_data(root_dir='./data/', mode='train')
    data_, target_, features, date = preprocess_data(data_, nn=True)

    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=5)

    input_size = data_.shape[-1]
    output_size = 1
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    models = []
    for i, (train_idx, val_idx) in enumerate(gts.split(data_, groups=date)):
        idx = np.concatenate([train_idx, val_idx])
        data = copy.deepcopy(data_[idx])
        target = copy.deepcopy(target_[idx])
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join('models/', "fold_{}".format(i)), monitor="val_auc", mode='max', save_top_k=1, period=10)
        model = Classifier(input_size=input_size,
                           output_size=output_size, params=p)
        if p['activation'] == nn.ReLU:
            model.apply(lambda m: init_weights(m, 'relu'))
        elif p['activation'] == nn.LeakyReLU:
            model.apply(lambda m: init_weights(m, 'leaky_relu'))
        train_idx = [i for i in range(0, max(train_idx) + 1)]
        val_idx = [i for i in range(len(train_idx), len(idx))]
        data[train_idx] = calc_data_mean(
            data[train_idx], './cache', train=True, mode='mean')
        data[val_idx] = calc_data_mean(
            data[val_idx], './cache', train=False, mode='mean')
        dataset = FinData(data=data, target=target, date=date)
        dataloaders = create_dataloaders(
            dataset, indexes={'train': train_idx, 'val': val_idx}, batch_size=p['batch_size'])
        es = EarlyStopping(monitor='val_auc', patience=10,
                           min_delta=0.0005, mode='max')
        trainer = pl.Trainer(logger=tb_logger,
                             max_epochs=500,
                             gpus=1,
                             callbacks=[checkpoint_callback, es],
                             precision=16)
        trainer.fit(
            model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
        torch.save(model.state_dict(), f'models/fold_{i}_state_dict.pth')
        models.append(model)
    return models, features


def final_train(p, load=False):
    data_ = load_data(root_dir='./data/', mode='train')
    data, target, features, date = preprocess_data(data_, nn=True)
    input_size = data.shape[-1]
    output_size = 1
    train_idx, val_idx = date[date <= 450].index.values.tolist(
    ), date[date > 450].index.values.tolist()
    data[train_idx] = calc_data_mean(data[train_idx], './cache', train=True)
    data[val_idx] = calc_data_mean(data[val_idx], './cache', train=False)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath='models/full_train',
                                                       monitor="val_auc", mode='max', save_top_k=1, period=10)
    model = Classifier(input_size=input_size,
                       output_size=output_size, params=p)
    if p['activation'] == nn.ReLU:
        model.apply(lambda m: init_weights(m, 'relu'))
    elif p['activation'] == nn.LeakyReLU:
        model.apply(lambda m: init_weights(m, 'leaky_relu'))
    dataset = FinData(data, target, date)
    dataloaders = create_dataloaders(
        dataset, indexes={'train': train_idx, 'val': val_idx}, batch_size=p['batch_size'])
    es = EarlyStopping(monitor='val_auc', patience=10,
                       min_delta=0.0005, mode='max')
    trainer = pl.Trainer(max_epochs=500,
                         gpus=1,
                         callbacks=[checkpoint_callback, es],
                         precision=16)
    trainer.fit(
        model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
    torch.save(model.state_dict(), 'models/final_train.pth')
    return model, features


def fillna_npwhere(array, values):
    if np.isnan(array.sum()):
        array = np.nan_to_num(array) + np.isnan(array) * values
    return array


def test_model(models, features, cache_dir='cache'):
    env = janestreet.make_env()
    iter_test = env.iter_test()
    if type(models) == list:
        models = [model.eval() for model in models]
    else:
        models.eval()
    f_mean = np.load(f'{cache_dir}/f_mean.npy')
    for (test_df, sample_prediction_df) in tqdm(iter_test):
        if test_df['weight'].item() > 0:
            vals = torch.FloatTensor(
                fillna_npwhere(test_df[features].values, f_mean))
            if type(models) == list:
                preds = [torch.sigmoid(model.forward(vals.view(1, -1))).item()
                         for model in models]
                pred = np.median(preds)
            else:
                pred = torch.sigmoid(models.forward(vals.view(1, -1))).item()
            sample_prediction_df.action = np.where(
                pred > 0.5, 1, 0).astype(int).item()
        else:
            sample_prediction_df.action = 0
        env.predict(sample_prediction_df)


# %%


def main(train=True):
    p = {'batch_size':   4986, 'dim_1': 248, 'dim_2': 487,
         'dim_3':        269, 'dim_4': 218, 'dim_5': 113,
         'activation':   nn.ReLU, 'dropout': 0.01563457578202565,
         'lr':           0.00026372556533974916, 'label_smoothing': 0.06834918091900156,
         'weight_decay': 0.005270589494631074, 'amsgrad': False}
    if train:
        models, features = train_cross_val(p)
        # models, features = final_train(p, load=False)
    else:
        data_ = load_data(root_dir='./data/', mode='train')
        data_, target_, features, date = preprocess_data(data_, nn=True)
        model_path = '/kaggle/input/model-files'
        f_mean = calc_data_mean(data_, 'cache')
        models = load_model(model_path, data_.shape[-1], 1, p, False)
    # model, checkpoint = final_train(p)
    # best_model_path = checkpoint.best_model_path
    # model, features = final_train(load=best_model_path)
    test_model(models, features)
    return models


if __name__ == '__main__':
    model = main()

# %%
