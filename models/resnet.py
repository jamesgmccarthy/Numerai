import torch
import copy
import os
import numpy as np
import pytorch_lightning as pl
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data_loading.purged_group_time_series import PurgedGroupTimeSeriesSplit
from data_loading.utils import load_data, preprocess_data, FinData, create_dataloaders, calc_data_mean, init_weights


class ResNet(pl.LightningModule):
    def __init__(self, input_size, output_size, params):
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
        self.input_size = input_size
        self.output_size = output_size
        self.loss = params['loss']()
        if params['embedding']:
            cat_dims = [5 for i in range(input_size)]
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
            self.d1 = nn.Linear(dim_1+self.input_size, dim_2)

        self.d2 = nn.Linear(dim_2 + dim_1, dim_3)
        self.d3 = nn.Linear(dim_3 + dim_2, dim_4)
        self.d4 = nn.Linear(dim_4 + dim_3, dim_5)
        self.out = nn.Linear(dim_4+dim_5, output_size)

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

    def forward(self, x, hidden):
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
        mse = mean_squared_error(y_true=y.cpu().numpy(),
                                 y_pred=logits.cpu().detach().numpy())
        self.log('train_mse', mse, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, hidden, y = batch['data'], batch.get(
            'hidden', None), batch['target']
        x = x.view(x.size(1), -1)
        y = y.T
        if hidden is not None:
            hidden = hidden.view(hidden.size(1), -1)
        logits = self(x, hidden)
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
        epoch_mse = torch.tensor([x['mse'] for x in outputs]).mean()
        self.log('test_loss', epoch_loss)
        self.log('test_auc', epoch_mse)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.1, min_lr=1e-7, eps=1e-08
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_mse'}


def cross_val(p) -> object:
    data_ = load_data(root_dir='./data/', mode='train')
    data_, target_, features, date = preprocess_data(
        data_, nn=True, action='multi')
    input_size = data_.shape[-1]
    output_size = target_.shape[-1]
    gts = PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=5)
    models = []
    tb_logger = pl_loggers.TensorBoardLogger('logs/multiclass_')
    for i, (train_idx, val_idx) in enumerate(gts.split(data_, groups=date)):
        idx = np.concatenate([train_idx, val_idx])
        data = copy.deepcopy(data_[idx])
        target = copy.deepcopy(target_[idx])
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            os.path.join('models/', 'multi_class_fold_{}'.format(i)), monitor='val_auc', save_top_k=1, period=10,
            mode='max'
        )
        model = ResNet(input_size, output_size, p)
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
        dataset = FinData(data=data, target=target, date=date, multi=True)
        dataloaders = create_dataloaders(
            dataset, indexes={'train': train_idx, 'val': val_idx}, batch_size=p['batch_size'])
        es = EarlyStopping(monitor='val_auc', patience=10,
                           min_delta=0.0005, mode='max')
        trainer = pl.Trainer(logger=tb_logger,
                             max_epochs=100,
                             gpus=1,
                             callbacks=[checkpoint_callback, es],
                             precision=16)
        trainer.fit(
            model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
        torch.save(model.state_dict(), f'models/fold_{i}_state_dict.pth')
        models.append(model)
    return models, features


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
                # calc mean of each models prediction of each response rather than mean of all predicted responses by each model
                preds = [torch.sigmoid(model.forward(vals.view(1, -1))).detach().numpy()
                         for model in models]
                pred = np.mean(np.mean(preds, axis=0))
            else:
                pred = torch.sigmoid(models.forward(vals.view(1, -1))).item()
            sample_prediction_df.action = np.where(
                pred > 0.5, 1, 0).astype(int).item()
        else:
            sample_prediction_df.action = 0
        env.predict(sample_prediction_df)


def main():
    p = {'dim_1': 167,
         'dim_2': 454,
         'dim_3': 371,
         'dim_4': 369,
         'dim_5': 155,
         'activation': nn.LeakyReLU,
         'dropout': 0.21062362698532755,
         'lr': 0.0022252024054478523,
         'label_smoothing': 0.05564974140461841,
         'weight_decay': 0.04106097088288333,
         'amsgrad': True,
         'batch_size': 10072}
    models, features = cross_val(p)
    test_model(models, features)


if __name__ == '__main__':
    main()
