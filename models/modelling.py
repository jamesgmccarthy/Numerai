# %%
from numpy.lib.function_base import _parse_input_dimensions
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve
import torch
import numpy as np
from torch._C import dtype
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, trainer
import datatable as dt
import pandas as pd
from torch.nn.modules.loss import BCEWithLogitsLoss
from group_time_split import GroupTimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset, dataloader
from torch.utils.data.sampler import BatchSampler, SequentialSampler
from tqdm import tqdm
import copy
import datetime
import matplotlib.pyplot as plt


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, dims, batch_size, learning_rate=0.05, early_stopping=10,
                 model_file='model.pth'):
        super(Classifier, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dims = dims
        self.layer_list = nn.ModuleList()
        self.learning_rate = learning_rate
        self.loss = nn.BCELoss()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.early_stopping = early_stopping
        self.model_file = model_file
        self.batch_size = batch_size
        self.train_log = pd.DataFrame({'auc': [0], 'loss': [0]})
        self.val_log = pd.DataFrame({'auc': [0], 'loss': [0]})
        for i in range(len(self.dims)+1):
            if i == 0:
                self.layer_list.append(
                    nn.Linear(self.input_size, self.dims[i]))
                self.layer_list.append(nn.BatchNorm1d(self.dims[i]))
            elif i == (len(self.dims)):
                self.layer_list.append(
                    nn.Linear(self.dims[i-1], self.output_size))
            else:
                self.layer_list.append(nn.Linear(self.dims[i-1], self.dims[i]))
                self.layer_list.append(nn.BatchNorm1d(self.dims[i]))
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.1, min_lr=1e-7, eps=1e-08
        )

    def forward(self, x):
        for i, layer in enumerate(self.layer_list):
            x = F.dropout(F.leaky_relu(self.layer_list[i](x)), p=0.2)
        return torch.sigmoid(x)

    def training_step(self, batch):
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            x, y = batch['data'].to(
                self.device), batch['target'].to(self.device)
            x = x.reshape(x.size(1), -1)
            y = y.reshape(-1, 1)
            logits = self(x)
            loss = self.loss(input=logits,
                             target=y)
            loss.backward()
            self.optimizer.step()
            return {'loss': loss, 'preds': logits, 'target': y}

    def validation_step(self, batch):
        with torch.set_grad_enabled(False):
            x, y = batch['data'].to(
                self.device), batch['target'].to(self.device)
            x = x.reshape(x.size(1), -1)
            y = y.reshape(-1, 1)
            logits = self(x)
            loss = self.loss(logits,
                             target=y)
            self.scheduler.step(loss)
            return {'loss': loss, 'preds': logits, 'target': y}

    def eval_step(self, data):
        with torch.set_grad_enabled(False):
            x, y = data['data'].to(self.device), data['target'].to(self.device)
            x = x.reshape(x.size(1), -1)
            y = y.reshape(-1, 1)
            preds = self(x)
            return y, preds

    def batch_step_end_metrics(self, num_samples, batch_number, output, running_loss, running_metric):
        running_loss += output['loss'].item()
        running_metric += roc_auc_score(
            output['target'].detach().cpu().numpy(),
            output['preds'].detach().cpu().numpy())
        return running_loss, running_metric

    def epoch_end_metrics(self, outputs):
        auc = torch.tensor([roc_auc_score(
            out['target'].detach().cpu().numpy(),
            out['preds'].detach().cpu().numpy()) for out in outputs])
        loss = torch.stack([out['loss'] for out in outputs])
        return torch.mean(auc), torch.mean(loss)

    def log_results(self, phase, auc, loss):
        if phase == 'train':
            self.train_log = self.train_log.append(
                {'auc': auc.item(), 'loss': loss.item()}, ignore_index=True)
        if phase == 'val':
            self.val_log = self.val_log.append(
                {'auc': auc.item(), 'loss': loss.item()}, ignore_index=True)

    def training_loop(self, epochs, dataloaders):
        es_counter = 0
        auc = {'train': -np.inf, 'eval': -np.inf}
        loss = {'train': np.inf, 'eval': np.inf}
        best_auc = -np.inf
        for e, epoch in enumerate(range(epochs), 1):
            for phase in ['train', 'val']:
                bar = tqdm(dataloaders[phase])
                outs = []
                running_loss = 0.0
                running_auc = 0.0
                for b, batch in enumerate(bar, 1):
                    bar.set_description(f'Epoch {epoch} {phase}'.ljust(20))
                    if phase == 'train':
                        self.train()
                        out = self.training_step(batch)
                    elif phase == 'val':
                        self.eval()
                        out = self.validation_step(batch)
                    outs.append(out)
                    num_samples = batch_size*b
                    running_loss, running_auc = self.batch_step_end_metrics(
                        num_samples, b, out, running_loss, running_auc)
                    bar.set_postfix(loss=f'{running_loss/b:0.5f}',
                                    auc=f'{running_auc/b:0.5f}')
                auc[phase], loss[phase] = self.epoch_end_metrics(outs)
                self.log_results(phase, auc[phase], loss[phase])
                if phase == 'val' and auc['val'] > best_auc:
                    # print('auc_val: ' + auc['val'], 'best_auc: ' + best_auc)
                    best_auc = auc['val']
                    best_model_weights = copy.deepcopy(self.state_dict())
                    torch.save(best_model_weights, self.model_file)
                    es_counter = 0
            es_counter += 1
            if es_counter > self.early_stopping:
                print(
                    f'Early Stopping limit reached. Best Model saved to {self.model_file}')
                print(f'Best Metric achieved: {best_auc}')
                break


class FinData(Dataset):
    def __init__(self, data, target, mode='train', transform=None, cache_dir=None):
        self.data = data
        self.target = target
        self.mode = mode
        self.transform = transform
        self.cache_dir = cache_dir
        self.date = date

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index.to_list()
        if self.transform:
            return self.transform(self.data.iloc[index].values)
        else:
            if type(index) is list:
                sample = {
                    'target': torch.Tensor(self.target.iloc[index].values),
                    'data': torch.FloatTensor(self.data[index]),
                    'date': torch.Tensor(self.date.iloc[index].values)
                }
                return sample
            else:
                sample = {
                    'target': torch.Tensor(self.target.iloc[index]),
                    'data': torch.FloatTensor(self.data[index]),
                    'date': torch.Tensor(self.date.iloc[index])
                }
                return sample

    def __len__(self):
        return len(self.data)


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight, nn.init.calculate_gain('leaky_relu'))
        m.bias.data.fill_(1)

# %%


def load_data(root_dir, mode, overide=None):
    if overide:
        data = dt.fread(overide).to_pandas()
    elif mode == 'train':
        data = dt.fread(root_dir+'train.csv').to_pandas()
    elif mode == 'test':
        data = dt.fread(root_dir+'example_test.csv').to_pandas()
    elif mode == 'sub':
        data = dt.fread(root_dir+'example_sample_submission.csv').to_pandas()
    return data


def preprocess_data(data):
    # data = data.query('weight > 0').reset_index(drop=True)
    data['action'] = ((data['resp'].values) > 0).astype('float32')
    features = [
        col for col in data.columns if 'feature' in col and col != 'feature_0']+['weight']
    for col in features:
        data[col].fillna(data[col].mean(), inplace=True)
    target = data['action']
    date = data['date']
    data = data[features]
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    return data, target, features, date


# %%
data = load_data('data/', mode='train', overide='filtered_train.csv')
data, target, features, date = preprocess_data(data)
# %%
dataset = FinData(data, target, features)
# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dims = [384, 896, 896, 394]
batch_size = 1000
epochs = 100
gts = GroupTimeSeriesSplit()
train_model = False
eval_model = True
for i, (train_idx, val_idx) in enumerate(gts.split(data, groups=date)):
    if train_model:
        model = Classifier(input_size=len(features), output_size=1,
                           dims=dims, batch_size=batch_size,
                           model_file=f'nn_model_fold_{i}.pth').to(device=device)

        # model.apply(init_weights)
        train_set, val_set = Subset(
            dataset, train_idx), Subset(dataset, val_idx)
        train_sampler = BatchSampler(SequentialSampler(
            train_set), batch_size=batch_size, drop_last=False)
        val_sampler = BatchSampler(SequentialSampler(
            val_set), batch_size=batch_size, drop_last=False)
        dataloaders = {'train': DataLoader(dataset, sampler=train_sampler, num_workers=6),
                       'val': DataLoader(dataset, sampler=val_sampler, num_workers=6)}
        model.training_loop(epochs=epochs, dataloaders=dataloaders)
        model.train_log.to_csv(
            f'logs/train_fold_{i}_{str(datetime.datetime.now())}.csv')
        model.val_log.to_csv(
            f'logs/val_fold_{i}_{str(datetime.datetime.now())}.csv')
    if eval_model:
        model = Classifier(input_size=len(features), output_size=1,
                           dims=dims, batch_size=batch_size,
                           model_file=f'nn_model_fold_{i}.pth').to(device=device)
        checkpoint = torch.load(model.model_file)
        model.load_state_dict(checkpoint)
        model.eval()
        val_set = Subset(dataset, val_idx)
        val_sampler = BatchSampler(SequentialSampler(
            val_set), batch_size=batch_size, drop_last=False)
        val_loader = DataLoader(dataset, sampler=val_sampler, num_workers=6)
        bar = tqdm(val_loader)
        all_preds = []
        all_y_true = []
        for b, batch in enumerate(bar, 1):
            bar.set_description(f'Evaluating Model')
            y_true, preds = model.eval_step(batch)
            all_y_true.append(y_true.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
        all_preds = np.concatenate(all_preds, axis=0)
        all_y_true = np.concatenate(all_y_true, axis=0)
        fpr, tpr, _ = roc_curve(all_y_true, all_preds)
        plt.plot(fpr, tpr, label='nn')
        plt.savefig(
            f'plots/val_fold_{i}_roc_curve.png')
# %%
val_set[:]['target']
# %%
