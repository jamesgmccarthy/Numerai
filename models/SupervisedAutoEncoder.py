import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.nn.modules import dropout
from torch.nn.modules.batchnorm import BatchNorm1d
from data_loading import utils
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from models import utils as m_utils


class SupAE(pl.LightningModule):
    def __init__(self, params):
        super(SupAE, self).__init__()
        self.lr = params['lr']
        self.loss_recon = params['loss_recon']()
        self.recon_loss_factor = params['recon_loss_factor']
        self.loss_sup_ae = params['loss_sup_ae']()
        self.reg_loss = params['loss_reg']()
        self.activation = params['activation']
        self.drop = params['dropout']
        self.drop_ae = params['dropout_ae']
        self.emb = params['emb']
        if self.emb:
            cat_dims = [5 for i in range(params['input_size'])]
            emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
            self.embedding_layers = nn.ModuleList(
                [nn.Embedding(x, y) for x, y in emb_dims]).to(self.device)
            self.input_size = sum([y for x, y in emb_dims])
        else:
            self.input_size = params['input_size']
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(self.input_size),
            m_utils.GaussianNoise(),
            nn.Linear(self.input_size, params['hidden']),
            nn.BatchNorm1d(params['hidden']),
            self.activation(),
            # nn.Dropout(self.drop_ae),
            # nn.Linear(params['dim_1'], params['dim_2']),
            # nn.BatchNorm1d(params['dim_2']),
            # self.activation(),
            # nn.Linear(params['dim_2'], params['hidden']),
        )
        self.decoder = nn.Sequential(
            nn.Dropout(self.drop),
            nn.Linear(params['hidden'], self.input_size)
        )
        self.MLP = nn.Sequential(
            nn.Linear(self.input_size + params['hidden'], params['dim_1']),
            nn.BatchNorm1d(params['dim_1']),
            self.activation(),
            nn.Dropout(self.drop),
            nn.Linear(params['dim_1'], params['dim_2']),
            nn.BatchNorm1d(params['dim_2']),
            self.activation(),
            nn.Dropout(self.drop),
            nn.Linear(params['dim_2'], params['dim_3']),
            nn.BatchNorm1d(params['dim_3']),
            self.activation(),
            nn.Dropout(self.drop),
            nn.Linear(params['dim_3'], params['output_size']),
            nn.Sigmoid()
        )

        """
       nn.BatchNorm1d(params['dim_2']),
       self.activation(),
       nn.Dropout(self.drop_ae),
       nn.Linear(params['dim_2'], params['dim_1']),
       nn.Dropout(self.drop_ae),
       nn.Linear(params['dim_1'], self.input_size)
       """
        self.regressor = nn.Sequential(
            nn.Linear(self.input_size, params['hidden']),
            nn.BatchNorm1d(params['hidden']),
            self.activation(),
            nn.Dropout(self.drop),
            nn.Linear(params['hidden'], params['output_size']),
            nn.Sigmoid()
        )

    def forward(self, x):
        if self.emb:
            x = [emb_lay(x[:, i])
                 for i, emb_lay in enumerate(self.embedding_layers)]
            x = torch.cat(x, 1)
        hidden = self.encoder(x)
        decoder_out = self.decoder(hidden)
        sup_ae = self.regressor(decoder_out)
        x = nn.BatchNorm1d(self.input_size).to('cuda')(x)
        x = nn.Dropout(self.drop).to('cuda')(x)
        reg_in = torch.cat([x, hidden], 1)
        reg_out = self.MLP(reg_in)
        return hidden, decoder_out, sup_ae, reg_out

    def training_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        x = x.view(x.size(1), -1)
        y = y.T
        embedding, decoder_out, sup_ae, reg_out = self(x)
        if self.emb:
            recon_loss = torch.mean(torch.tensor(
                [self.loss_recon(decoder_out[i], x[i]) for i in range(x.shape[0])]))
        else:
            recon_loss = self.loss_recon(decoder_out, x)
        sup_loss = self.loss_sup_ae(sup_ae, y)
        reg_loss = self.reg_loss(reg_out, y)
        loss = reg_loss + sup_loss + (self.recon_loss_factor * recon_loss)
        self.log('reg_loss', reg_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('sup_loss', sup_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('recon_loss', recon_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        x = x.view(x.size(1), -1)
        y = y.T
        embedding, decoder_out, sup_ae, reg_out = self(x)
        if self.emb:
            recon_loss = torch.mean(torch.tensor(
                [self.loss_recon(decoder_out[i], x[i]) for i in range(x.shape[0])]))
        else:
            recon_loss = self.loss_recon(decoder_out, x)
        sup_loss = self.loss_sup_ae(sup_ae, y)
        reg_loss = self.reg_loss(reg_out, y)
        loss = reg_loss + sup_loss + (self.recon_loss_factor * recon_loss)
        return {'val_loss': loss, 'val_sup_loss': sup_loss, 'val_reg_loss': reg_loss, 'val_recon_loss': recon_loss}

    def validation_epoch_end(self, outputs) -> None:
        epoch_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        epoch_sup_loss = torch.tensor(
            [x['val_sup_loss'] for x in outputs]).mean()
        epoch_reg_loss = torch.tensor(
            [x['val_reg_loss'] for x in outputs]).mean()
        epoch_recon_loss = torch.tensor(
            [x['val_recon_loss'] for x in outputs]).mean()
        self.log('val_reg_loss', epoch_reg_loss, prog_bar=True)
        self.log('val_sup_loss', epoch_sup_loss, prog_bar=True)
        self.log('val_recon_loss', epoch_recon_loss, prog_bar=True)
        self.log('val_loss', epoch_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.1, min_lr=1e-7, eps=1e-08
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


def train_ae_model(data_dict):
    """Deprecated"""
    # TODO Dynamic
    p = joblib.load('./hpo/params/ae_sup_params.pkl').best_params
    act_dict = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU,
                'gelu': nn.GELU, 'silu': nn.SiLU}
    p['activation'] = act_dict[p['activation']]
    p['loss_sup_ae'] = nn.MSELoss
    p['loss_recon'] = nn.MSELoss
    p['embedding'] = False
    # if %
    t_idx = np.where(data_dict['era'] < 121)[0].tolist()
    v_idx = np.where(data_dict['era'] >= 121)[0].tolist()
    p['input_size'] = len(data_dict['features'])
    p['output_size'] = 1
    dataset = utils.FinData(
        data=data_dict['data'], target=data_dict['target'], era=data_dict['era'])
    dataloaders = utils.create_dataloaders(dataset=dataset, indexes={
        'train': t_idx, 'val': v_idx}, batch_size=p['batch_size'])
    model = SupAE(p)
    es = EarlyStopping(monitor='val_loss', patience=10,
                       min_delta=0.005, mode='min')
    trainer = pl.Trainer(max_epochs=100,
                         gpus=1,
                         callbacks=[es])
    trainer.fit(
        model, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['val'])
    torch.save(model.state_dict(), f'./saved_models/trained/trained_ae.pth')
    return model


def create_hidden_rep(model, data_dict):
    model.eval()
    index = np.linspace(
        0, data_dict['data'].shape[0] - 1, data_dict['data'].shape[0], dtype='int').tolist()
    dataset = utils.FinData(
        data_dict['data'], target=data_dict['target'], era=data_dict['era'])
    batch_size = 4000
    dataloaders = utils.create_dataloaders(
        dataset, {'train': index}, batch_size=batch_size)
    hiddens = []
    predictions = []
    for i, batch in enumerate(dataloaders['train']):
        _, hidden, pred, _ = model(batch['data'].view(
            batch['data'].size(1), -1))
        hiddens.append(hidden.cpu().detach().numpy().tolist())
        predictions.append(pred.cpu().detach().numpy().tolist())
    hiddens = np.array([hiddens[i][j] for i in range(
        len(hiddens)) for j in range(len(hiddens[i]))])
    preds = np.array([predictions[i][j] for i in range(
        len(predictions)) for j in range(len(predictions[i]))])
    return {'hidden': hiddens, 'preds': preds}
