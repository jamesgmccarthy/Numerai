import joblib
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_loading import utils
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


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
        cat_dims = [5 for i in range(params['input_size'])]
        emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in emb_dims]).to(self.device)
        self.num_embeddings = sum([y for x, y in emb_dims])
        self.encoder = nn.Sequential(
            nn.Linear(self.num_embeddings, params['dim_1']),
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
            nn.Linear(params['dim_3'], params['hidden'])
        )
        self.MLP = nn.Sequential(
            nn.BatchNorm1d(self.num_embeddings + params['hidden']),
            nn.Dropout(self.drop),
            nn.Linear(self.num_embeddings + params['hidden'], params['dim_1']),
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
            nn.Linear(params['dim_3'], params['output_size'])
        )
        self.decoder = nn.Sequential(
            nn.Linear(params['hidden'], params['dim_3']),
            nn.BatchNorm1d(params['dim_3']),
            self.activation(),
            nn.Dropout(self.drop),
            nn.Linear(params['dim_3'], params['dim_2']),
            nn.BatchNorm1d(params['dim_2']),
            self.activation(),
            nn.Dropout(self.drop),
            nn.Linear(params['dim_2'], params['dim_1']),
            nn.BatchNorm1d(params['dim_1']),
            self.activation(),
            nn.Dropout(self.drop),
            nn.Linear(params['dim_1'], self.num_embeddings)
        )
        self.regressor = nn.Sequential(
            nn.Linear(self.num_embeddings, params['dim_1']),
            nn.BatchNorm1d(params['dim_1']),
            self.activation(),
            nn.Dropout(self.drop),
            nn.Linear(params['dim_1'], params['output_size'])
        )

    def forward(self, x):
        x = [emb_lay(x[:, i])
             for i, emb_lay in enumerate(self.embedding_layers)]
        emb = torch.cat(x, 1)
        hidden = self.encoder(emb)
        reg_in = torch.cat([emb, hidden], 1)
        reg_out = self.MLP(reg_in)
        decoder_out = self.decoder(hidden)
        sup_ae = self.regressor(decoder_out)
        return emb, hidden, reg_out, decoder_out, sup_ae

    def training_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        x = x.view(x.size(1), -1)
        y = y.T
        emb, _, reg_out, decoder_out, sup_ae = self(x)
        reg_loss = self.reg_loss(reg_out, y)
        sup_loss = self.loss_sup_ae(sup_ae, y)
        recon_loss = torch.mean(torch.tensor(
            [self.loss_recon(decoder_out[i], emb[i]) for i in range(x.shape[0])]))
        loss = reg_loss + sup_loss + (self.recon_loss_factor * recon_loss)
        self.log('reg_loss', reg_loss, on_step=False, on_epoch=True, prog_bar=True)
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
        emb, _, reg_out, decoder_out, sup_ae = self(x)
        reg_loss = self.reg_loss(reg_out, y)
        sup_loss = self.loss_sup_ae(sup_ae, y)
        recon_loss = torch.mean(torch.tensor(
            [self.loss_recon(decoder_out[i], emb[i]) for i in range(x.shape[0])]))
        loss = reg_loss + sup_loss + (self.recon_loss_factor * recon_loss)
        """
        self.log('sup_loss', sup_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        """
        return {'val_loss': loss, 'val_sup_loss': sup_loss, 'val_reg_loss': reg_loss}

    def validation_epoch_end(self, outputs) -> None:
        epoch_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        epoch_sup_loss = torch.tensor(
            [x['val_sup_loss'] for x in outputs]).mean()
        epoch_reg_loss = torch.tensor([x['val_reg_loss'] for x in outputs]).mean()
        self.log('val_loss', epoch_loss, prog_bar=True)
        self.log('val_reg_loss', epoch_reg_loss, prog_bar=True)
        self.log('val_sup_loss', epoch_sup_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.1, min_lr=1e-7, eps=1e-08
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}


def train_ae_model(data_dict):
    # TODO Dynamic
    p = joblib.load('./hpo/params/ae_sup_params.pkl').best_params
    act_dict = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU,
                'gelu': nn.GELU, 'silu': nn.SiLU}
    p['activation'] = act_dict[p['activation']]
    p['loss_sup_ae'] = nn.MSELoss
    p['loss_recon'] = nn.MSELoss
    p['embedding'] = True
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
