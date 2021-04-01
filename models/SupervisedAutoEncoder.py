import torch
import torch.nn as nn
import numpy
import pytorch_lightning as pl
from sklearn.metrics import mean_squared_error, mean_absolute_error


class SupAE(pl.LightningModule):
    def __init__(self, params):
        super(SupAE, self).__init__()
        self.lr = params['lr']
        self.loss_recon = params['loss_recon']()
        self.recon_loss_factor = params['recon_loss_factor']
        self.loss_sup_ae = params['loss_sup_ae']()
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
        self.regressor = nn.Sequential(
            nn.BatchNorm1d(params['hidden']),
            self.activation(),
            nn.Dropout(self.drop),
            nn.Linear(params['hidden'], params['output_size'])
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

    def forward(self, x):
        x = [emb_lay(x[:, i])
             for i, emb_lay in enumerate(self.embedding_layers)]
        emb = torch.cat(x, 1)
        hidden = self.encoder(emb)
        reg_out = self.regressor(hidden)
        decoder_out = self.decoder(hidden)
        return emb, hidden, reg_out, decoder_out

    def training_step(self, batch, batch_idx):
        x, y = batch['data'], batch['target']
        x = x.view(x.size(1), -1)
        y = y.T
        emb, _, reg_out, decoder_out = self(x)
        sup_loss = self.loss_sup_ae(reg_out, y)
        recon_loss = torch.mean(torch.tensor(
            [self.loss_recon(decoder_out[i], emb[i]) for i in range(x.shape[0])]))
        loss = sup_loss + self.recon_loss_factor*recon_loss
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
        emb, _, reg_out, decoder_out = self(x)
        sup_loss = self.loss_sup_ae(reg_out, y)
        recon_loss = torch.mean(torch.tensor(
            [self.loss_recon(decoder_out[i], emb[i]) for i in range(x.shape[0])]))
        loss = sup_loss + self.recon_loss_factor*recon_loss
        """
        self.log('sup_loss', sup_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('recon_loss', recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        """
        return {'val_loss': loss, 'val_sup_loss': sup_loss}

    def validation_epoch_end(self, outputs) -> None:
        epoch_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        epoch_sup_loss = torch.tensor(
            [x['val_sup_loss'] for x in outputs]).mean()
        self.log('val_loss', epoch_loss, prog_bar=True)
        self.log('val_sup_loss', epoch_sup_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.1, min_lr=1e-7, eps=1e-08
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
