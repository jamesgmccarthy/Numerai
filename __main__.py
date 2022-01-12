import os
import pickle

# poimport catboost as cat
import lightgbm as lgb
import numerapi
import pandas as pd
import torch.nn as nn
from dotenv import load_dotenv

from data_loading import utils
from metrics.corr_loss_function import CorrLoss
from models import resnet as res
from feature_engineering import utils as fe_utils

from hpo import gbm_hpo, ae_hpo, nn_hpo


# from models import SupervisedAutoEncoder, train_utils


def credentials(override=False):
    dotenv_path = 'num_config.env'
    load_dotenv(dotenv_path=dotenv_path, override=override)
    pub_id = os.getenv('PUBLIC_ID')
    priv_key = os.getenv('PRIVATE_KEY')
    latest_round = os.getenv('LATEST_ROUND')
    model_1 = os.getenv('MODEL_1')
    model_2 = os.getenv('MODEL_2')
    return {'PUBLIC_ID': pub_id, 'PRIVATE_KEY': priv_key, 'LATEST_ROUND': latest_round, 'MODEL_1': model_1,
            'MODEL_2':   model_2}


def download_data(api: numerapi.NumerAPI, keys, dataset='live_data'):
    current_round = api.get_current_round()
    if int(keys['LATEST_ROUND']) == current_round and os.path.isfile(f'./data/round_{current_round}/live_data.csv'):
        return int(keys['LATEST_ROUND'])
    else:

        if not os.path.exists(f'./data/round_{current_round}'):
            os.makedirs(f'./data/round_{current_round}')
        api.download_dataset(f'numerai_{dataset}_int8.parquet',
                             dest_path=f'./data/round_{current_round}/{dataset}')
        return current_round


def update_env_file(env_vars):
    with open('num_config.env', 'w') as f:
        f.write(f'LATEST_ROUND={env_vars["LATEST_ROUND"]}\n')
        f.write(f'PUBLIC_ID={env_vars["PUBLIC_ID"]}\n')
        f.write(f'PRIVATE_KEY={env_vars["PRIVATE_KEY"]}\n')
        f.write(f'MODEL_1={env_vars["MODEL_1"]}\n')
        f.write(f'MODEL_2={env_vars["MODEL_2"]}\n')


def create_preds(val=False):
    paths = {  # 'xgb':           './saved_models/xgb/full_train/2021-05-16/',
        # 'xgb_cross_val': './saved_models/xgb/cross_val/2021-05-16/',
        'lgb':           './saved_models/lgb/full_train/2021-10-12/',
        'lgb_cross_val': './saved_models/lgb/cross_val/2021-10-11/',
        'cat':           './saved_models/cat/full_train/2021-10-12/',
        'cat_cross_val': './saved_models/cat/cross_val/2021-10-11/'}
    models_loaded = load_models(paths)
    utils.create_predictions(models=models_loaded, val=val)
    # pred_path = utils.create_prediction_file()
    # return pred_path


def load_models(paths):
    p_res = {'dim_1':      935, 'dim_2': 2698, 'dim_3': 1016,
             'dim_4':      609, 'dim_5': 236, 'activation': nn.GELU,
             'dropout':    0.28316234733039947, 'lr': 0.028606289302768702,
             'batch_size': 12963, 'loss': nn.MSELoss, 'corr_loss': CorrLoss, 'embedding': False,
             'features':   [feat for feat in range(310)]}

    models_loaded = {'lgb': {}, 'cat': {}}
    for model in paths.keys():
        for file in sorted(os.listdir(paths[model])):
            if model == 'xgb_cross_val' or model == 'xgb':
                models_loaded['xgb'][file] = pickle.load(
                    open(f'{paths[model]}{file}', 'rb'))
            if model == 'lgb' or model == 'lgb_cross_val':
                models_loaded['lgb'][file] = lgb.Booster(
                    model_file=f'{paths[model]}{file}')
            if model == 'cat' or model == 'cat_cross_val':
                models_loaded['cat'][file] = cat.CatBoostRegressor(
                ).load_model(f'{paths[model]}{file}')
            if model == 'resnet' or model == 'resnet_cross_val':
                print(f'{paths[model]}/{file}')
                models_loaded['resnet'][file] = utils.load_model(
                    path=f'{paths[model]}/{file}', p=p_res, pl_lightning=False, model=res.ResNet)
    return models_loaded


def main():
    keys = credentials()
    numapi = numerapi.NumerAPI(
        verbosity='INFO', public_id=keys['PUBLIC_ID'], secret_key=keys['PRIVATE_KEY'])
    keys['LATEST_ROUND'] = download_data(numapi, keys, dataset='live_data')
    update_env_file(keys)
    keys = credentials(override=True)
    utils.seed_everything(0)
    # fe_utils.main()
    gbm_hpo.main()
    ae_hpo.main(embedding=False)
    # gbm_hpo.main(ae_train=True)
    # nn_hpo.main(train_ae=False)
    # train_utils.main()
    # models = load_models('./saved_models/trained/cross_val/')
    # utils.create_predictions(models=models)
    # utils.create_prediction_file()
    # fenn.main()
    # res.main()
    # train_utils.main()
    # pred_path = create_preds(val=True)
    # CORR
    # numapi.upload_predictions(file_path=pred_path + 'predictions.csv',
    #                          model_id=keys['MODEL_1'])
    # Corr + 2x MMC
    # numapi.upload_predictions(file_path=pred_path + 'predictions.csv',
    #                          model_id=keys['MODEL_2'])
    # res.main()


if __name__ == '__main__':
    main()
