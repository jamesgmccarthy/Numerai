import os
import pickle

import catboost as cat
import lightgbm as lgb
import numerapi
import torch.nn as nn
from dotenv import load_dotenv

from data_loading import utils
from metrics.corr_loss_function import CorrLoss
from models import resnet as res


def credentials(override=False):
    dotenv_path = 'num_config.env'
    load_dotenv(dotenv_path=dotenv_path, override=override)
    pub_id = os.getenv('PUBLIC_ID')
    priv_key = os.getenv('PRIVATE_KEY')
    latest_round = os.getenv('LATEST_ROUND')
    return {'PUBLIC_ID': pub_id, 'PRIVATE_KEY': priv_key, 'LATEST_ROUND': latest_round}


def download_data(api: numerapi.NumerAPI, keys):
    if int(keys['LATEST_ROUND']) == api.get_current_round():
        return int(keys['LATEST_ROUND'])
    else:
        LATEST_ROUND = api.get_current_round()
        api.download_current_dataset('./data')
        return LATEST_ROUND


def update_env_file(env_vars):
    with open('num_config.env', 'w') as f:
        f.write(f'LATEST_ROUND={env_vars["LATEST_ROUND"]}\n')
        f.write(f'PUBLIC_ID={env_vars["PUBLIC_ID"]}\n')
        f.write(f'PRIVATE_KEY={env_vars["PRIVATE_KEY"]}\n')


def create_preds():
    paths = {  # 'xgb':           './saved_models/xgb/full_train/2021-05-16/',
        'xgb_cross_val': './saved_models/xgb/cross_val/2021-05-16/',
        # 'lgb':           './saved_models/lgb/full_train/2021-05-16/',
        'lgb_cross_val': './saved_models/lgb/cross_val/2021-05-16/',
        # 'cat':           './saved_models/cat/full_train/2021-05-16/',
        'cat_cross_val': './saved_models/cat/cross_val/2021-05-16/'}
    models_loaded = load_models(paths)
    utils.create_predictions(models=models_loaded)
    utils.create_prediction_file()


def load_models(paths):
    p_res = {'dim_1':      935, 'dim_2': 2698, 'dim_3': 1016,
             'dim_4':      609, 'dim_5': 236, 'activation': nn.GELU,
             'dropout':    0.28316234733039947, 'lr': 0.028606289302768702,
             'batch_size': 12963, 'loss': nn.MSELoss, 'corr_loss': CorrLoss, 'embedding': False,
             'features':   [feat for feat in range(310)]}

    models_loaded = {'xgb': [], 'lgb': [], 'cat': []}
    for model in paths.keys():
        for file in sorted(os.listdir(paths[model])):
            if model == 'xgb_cross_val' or model == 'xgb':
                models_loaded['xgb'].append(
                    pickle.load(open(f'{paths[model]}{file}', 'rb')))
            if model == 'lgb' or model == 'lgb_cross_val':
                models_loaded['lgb'].append(
                    lgb.Booster(model_file=f'{paths[model]}{file}'))
            if model == 'ae':
                models_loaded['ae'].append(utils.load_model(
                    path=f'./saved_models/trained/{file}', p=p_ae, pl_lightning=False, model=SupAE))
            if model == 'cat' or model == 'cat_cross_val':
                models_loaded['cat'].append(
                    cat.CatBoostRegressor().load_model(f'{paths[model]}{file}'))
            if model == 'resnet' or model == 'resnet_cross_val':
                print(f'{paths[model]}/{file}')
                models_loaded['resnet'].append(utils.load_model(
                    path=f'{paths[model]}/{file}', p=p_res, pl_lightning=False, model=res.ResNet))

    return models_loaded


def main():
    keys = credentials()
    numapi = numerapi.NumerAPI(
        verbosity='INFO', public_id=keys['PUBLIC_ID'], secret_key=keys['PRIVATE_KEY'])
    keys['LATEST_ROUND'] = download_data(numapi, keys)
    update_env_file(keys)
    keys = credentials(override=True)
    utils.seed_everything(0)

    # gbm_hpo.main()
    # ae_hpo.main(embedding=False)
    # gbm_hpo.main(ae_train=True)
    # nn_hpo.main(train_ae=False)
    # train_utils.main()
    # models = load_models('./saved_models/trained/cross_val/')
    # utils.create_predictions(models=models)
    # utils.create_prediction_file()
    # fenn.main()
    # res.main()
    # train_utils.main()
    create_preds()
    # res.main()


if __name__ == '__main__':
    main()
