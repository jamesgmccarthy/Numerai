# %%
import datetime
import gc
import copy
import pandas as pd
from git.objects import util
from models.SupervisedAutoEncoder import create_hidden_rep, train_ae_model
import datatable as dt
import joblib
import lightgbm as lgb
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from data_loading import utils
from data_loading import purged_group_time_series as pgs
import catboost as cat
from sklearn.model_selection import GroupKFold


def optimize(trial: optuna.trial.Trial, downsample=None, count=1):
    p = {'learning_rate':    trial.suggest_uniform('learning_rate', 1e-4, 1e-1),
         'max_depth':        trial.suggest_int('max_depth', 5, 30),
         'max_leaves':       trial.suggest_int('max_leaves', 5, 50),
         'subsample':        trial.suggest_uniform('subsample', 0.3, 1.0),
         'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 1.0),
         'min_child_weight': trial.suggest_int('min_child_weight', 5, 100),
         'lambda':           trial.suggest_uniform('lambda', 0.05, 0.2),
         'alpha':            trial.suggest_uniform('alpha', 0.05, 0.2),
         'objective':        'reg:squarederror',
         'booster':          'gbtree',
         'tree_method':      'gpu_hist',
         'verbosity':        1,
         'n_jobs':           10,
         'eval_metric':      'rmse'}
    print('Choosing parameters:', p)
    scores = []
    sizes = []
    # gts =GroupKFold(n_splits=10)

    gts = pgs.PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=5)
    data = utils.get_eras()
    for i, (tr_idx, val_idx) in enumerate(gts.split(data['target'], groups=data['era'])):
        print(f'Training on split {i + 1}')
        x_tr, y_tr, x_val, y_val = utils.data_sampler(tr_idx, val_idx, downsampling=downsample, count=count)
        d_tr = xgb.DMatrix(x_tr, label=y_tr)
        del x_tr
        d_val = xgb.DMatrix(x_val, label=y_val)
        del x_val, y_tr
        gc.collect()
        clf = xgb.train(p, d_tr, 500, [
            (d_val, 'eval')], early_stopping_rounds=50, verbose_eval=True)
        del d_tr
        gc.collect()
        val_pred = clf.predict(d_val)
        score = mean_squared_error(y_val, val_pred)
        scores.append(score)
        sizes.append(len(tr_idx) + len(val_idx))
        del clf, val_pred, d_val, y_val, score
        gc.collect()
    print(scores)
    avg_score = utils.weighted_mean(scores, sizes)
    print('Avg Score:', avg_score)
    return avg_score


def loptimize(trial, downsample=1, count=1):
    p = {'learning_rate':    trial.suggest_uniform('learning_rate', 1e-5, 1e-1),
         'max_leaves':       trial.suggest_int('max_leaves', 5, 100),
         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.3, 0.99),
         'bagging_freq':     trial.suggest_int('bagging_freq', 1, 10),
         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.3, 0.99),
         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 1000),
         'lambda_l1':        trial.suggest_uniform('lambda_l1', 0.005, 0.05),
         'lambda_l2':        trial.suggest_uniform('lambda_l2', 0.005, 0.05),
         'boosting':         trial.suggest_categorical('boosting', ['gbdt', 'goss', 'rf']),
         'objective':        'regression',
         'verbose':          1,
         'n_jobs':           10,
         'metric':           'mse',
         'seed':             0}
    if p['boosting'] == 'goss':
        p['bagging_freq'] = 0
        p['bagging_fraction'] = 1.0
    scores = []
    sizes = []
    # gts = GroupKFold(n_splits=10)

    gts = pgs.PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=3)
    data = utils.get_eras()
    for i, (tr_idx, val_idx) in enumerate(gts.split(data['target'], groups=data['era'])):
        print(f'Training on split {i}')
        x_tr, y_tr, x_val, y_val = utils.data_sampler(tr_idx, val_idx, downsampling=downsample, count=count)
        train = lgb.Dataset(x_tr, label=y_tr)
        del x_tr
        gc.collect()
        val = lgb.Dataset(x_val, label=y_val)
        del y_tr
        gc.collect()
        clf = lgb.train(p, train, 500, valid_sets=[
            val], early_stopping_rounds=50, verbose_eval=True)
        del train
        preds = clf.predict(x_val)
        score = mean_squared_error(y_val, preds)
        scores.append(score)
        sizes.append(len(tr_idx) + len(val_idx))
        del clf, preds, x_val, val, y_val, score
        gc.collect()
    print(scores)
    avg_score = utils.weighted_mean(scores, sizes)
    print('Avg Score:', avg_score)
    return avg_score


def catboost_optimize(trial, downsample=1, count=1):
    p = {'learning_rate':       trial.suggest_uniform('learning_rate', 1e-5, 1e-1),
         # 'max_leaves':       trial.suggest_int('max_leaves', 5, 50),
         'min_data_in_leaf':    trial.suggest_int('min_data_in_leaf', 50, 1000),
         'l2_leaf_reg':         trial.suggest_uniform('l2_leaf_reg', 0.005, 0.05),
         'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0.05, 0.5),
         'depth':               trial.suggest_int('depth', 5, 15),
         'verbose':             10,
         'loss_function':       'RMSE',
         'eval_metric':         'RMSE',
         'random_seed':         0,
         'bootstrap_type':      'Bayesian',
         'use_best_model':      True}

    scores = []
    sizes = []
    # gts = GroupKFold(n_splits=10)

    gts = pgs.PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=3)
    data = utils.get_eras()
    for i, (tr_idx, val_idx) in enumerate(gts.split(data['target'], groups=data['era'])):
        print(f'Training on split {i}')
        x_tr, y_tr, x_val, y_val = utils.data_sampler(tr_idx, val_idx, downsampling=downsample, count=count)
        model = cat.CatBoostRegressor(iterations=1000, task_type='GPU', **p)
        model.fit(X=x_tr, y=y_tr, eval_set=(x_val, y_val),
                  early_stopping_rounds=50, verbose_eval=10)
        preds = model.predict(x_val)
        score = mean_squared_error(y_val, preds)
        scores.append(score)
        sizes.append(len(tr_idx) + len(val_idx))
        del preds, x_tr, x_val, y_tr, y_val, score
        rubbish = gc.collect()
    print(scores)
    avg_score = utils.weighted_mean(scores, sizes)
    print('Avg Score:', avg_score)
    return avg_score


def main(ae_train=False, old=False):
    if ae_train:
        data = data = utils.load_data('data/', mode='train')
        data, target, features, era = utils.preprocess_data(data, ordinal=True)
        test_data = utils.load_data('data/', mode='test')
        test_data = test_data[test_data['data_type'] == 'validation']
        test_data, test_target, test_features, test_era = utils.preprocess_data(
            test_data, ordinal=True)
        data = np.concatenate([data, test_data], 0)
        target = np.concatenate([target, test_target], 0)
        era = np.concatenate([era, test_era], 0)
        data_dict = {'data':     data, 'target': target,
                     'features': features, 'era': era}
        ae_model = train_ae_model(data_dict=data_dict)
        del data, target, era, data_dict
        data = utils.load_data('data/', mode='train')
        data, target, features, era = utils.preprocess_data(
            data, ordinal=True)
        data_dict = {'data':     data, 'target': target,
                     'features': features, 'era': era}
        ae_output = create_hidden_rep(ae_model, data_dict)
        data = np.concatenate([data, ae_output['hidden']], axis=1)

    else:

        # legacy - can't fit new dataset into memory with val set
        if old:
            data = utils.load_data('data/', mode='train')
            data, target, features, era = utils.preprocess_data(data, nn=True)
            data_dict = {'data':     data, 'target': target,
                         'features': features, 'era': era}
            data = utils.load_data(root_dir='./data', mode='test')
            data = data[data['data_type'] == 'validation']
            data, target, features, era = utils.preprocess_data(data, nn=True)
            data_dict['data'] = np.concatenate(
                [data_dict['data'], data], 0)
            data_dict['target'] = np.concatenate(
                [data_dict['target'], target], 0)
            data_dict['era'] = pd.Series(
                np.concatenate([data_dict['era'], era], 0))

    api_token = utils.read_api_token()
    neptune.init(api_token=api_token,
                 project_qualified_name='jamesmccarthy65/Numerai')
    downsample = 4
    print('creating XGBoost Trials')
    for i in range(downsample):
        xgb_exp = neptune.create_experiment(f'XGBoost_HPO_downsample_{downsample}_{i}')
        xgb_neptune_callback = opt_utils.NeptuneCallback(experiment=xgb_exp)
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: optimize(trial, downsample=downsample, count=i),
                       n_trials=20, callbacks=[xgb_neptune_callback])
        joblib.dump(
            study, f'hpo/params/xgb_hpo_downsample_{i}_{str(datetime.datetime.now().date())}.pkl')

        print('Creating LightGBM Trials')
        lgb_exp = neptune.create_experiment(f'LGBM_HPO_downsample_{downsample}_{i}')
        lgbm_neptune_callback = opt_utils.NeptuneCallback(experiment=lgb_exp)
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: loptimize(trial, downsample=downsample, count=i),
                       n_trials=40, callbacks=[lgbm_neptune_callback])
        joblib.dump(
            study, f'hpo/params/lgb_hpo_downsample_{i}_{str(datetime.datetime.now().date())}.pkl')

        print('Creating Catboost Trials')
        cat_exp = neptune.create_experiment(f'CAT_HPO_downsample_{downsample}_{i}')
        cat_callback = opt_utils.NeptuneCallback(experiment=cat_exp)
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: catboost_optimize(
            trial, downsample=downsample, count=i), n_trials=40, callbacks=[cat_callback])
        joblib.dump(
            study, f'hpo/params/cat_hpo_downsample_{i}_{str(datetime.datetime.now().date())}.pkl')


if __name__ == '__main__':
    main()
