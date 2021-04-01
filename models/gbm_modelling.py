# %%
import datetime
import gc

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


def optimize(trial: optuna.trial.Trial, datasets):
    p = {'learning_rate': trial.suggest_uniform('learning_rate', 1e-4, 1e-1),
         'max_depth': trial.suggest_int('max_depth', 5, 30),
         'max_leaves': trial.suggest_int('max_leaves', 5, 50),
         'subsample': trial.suggest_uniform('subsample', 0.3, 1.0),
         'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.3, 1.0),
         'min_child_weight': trial.suggest_int('min_child_weight', 5, 100),
         'lambda': trial.suggest_uniform('lambda', 0.05, 0.2),
         'alpha': trial.suggest_uniform('alpha', 0.05, 0.2),
         'objective': 'regression',
         'booster': 'gbtree',
         'tree_method': 'gpu_hist',
         'verbosity': 1,
         'n_jobs': 10,
         'eval_metric': 'mse'}
    print('Choosing parameters:', p)
    scores = []
    sizes = []
    # gts = GroupTimeSeriesSplit()
    gts = pgs.PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=31)
    for i, (tr_idx, val_idx) in enumerate(gts.split(datasets['data'], groups=datasets['era'])):
        sizes.append(len(tr_idx))
        x_tr, x_val = datasets['data'][tr_idx], datasets['data'][val_idx]
        y_tr, y_val = datasets['target'][tr_idx], datasets['target'][val_idx]
        d_tr = xgb.DMatrix(x_tr, y_tr)
        d_val = xgb.DMatrix(x_val, y_val)
        clf = xgb.train(p, d_tr, 1000, [
            (d_val, 'eval')], early_stopping_rounds=50, verbose_eval=True)
        val_pred = clf.predict(d_val)
        score = mean_squared_error(y_val, val_pred)
        scores.append(score)
        del clf, val_pred, d_tr, d_val, x_tr, x_val, y_tr, y_val, score
        rubbish = gc.collect()
    print(scores)
    avg_score = utils.weighted_mean(scores, sizes)
    print('Avg Score:', avg_score)
    return avg_score


def loptimize(trial, datasets):
    p = {'learning_rate': trial.suggest_uniform('learning_rate', 1e-4, 1e-1),
         'max_leaves': trial.suggest_int('max_leaves', 5, 100),
         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.3, 0.99),
         'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.3, 0.99),
         'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 1000),
         'lambda_l1': trial.suggest_uniform('lambda_l1', 0.005, 0.05),
         'lambda_l2': trial.suggest_uniform('lambda_l2', 0.005, 0.05),
         'boosting': trial.suggest_categorical('boosting', ['gbdt', 'goss', 'rf']),
         'objective': 'binary',
         'verbose': 1,
         'n_jobs': 10,
         'metric': 'auc'}
    if p['boosting'] == 'goss':
        p['bagging_freq'] = 0
        p['bagging_fraction'] = 1.0
    scores = []
    sizes = []
    # gts = GroupTimeSeriesSplit()
    gts = pgs.PurgedGroupTimeSeriesSplit(n_splits=5, group_gap=31)
    for i, (tr_idx, val_idx) in enumerate(gts.split(datasets['data'], groups=datasets['era'])):
        sizes.append(len(tr_idx))
        x_tr, x_val = datasets['data'][tr_idx], datasets['data'][val_idx]
        y_tr, y_val = datasets['target'][tr_idx], datasets['target'][val_idx]
        train = lgb.Dataset(x_tr, label=y_tr)
        val = lgb.Dataset(x_val, label=y_val)
        clf = lgb.train(p, train, 1000, valid_sets=[
            val], early_stopping_rounds=50, verbose_eval=True)
        preds = clf.predict(x_val)
        score = mean_squared_error(y_val, preds)
        scores.append(score)
        del clf, preds, train, val, x_tr, x_val, y_tr, y_val, score
        rubbish = gc.collect()
    print(scores)
    avg_score = utils.weighted_mean(scores, sizes)
    print('Avg Score:', avg_score)
    return avg_score


def main():
    api_token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYWQxMjg3OGEtMGI1NC00NzFmLTg0YmMtZmIxZjcxZDM2NTAxIn0='
    neptune.init(api_token=api_token,
                 project_qualified_name='jamesmccarthy65/Numerai')
    data = utils.load_data('data/', mode='train')
    data, target, features, era = utils.preprocess_data(data, nn=True)
    datasets = {'data': data, 'target': target,
                'features': features, 'era': era}
    print('creating XGBoost Trials')
    xgb_exp = neptune.create_experiment('XGBoost_HPO')
    xgb_neptune_callback = opt_utils.NeptuneCallback(experiment=xgb_exp)
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda x: optimize(x, datasets), n_trials=10,
                   callbacks=[xgb_neptune_callback])
    joblib.dump(
        study, f'HPO/xgb_hpo_{str(datetime.datetime.now().date())}.pkl')
    print('Creating LightGBM Trials')
    lgb_exp = neptune.create_experiment('LGBM_HPO')
    lgbm_neptune_callback = opt_utils.NeptuneCallback(experiment=lgb_exp)
    study = optuna.create_study(direction='minimize')
    study.optimize(loptimize, n_trials=10, callbacks=[lgbm_neptune_callback])
    joblib.dump(
        study, f'HPO/lgb_hpo_{str(datetime.datetime.now().date())}.pkl')


if __name__ == '__main__':
    main()
