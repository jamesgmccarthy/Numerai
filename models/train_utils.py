import joblib
import data_loading.utils as utils
from models.SupervisedAutoEncoder import SupAE
from hpo.ae_hpo import create_param_dict
from models.resnet import ResNet
import xgboost as xgb
import lightgbm as lgb


def train(models):
    