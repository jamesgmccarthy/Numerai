import numpy as np
import pandas as pd
from data_loading import utils as d_utils
import json


def create_corr_matrix(df, features=None, era=False, target='target'):
    target_ = ['target'] if target == 'target' else [col for col in df.columns if col.startswith('target')]
    if features:
        features = [item for sublist in list(features.values()) for item in sublist]  # change dict to list
    else:
        features = [col for col in df.columns if col.startswith('feature')]
        meta_features = ['era', 'data_type']  # already added above
        features = features + meta_features
    df = df[features + target_]
    if era:
        corr_matrix = df.groupby('era').corr().reset_index()
        corr_matrix.rename(columns={'level_1': 'feature_names'})
    else:
        corr_matrix = df.corr()
    return corr_matrix


def main():
    root_dir = './data'
    df = d_utils.load_data(root_dir, mode='train')
    features = d_utils.get_feature_sets(root_dir)
    era = True
    corr_matrix = create_corr_matrix(df, era=era)
    corr_matrix.to_parquet(f'{root_dir}/datasets/corr_matrix_era_{era}.parquet')


if __name__ == '__main__':
    main()
