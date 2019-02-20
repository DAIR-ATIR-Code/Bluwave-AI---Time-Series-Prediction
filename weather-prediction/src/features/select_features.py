# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from pandas import Series, read_csv
import tensorflow as tf
import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / "src/models/"))
from train_model import create_and_fit_model


def backward_search(data, target, num_features):
    features = list(data.columns)
    features.remove(target)
    for i in range(0, len(features) - num_features):
        feat_scores = Series(index=features)
        # Experiment with the removal of each feature
        for feat in features:
            features_sub = features.copy()
            features_sub.remove(feat)
            # Add target so that model can run properly
            features_sub.append(target)
            metric = run_model_on_feature_subset(data, features_sub, target)
            feat_scores.loc[feat] = metric
        feat_round_ranking = feat_scores.sort_values()
        optimal_feature = feat_round_ranking.index[0]
        features.remove(optimal_feature)
    return features


def forward_search(data, target, num_features):
    features = list(data.columns)
    features.remove(target)
    # Keep target so that model can run properly
    features_sub = [target]
    for i in range(0, num_features):
        feat_scores = Series(index=features)
        # Experiment with the addition of each feature
        for feat in features:
            features_sub.append(feat)
            metric = run_model_on_feature_subset(data, features_sub, target)
            feat_scores.loc[feat] = metric
            features_sub.remove(feat)
        feat_round_ranking = feat_scores.sort_values()
        optimal_feature = feat_round_ranking.index[0]
        features_sub.append(optimal_feature)
        features.remove(optimal_feature)
    return features_sub


def run_model_on_feature_subset(data, target, feat_sub):
    train_end = '2016-12'
    val_start = '2017-01'
    val_end = '2017-12'
    
    train = data.loc[:train_end, feat_sub]
    validate = data.loc[val_start:val_end, feat_sub]
    # Choose reasonable, but not necessarily optimal, hyperparameters
    params = {'num_hidden': 50,
              'learn_rate': 0.001,
              'lambda': 0,
              'dropout': 0.2,
              'num_epochs': 5000,
              'activation': [tf.nn.relu]}
    model, hist, metric, sec = create_and_fit_model(train, validate,
                                                    params, target)
    return metric


def load(logger):
    try:
        data = read_csv(str(project_dir / "processed/all_features.csv"),
                        parse_dates=True, infer_datetime_format=True,
                        index_col=0)
        logger.info('Features data set was loaded.')
    except Exception:
        logger.error('data/processed/all_features.csv could not be read.')
        raise ValueError('DataFrame is empty.')
    return data


def main():
    """ Performs feature selection on data from (../processed) and saves data
        and selected features in (../processed) ready for model training.
        Feature selection methods available are (a) forward and (b) backward
        recursive search.
    """
    logger = logging.getLogger(__name__)
    logger.info('Selecting features from data.')

    target = 'Wind Spd (km/h)'
    train_end = '2016-12'
    num_features = 5
    forward = False
    backward = False

    data = load(logger)
    core_predictors = list(data.columns[:6])
    extra_predictors = list(data.columns[6:])
    
    # Only run feature selection algorithm on predictors of training data
    select_data = data.loc[:train_end, core_predictors]
    
    if (forward):
        select_features = forward_search(select_data, target, num_features)
    elif (backward):
        select_features = backward_search(select_data, target, num_features)
    else:
        select_features = core_predictors
        
    print('>>> Core features selected:')
    select_features.remove(target)
    print(select_features)
    
    select_features.extend(extra_predictors)
    select_features.append(target)
    final_data = data.loc[:, select_features] 
    
    final_data.to_csv(str(project_dir / "processed/select_features.csv"))
    logger.info('Features have been selected.')
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2] / "data"

    main()
    