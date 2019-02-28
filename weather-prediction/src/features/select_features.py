# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from pandas import Series, read_csv
import tensorflow as tf
import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / "src/models/"))
from train_model import build_and_train_model


# Backward recursive search incrementally removes the feature that,
# when removed, achieves the best model performance. 
def backward_search(data, target, num_features):
    features = list(data.columns)
    features.remove(target)
    for i in range(0, len(features) - num_features):
        print(features)
        feat_scores = Series(index=features)
        # Experiment with the removal of each feature
        for feat in features:
            features_sub = features.copy()
            features_sub.remove(feat)
            metric = run_model_on_feature_subset(data, target, features_sub)
            feat_scores.loc[feat] = metric
        # Remove feature whose removal achieved the best performance
        feat_round_ranking = feat_scores.sort_values()
        optimal_feature = feat_round_ranking.index[0]
        features.remove(optimal_feature)
    return features


# Forward recursive search incrementally adds the feature that,
# when added, achieves the best model performance. 
def forward_search(data, target, num_features):
    features = list(data.columns)
    features.remove(target)
    features_sub = []
    for i in range(0, num_features):
        if (i > 0): print(features_sub)
        feat_scores = Series(index=features)
        # Experiment with the addition of each feature
        for feat in features:
            features_sub.append(feat)
            metric = run_model_on_feature_subset(data, target, features_sub)
            feat_scores.loc[feat] = metric
            features_sub.remove(feat)
        # Add feature whose addition achieved the best performance
        feat_round_ranking = feat_scores.sort_values()
        optimal_feature = feat_round_ranking.index[0]
        features_sub.append(optimal_feature)
        features.remove(optimal_feature)
    return features_sub


def run_model_on_feature_subset(data, target, features_sub):
    train_end = '2016-12'
    val_start = '2017-01'
    val_end = '2017-12'
    
    # Add target so that model can run properly
    columns_sub = features_sub.copy()
    columns_sub.append(target)
    train = data.loc[:train_end, columns_sub]
    validate = data.loc[val_start:val_end, columns_sub]
    # Choose reasonable, but not necessarily optimal, hyperparameters
    params = {'num_hidden': 50,
              'learn_rate': 0.001,
              'lambda': 0,
              'dropout': 0.2,
              'num_epochs': 5000,
              'activation': tf.nn.relu}
    sess, saver, history, metric, sec = build_and_train_model(train, validate,
                                                              params, target)
    sess.close()
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
    # We select 5 to demonstrate the algorithm, but in practice we
    # would probably keep all 6 "core predictors"
    num_features = 5
    forward = False
    backward = True

    data = load(logger)
    core_predictors = list(data.columns[:6])
    extra_predictors = list(data.columns[6:])
    
    # Only run feature selection algorithm on core predictors, since
    # including all predictors would be too expensive
    select_data = data.loc[:, core_predictors]
    
    if (forward):
        print('>>> Forward recursive search:')
        select_features = forward_search(select_data, target, num_features)
    elif (backward):
        print('>>> Backward recursive search:')
        select_features = backward_search(select_data, target, num_features)
    else:
        select_features = core_predictors
        
    print('>>> Core features selected:')
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
    