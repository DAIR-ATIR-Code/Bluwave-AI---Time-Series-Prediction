# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from pandas import Series, read_csv
#import sklearn.feature_selection as fs
#from scipy.stats import pearsonr

import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / "src/models/"))
from train_model import create_and_fit_model


def backward_search(data, num_features):
    features = list(data.columns)
    features.remove('Wind Spd (km/h)')
    for i in range(0, len(features) - num_features):
        feat_scores = Series(index=features)
        # Experiment with the removal of each feature
        for feat in features:
            features_sub = features.copy()
            features_sub.remove(feat)
            # Add wind speed so that model can run properly
            features_sub.append('Wind Spd (km/h)')
            metric = run_model_on_feature_subset(data, features_sub)
            feat_scores.loc[feat] = metric
            #print(features_sub)
        feat_round_ranking = feat_scores.sort_values()
        optimal_feature = feat_round_ranking.index[0]
        features.remove(optimal_feature)
        #print(features)
    return features


def forward_search(data, num_features):
    features = list(data.columns)
    features.remove('Wind Spd (km/h)')
    # Keep wind speed so that model can run properly
    features_sub = ['Wind Spd (km/h)']
    for i in range(0, num_features):
        feat_scores = Series(index=features)
        # Experiment with the addition of each feature
        for feat in features:
            features_sub.append(feat)
            metric = run_model_on_feature_subset(data, features_sub)
            feat_scores.loc[feat] = metric
            #print(features_sub)
            features_sub.remove(feat)
        feat_round_ranking = feat_scores.sort_values()
        optimal_feature = feat_round_ranking.index[0]
        features_sub.append(optimal_feature)
        features.remove(optimal_feature)
        #print(features_sub)
    return features_sub


# def do_feat_ranking(data):
#     y = data.pop('Wind Spd (km/h)')
#     X = data.copy()
#     feat_scores = DataFrame(columns=X.columns)
    
#     for col in X.columns:
#         feat_scores.loc['pearson', col] = pearsonr(X.loc[:,col], y)[0]
#     feat_scores.loc['f-score'] = fs.f_regression(X, y)[0]
#     feat_scores.loc['mic'] = fs.mutual_info_regression(X, y)
   
#     feat_scores_reg = feat_scores.apply(lambda x : x / x.max(), axis=1)
#     feat_scores_avg = feat_scores_reg.mean(axis=0)
#     feat_ranking = feat_scores_avg.sort_values(ascending=False)
#     return feat_ranking


def run_model_on_feature_subset(data, feat_sub):
    train_end = '2017-12'
    val_start = '2018-01'
    val_end = '2018-06'
    
    train = data.loc[:train_end, feat_sub]
    validate = data.loc[val_start:val_end, feat_sub]
    params = {'num_hidden': 72,
              'activation': 'relu',
              'learn_rate': 0.001,
              'lambda': 0.01,
              'dropout': 0.2,
              'num_epochs': 5000}
    model, hist, metric, sec = create_and_fit_model(train, validate, params)
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
        and winnowed features in (../processed) ready for model training.
        Feature selection methods available are (a) forward and (b) backward
        recursive search.
    """
#%%
    logger = logging.getLogger(__name__)
    logger.info('Selecting features from clean data.')

    data = load(logger)
       
    forward = False
    backward = False
    #train_end = '2017-12'
    
    # Remove Visibility since it has very low variance (non informative)
    data.pop('Visibility (km)')
    # Remove Dew Point Temp since it is highly correlated with Temp
    data.pop('Dew Point Temp (°C)')
    
    #train_data = data.loc[:train_end].copy()
    #feat_ranking = do_feat_ranking(train_data)
    
    # Compare to scikit learn?
    #kbest = fs.SelectKBest(score_func=fs.mutual_info_regression, k=30).fit_transform(X, y)
    #feature_scores = kbest.scores_
    
    core_predictors = list(data.columns[:6])
    extra_predictors = list(data.columns[6:])
    
    # Only run feature selection algorithm on core data predictors
    select_data = data.loc[:, core_predictors].copy()
    
    if (forward):
        select_features = forward_search(select_data, 5)
    elif (backward):
        select_features = backward_search(select_data, 5)
    else:
        select_features = core_predictors
    print('>>> Core features selected:')
    select_features.remove('Wind Spd (km/h)')
    print(select_features)
    select_features.extend(extra_predictors)
    select_features.append('Wind Spd (km/h)')
    final_data = data.loc[:, select_features] 
    
    final_data.to_csv(str(project_dir / "processed/select_features.csv"))
    logger.info('Features have been selected.')
    
#%%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2] / "data"

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
    