# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from pandas import DataFrame, read_csv
import sklearn.feature_selection as fs
from scipy.stats import pearsonr


def do_feat_ranking(data, target):
    y = data.pop(target)
    X = data.copy()
    feat_scores = DataFrame(columns=X.columns)

    for col in X.columns:
        feat_scores.loc['pearson', col] = pearsonr(X.loc[:,col], y)[0]
    feat_scores.loc['f-score'] = fs.f_regression(X, y)[0]
    feat_scores.loc['mic'] = fs.mutual_info_regression(X, y)

    feat_scores_reg = feat_scores.apply(lambda x : x / x.max(), axis=1)
    feat_scores_avg = feat_scores_reg.mean(axis=0)
    feat_ranking = feat_scores_avg.sort_values(ascending=False)
    return feat_ranking


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
        Feature selection method is averaged univariate feature ranking.
    """
    logger = logging.getLogger(__name__)
    logger.info('Selecting features from clean data.')

    target = 'System_Load'
    data = load(logger)
    train_end = '2018-04'

    core_predictors = list(data.columns[:16])
    extra_predictors = list(data.columns[16:])
    
    train_data = data.loc[:train_end].copy()
    feat_ranking = do_feat_ranking(train_data[core_predictors], target)
    
    select_features = list(feat_ranking[:10].index)
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
    