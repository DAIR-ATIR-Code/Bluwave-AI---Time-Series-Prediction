# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from pandas import DataFrame, read_csv
import sklearn.feature_selection as fs
from scipy.stats import pearsonr


def do_feat_ranking(data, target):
    X = data.drop(columns=[target])
    y = data.loc[:, target]
    feat_scores = DataFrame(columns=X.columns)

    # Calculate various measurements of relationship between each feature and
    # the target. Choose (a) Pearson correlation coefficient, (b) f-score, and
    # (c) mutual information for their popularity.
    for col in X.columns:
        feat_scores.loc['pearson', col] = pearsonr(X.loc[:, col], y)[0]
    feat_scores.loc['f-score'] = fs.f_regression(X, y)[0]
    feat_scores.loc['m-info'] = fs.mutual_info_regression(X, y)

    # Rank features based on mean of regularized feature scoring metrics
    feat_scores_reg = feat_scores.apply(lambda x: x / x.max(), axis=1)
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
        and selected features in (../processed) ready for model training.
        Feature selection method is ranking by mean univariate feature scores.
    """
    logger = logging.getLogger(__name__)
    logger.info('Selecting features from data.')

    target = 'System_Load'
    val_end = '2018-06'
    num_features = 20

    data = load(logger)
    # Only run feature selection algorithm on training and validation data
    feat_ranking = do_feat_ranking(data.loc[:val_end], target)
    select_features = list(feat_ranking[:num_features].index)
    print('>>> Features selected:')
    print(select_features)
    
    select_features.append(target)
    final_data = data.loc[:, select_features]
    
    final_data.to_csv(str(project_dir / "processed/select_features.csv"))
    logger.info('Features have been selected.')
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2] / "data"

    main()
    