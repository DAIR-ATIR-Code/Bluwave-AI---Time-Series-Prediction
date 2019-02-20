# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from pandas import Series, read_csv
import tensorflow as tf
from numpy import sqrt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def load(logger, target):
    try:
        data = read_csv(str(project_dir / "data/processed/select_features.csv"),
                        parse_dates=True, infer_datetime_format=True,
                        index_col=0)
        logger.info('Select features data set was loaded.')
    except Exception:
        logger.error('data/processed/select_features.csv could not be read.')
        raise ValueError('DataFrame is empty.')
          
    try:
        clean_target = read_csv(str(project_dir / "data/interim/clean.csv"),
                                parse_dates=True, infer_datetime_format=True,
                                usecols=['Date/Time', target],
                                index_col=0)
        logger.info('Pre-normalized data set was loaded.')
    except Exception:
        logger.error('data/interim/clean.csv could not be read.')
        raise ValueError('DataFrame is empty.')
        
    try:
        sess = tf.Session()
        path = str(project_dir / "models/trained_model")
        saver = tf.train.import_meta_graph(path + ".meta")
        saver.restore(sess, path)
        logger.info('Trained model was loaded.')
    except Exception:
        logger.error('models/trained_model.meta could not be loaded.')
        raise ValueError('Model is unavailable.')
        
    return data, clean_target, sess


def unnormalize(x, clean_data):
    minimum, maximum = float(clean_data.min()), float(clean_data.max())
    return x * (maximum - minimum) + minimum


def rmse(true, predicted):
    mse = ((true - predicted)**2).mean()
    return sqrt(mse)


def main():
    """ Makes predictions using trained model in (../models)
        and saves output to (../models).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making predictions with trained model.')
    
    target = 'Wind Spd (km/h)'
    test_start = '2018-01'

    data, clean_target, sess = load(logger, target)
    
    test = data.loc[test_start:]
    persistence = test.loc[:, [target, target + ' [t-1hr]']].copy()
    test.pop(target)

    X = tf.get_collection('X')[0]
    y_predict = tf.get_collection('y_predict')[0]

    # Make predictions with model on test data
    predict_y = sess.run(y_predict, feed_dict={X: test})[:, 0]
    predictions = Series(predict_y, index=test.index, name=target)
    sess.close()
    
    # Restore original scale of values
    norm_predictions = unnormalize(predictions, clean_target)
    norm_target = unnormalize(persistence, clean_target)
    
    predict_rmse = rmse(norm_target.loc[:, target], norm_predictions)
    persist_rmse = rmse(norm_target.loc[:, target],
                        norm_target.loc[:, target + ' [t-1hr]'])
    
    print('>>> Prediction RMSE: \t{:.4f}'.format(predict_rmse))
    print('>>> Persistence RMSE: \t{:.4f}'.format(persist_rmse))
    
    norm_predictions.to_csv(str(project_dir / "models/predictions.csv"),
                            header=False)
    logger.info('Model predictions saved.')
   

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]
    tf.logging.set_verbosity(logging.WARN)

    main()