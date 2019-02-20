# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from pandas import DataFrame, read_csv, concat
from numpy import arange, newaxis
import tensorflow as tf
from tensorflow.python.client import device_lib
from itertools import product
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def create_model(data, params):
    num_inputs = len(data.columns) - 1
    num_hidden = params['num_hidden']
    num_outputs = 1
    
    X = tf.placeholder('float', shape=[None, num_inputs])
    y = tf.placeholder('float', shape=[None, num_outputs])
    
    # Initialize weights and biases
    w1 = tf.Variable(tf.random_normal([num_inputs, num_hidden], stddev=0.1))
    w2 = tf.Variable(tf.random_normal([num_hidden, num_outputs], stddev=0.1))
    b1 = tf.Variable(tf.zeros([num_hidden]))
    b2 = tf.Variable(tf.zeros([num_outputs]))
    
    # Forward propagation
    hidden_layer = params['activation'](tf.matmul(X, w1) + b1)
    dropout_layer = tf.nn.dropout(hidden_layer, 1 - params['dropout'])
    y_predict = tf.matmul(dropout_layer, w2) + b2
    
    # Backward propagation
    # Use MSE as the loss (objective) function
    loss = tf.reduce_mean(tf.square(y - y_predict))
    reg = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    reg_loss = tf.reduce_mean(loss + params['lambda'] * reg)
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learn_rate'])
    updates = optimizer.minimize(reg_loss)
    return X, y, y_predict, reg_loss, updates


def create_and_fit_model(train, validate, params, target):
    train_X = train.drop(columns=[target])
    train_y = train.loc[:, target][:, newaxis]
    validate_X = validate.drop(columns=[target])
    validate_y = validate.loc[:, target][:, newaxis]
    
    tf.reset_default_graph()
    X, y, y_predict, loss, updates = create_model(train, params)
    hist = DataFrame(index=arange(params['num_epochs']),
                     columns=['train_loss', 'validate_loss'])
    
    start = time.time()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for epoch in range(params['num_epochs']):
        sess.run(updates, feed_dict={X: train_X, y: train_y})

        hist.loc[epoch, 'train_loss'] = \
            sess.run(loss, feed_dict={X: train_X, y: train_y})
        hist.loc[epoch, 'validate_loss'] = \
            sess.run(loss, feed_dict={X: validate_X, y: validate_y})
    seconds = time.time() - start
    metric = hist.loc[epoch, 'validate_loss']
    
    tf.add_to_collection('X', X)
    tf.add_to_collection('y_predict', y_predict)
    saver = tf.train.Saver(max_to_keep=0)
    return sess, saver, hist, metric, seconds
            

def load(logger):
    try:
        data = read_csv(str(project_dir / "data/processed/select_features.csv"),
                        parse_dates=True, infer_datetime_format=True,
                        index_col=0)
        logger.info('Select features set was loaded.')
    except Exception:
        logger.error('data/processed/select_features.csv could not be read.')
        raise ValueError('DataFrame is empty.')
    return data


def main():
    """ Tunes an artificial neural network with a grid search of
        hyperparameters, saving optimal model in (../models)
    """
    logger = logging.getLogger(__name__)
    logger.info('Tuning artificial neural network.')

    target = 'Wind Spd (km/h)'
    train_end = '2016-12'
    val_start = '2017-01'
    val_end = '2017-12'
    
    data = load(logger)
    train = data.loc[:train_end]
    validate = data.loc[val_start:val_end]
    
    all_params = {'num_hidden': [50, 100, 250],
                  'learn_rate': [0.001],
                  'lambda': [0, 0.01],
                  'dropout': [0.2],
                  'num_epochs': [7500],
                  'activation': [tf.nn.relu]}
    
    devices = [x.device_type for x in device_lib.list_local_devices()]
    num_gpus = devices.count('GPU')
    print('{} GPU(s) available to TensorFlow.'.format(num_gpus))

    # Basic grid search of hyperparameters
    keys, values = zip(*all_params.items())
    param_scores = DataFrame(index=product(*values))
    print('>>> Model hyperparameters grid search:')
    count = 0
    min_model_metric = 1
    print('\r{}/{} configurations'.format(count, len(param_scores)), end='')
    for variation in product(*values):
        params = dict(zip(keys, variation))
        sess, saver, history, metric, sec = create_and_fit_model(train,
                                                                 validate,
                                                                 params,
                                                                 target)
        param_scores.loc[variation, 'loss'] = metric
        param_scores.loc[variation, 'seconds'] = int(sec)
        if (metric < min_model_metric):
            saver.save(sess, str(project_dir / "models/trained_model"))
            optimal_history = history
            min_model_metric = metric
        sess.close()
        count = count + 1
        print('\r{}/{} configurations'.format(count, len(param_scores)),
              end='')
    print('')
    params_ranking = param_scores.sort_values(by='loss')
    print('>>> Ranking of hyperparameter combinations:')
    print(params_ranking)
    
    logger.info('Hyperparameters tuned.')
    optimal_history.to_csv(str(project_dir / "models/training_history.csv"))
    logger.info('Trained model saved.')
   

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)
    
    project_dir = Path(__file__).resolve().parents[2]
    tf.logging.set_verbosity(logging.WARN)

    main()