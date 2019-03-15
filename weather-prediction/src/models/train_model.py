# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from pandas import DataFrame, read_csv
from numpy import arange, newaxis
import tensorflow as tf
from tensorflow.python.client import device_lib
from itertools import product
import pickle
import time
import sys
import os

# Reduce TensorFlow warnings about optimal CPU setup/usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


# Build feedforward neural network with one hidden layer
# Various hyperparameters are given by the values in dictionary 'params'
def build_model(data, params):
    # Input size is the number of features
    num_inputs = len(data.columns) - 1
    # Hidden layer should have number of nodes corresponding to the
    # complexity of the system we want to model
    num_hidden = params['num_hidden']
    # Output layer has one node because prediction is a single value
    num_outputs = 1
    
    # Create placeholders, to which values will be assigned during training
    X = tf.placeholder('float', shape=[None, num_inputs])
    y = tf.placeholder('float', shape=[None, num_outputs])
    
    # Initialize weights as non-zero random numbers
    w1 = tf.Variable(tf.random_normal([num_inputs, num_hidden], stddev=0.1))
    w2 = tf.Variable(tf.random_normal([num_hidden, num_outputs], stddev=0.1))
    # Initialize biases as zero
    b1 = tf.Variable(tf.zeros([num_hidden]))
    b2 = tf.Variable(tf.zeros([num_outputs]))
    
    # Forward propagation
    # Activation function determines the value of the nodes in the layer
    # Each layer is computed as: activation((<inputs> * w) + b)
    hidden_layer = params['activation'](tf.matmul(X, w1) + b1)
    # Dropout is not a true layer, but randomly removes some % of hidden
    # nodes to help the model generalize
    dropout_layer = tf.nn.dropout(hidden_layer, 1 - params['dropout'])
    # Output layer has no activation function because we are building a
    # regression model
    output_layer = tf.matmul(dropout_layer, w2) + b2
    
    # Backward propagation
    # Choose loss function to be MSE
    loss = tf.reduce_mean(tf.square(y - output_layer))
    # Perform L2 regularization, which punishes the loss function to help
    # the model generalize. Lambda determines the degree of regularization
    reg = tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
    reg_loss = tf.reduce_mean(loss + params['lambda'] * reg)
    # Learning rate decides how quickly optimization (gradient descent) happens
    optimizer = tf.train.AdamOptimizer(learning_rate=params['learn_rate'])
    # Setup model to optimize regularized loss function with Adam optimizer
    updates = optimizer.minimize(reg_loss)
    return X, y, output_layer, reg_loss, updates


# Train neural network created in build_model()
def build_and_train_model(train, validate, params, target):
    # Separate the target (y) from the input (X)
    # Training data used to minimize the loss function and validation data
    # used to calculate loss in each epoch. The model is somewhat biased toward
    # the training data it is learning from, but the validation loss gives
    # a fair sense of how well the model is learning and also generalizing.
    train_X = train.drop(columns=[target])
    train_y = train.loc[:, target][:, newaxis]
    validate_X = validate.drop(columns=[target])
    validate_y = validate.loc[:, target][:, newaxis]
    
    tf.reset_default_graph()
    X, y, y_predict, loss, updates = build_model(train, params)
    hist = DataFrame(index=arange(params['num_epochs']),
                     columns=['train_loss', 'validate_loss'])
    
    start = time.time()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # Model runs through all the training data in each epoch, doing forward
    # and backward propagation of weights. No batching because our dataset is
    # small enough and optimization still performs well, so we achieve more
    # efficiency this way.
    for epoch in range(params['num_epochs']):
        # Run on training data to update model weights
        _, train_loss = sess.run([updates, loss], feed_dict={X: train_X,
                                                             y: train_y})
        # Run on validation data to evaluate performance
        validate_loss = sess.run(loss, feed_dict={X: validate_X,
                                                  y: validate_y})
        hist.loc[epoch, 'train_loss'] = train_loss
        hist.loc[epoch, 'validate_loss'] = validate_loss
    seconds = time.time() - start
    metric = hist.loc[epoch, 'validate_loss']
    
    tf.add_to_collection('X', X)
    tf.add_to_collection('y_predict', y_predict)
    saver = tf.train.Saver(max_to_keep=0)
    return sess, saver, hist, metric, seconds
            

def load(logger, which_features):
    filename = which_features + '_features.csv'
    try:
        data = read_csv(str(project_dir / "data/processed" / filename),
                        parse_dates=True, infer_datetime_format=True,
                        index_col=0)
        logger.info(which_features.capitalize() + ' features set was loaded.')
    except Exception:
        logger.error('data/processed/' + filename + ' could not be read.')
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
    
    if (len(sys.argv) < 2):
        logger.warning('No feature set specified for training the model.')
        logger.info('Model defaulted to training on all features.')
        which_features = 'all'
    elif not (sys.argv[1] == 'all' or sys.argv[1] == 'select'):
        logger.warning('Invalid feature set specified for training the model.')
        logger.info('Model defaulted to training on all features.')
        which_features = 'all'
    else:
        which_features = sys.argv[1]
    
    data = load(logger, which_features)
    train = data.loc[:train_end]
    validate = data.loc[val_start:val_end]
    
    # Determine all the hyperparameter options to optimize on
    all_params = {'num_hidden': [5, 50],
                  'learn_rate': [0.001, 0.01],
                  'lambda': [0],
                  'dropout': [0],
                  'num_epochs': [10000],
                  'activation': [tf.nn.relu]}
    
    devices = [x.device_type for x in device_lib.list_local_devices()]
    num_gpus = devices.count('GPU')
    print('{} GPU(s) available to TensorFlow.'.format(num_gpus))
    print('Model training on ' + which_features + ' features.')

    # Basic grid search of hyperparameters
    keys, values = zip(*all_params.items())
    param_scores = DataFrame(index=product(*values))
    print('>>> Model hyperparameters grid search:')
    count = 0
    optimal_metric = 2
    print('\r{}/{} configurations'.format(count, len(param_scores)), end='')
    # Permute through the hyperparameters
    for variation in product(*values):
        # Run the model with this set of hyperparameters
        params = dict(zip(keys, variation))
        sess, saver, history, metric, sec = build_and_train_model(train,
                                                                  validate,
                                                                  params,
                                                                  target)
        param_scores.loc[variation, 'loss'] = metric
        param_scores.loc[variation, 'seconds'] = int(sec)
        # Save this model output if it has achieved minimal error thus far
        if (metric < optimal_metric):
            model_filename = 'trained_model_' + which_features + '_features'
            saver.save(sess, str(project_dir / "models" / model_filename))
            optimal_history = history
            optimal_params = params
            optimal_metric = metric
        sess.close()
        count = count + 1
        print('\r{}/{} configurations'.format(count, len(param_scores)),
              end='')
    print('')
    params_ranking = param_scores.sort_values(by='loss')
    print('>>> Ranking of hyperparameter combinations:')
    print(params_ranking)
    
    logger.info('Hyperparameters tuned.')
    history_filename = 'training_history_' + which_features + '_features.csv'
    optimal_history.to_csv(str(project_dir / "models" / history_filename))
    params_filename = 'model_params_' + which_features + '_features.pkl'
    with open(str(project_dir / "models" / params_filename), 'wb') as f:
        pickle.dump(optimal_params, f)
    logger.info('Trained model saved.')
   

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)
    
    project_dir = Path(__file__).resolve().parents[2]
    # Reduce TensorFlow verbosity
    tf.logging.set_verbosity(logging.WARN)

    main()