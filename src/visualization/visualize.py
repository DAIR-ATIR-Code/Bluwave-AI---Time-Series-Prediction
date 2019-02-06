# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from pandas import read_csv
import matplotlib.pyplot as plt


def load(logger):
    try:
        clean = read_csv(str(project_dir / "data/interim/clean.csv"),
                         parse_dates=True, infer_datetime_format=True,
                         index_col=0, low_memory=False)
        logger.info('Clean data set was loaded.')
    except Exception:
        logger.error('\\data\\interim\\integrated.csv could not be read.')
        raise ValueError('DataFrame is empty.')

    try:
        predict = read_csv(str(project_dir / "models/predictions.csv"),
                           parse_dates=True, infer_datetime_format=True,
                           index_col=0, header=None, names=['Wind Spd (km/h)'])
        logger.info('Predictions data set was loaded.')
    except Exception:
        logger.error('models/predictions.csv could not be read.')
        raise ValueError('DataFrame is empty.')
    
    return clean, predict


def main():
    """ Visualizes the performance of the model.
    """
#%%
    logger = logging.getLogger(__name__)
    logger.info('Visualizing performance of trained model.')

    clean, predict = load(logger)
    
    test_start = '2018-07'
    test = clean.loc[test_start:]
    test_y = test.pop('Wind Spd (km/h)')
    predict_y = predict['Wind Spd (km/h)']
        
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(test_y)
    ax.plot(predict_y)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(test_y[:45], label='actual')
    ax.plot(predict_y[:45], label='prediction')
    plt.legend()
    plt.show()
   
#%%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()