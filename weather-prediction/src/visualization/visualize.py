# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from pandas import read_csv
import matplotlib.pyplot as plt


def load(logger, target):
    try:
        clean = read_csv(str(project_dir / "data/interim/clean.csv"),
                         parse_dates=True, infer_datetime_format=True,
                         index_col=0, low_memory=False)
        logger.info('Clean data set was loaded.')
    except Exception:
        logger.error('data/interim/clean.csv could not be read.')
        raise ValueError('DataFrame is empty.')

    try:
        predict = read_csv(str(project_dir / "models/predictions.csv"),
                           parse_dates=True, infer_datetime_format=True,
                           index_col=0, header=None, names=[target])
        logger.info('Predictions data set was loaded.')
    except Exception:
        logger.error('models/predictions.csv could not be read.')
        raise ValueError('DataFrame is empty.')
    
    return clean, predict


def main():
    """ Visualizes the performance of the model.
    """
    logger = logging.getLogger(__name__)
    logger.info('Visualizing performance of trained model.')

    target = 'Wind Spd (km/h)'
    test_start = '2018-01'

    clean, predict = load(logger, target)

    test = clean.loc[test_start:]
    test_y = test.pop(target)
    predict_y = predict[target]
        
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(test_y, label='actual')
    ax.plot(predict_y, label='prediction')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Wind Speed (km/h)')
    plt.title('Wind Speed Time Series (Macro)')
    plt.legend()
    plt.savefig(str(project_dir / "reports/figures/wind_macro_time_series.png"))
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(test_y[:90], label='actual')
    ax.plot(predict_y[:90], label='prediction')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Wind Speed (km/h)')
    plt.title('Wind Speed Time Series (Micro)')
    plt.legend()
    plt.savefig(str(project_dir / "reports/figures/wind_micro_time_series.png"))
    plt.show()
   

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    main()