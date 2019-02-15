# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from pandas import DataFrame, read_csv, concat
from tensorflow.keras.utils import to_categorical


def load(logger):
    try:
        data = read_csv(str(project_dir / "interim/regularized.csv"),
                        parse_dates=True, infer_datetime_format=True,
                        index_col=0)
        logger.info('Pre-processed data set was loaded.')
    except Exception:
        logger.error('data/interim/regularized.csv could not be read.')
        raise ValueError('DataFrame is empty.')
    return data


def main():
    """ Builds features on top of clean, regularized data from (../interim)
        and saves the data and features together in (../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating features from clean data.')

    target = 'Wind Spd (km/h)'
    data = load(logger)
    
    # Drop columns with extremely high correlation
    #data.pop('Dew Point Temp (°C)')
    # Drop columns with extremely low variance
    data.pop('Visibility (km)')
    
    # Identify entries that include an outlier
    stats = data.describe()
    stats.loc['IQR'] = (stats.loc['75%'] - stats.loc['25%'])
    stats.loc['low'] = stats.loc['25%'] - (1.5 * stats.loc['IQR'])
    stats.loc['high'] = stats.loc['75%'] + (1.5 * stats.loc['IQR'])

    outliers = data.apply(lambda x: x[(x < stats.loc['low', x.name]) |
                            (x > stats.loc['high', x.name])], axis=0)

    data.loc[:, 'Outlier'] = 0
    data.loc[outliers.index, 'Outlier'] = 1
    
    # Create features out of lagged wind speed measurements
    data[target + ' [t-1hr]'] = data[target].shift(1)
    data[target + ' [t-2hr]'] = data[target].shift(2)
    data[target + ' [t-12hr]'] = data[target].shift(12)
    data[target + ' [t-24hr]'] = data[target].shift(24)
    data[target + ' [t-36hr]'] = data[target].shift(36)
    data[target + ' [t-48hr]'] = data[target].shift(48)
    data.dropna(inplace=True)
        
    # Create binary indicators for hour and month
    hour_names = ['H' + str(x+1) for x in range(12)]
    one_hot_hour = to_categorical(data.index.hour % 12)
    hour_df = DataFrame(one_hot_hour, index=data.index, columns=hour_names)
    
    month_names = ['M' + str(x+1) for x in range(12)]
    one_hot_month = to_categorical(data.index.month - 1)
    month_df = DataFrame(one_hot_month, index=data.index, columns=month_names)
    
    features = concat([data, hour_df, month_df], axis=1)
    
    features.to_csv(str(project_dir / "processed/all_features.csv"))
    logger.info('Features were created.')
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2] / "data"

    main()
    