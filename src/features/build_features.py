# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame, read_csv, concat
from tensorflow.keras.utils import to_categorical


def load(logger):
    try:
        data = read_csv(project_dir + '\\data\\interim\\regularized.csv', \
                        parse_dates=True, infer_datetime_format=True, \
                        index_col=0)
        logger.info('Pre-processed data set was loaded.')
    except Exception:
        logger.error('\\data\\interim\\regularized.csv could not be read.')
        raise ValueError('DataFrame is empty.')
    return data


def main():
    """ Builds features on top of clean, regularized data from (../interim)
        and saves the data and features together in (../processed).
    """
#%%
    logger = logging.getLogger(__name__)
    logger.info('Creating features from clean data.')

    data = load(logger)
    
    # Identify entries that include an outlier
    stats = data.describe()
    stats.loc['IQR'] = (stats.loc['75%'] - stats.loc['25%'])
    stats.loc['low'] = stats.loc['25%'] - (1.5 * stats.loc['IQR'])
    stats.loc['high'] = stats.loc['75%'] + (1.5 * stats.loc['IQR'])

    outliers = data.iloc[:,2:].apply(lambda x: x[(x<stats.loc['low',x.name]) | (x>stats.loc['high',x.name])], axis=0)

    data.loc[:,'Outlier'] = 0
    data.loc[outliers.index,'Outlier'] = 1
    
    # Create features out of lagged wind speed measurements
    data['Wind Spd (km/h) [t-1hr]'] = data['Wind Spd (km/h)'].shift(1)
    data['Wind Spd (km/h) [t-2hr]'] = data['Wind Spd (km/h)'].shift(2)
    data['Wind Spd (km/h) [t-12hr]'] = data['Wind Spd (km/h)'].shift(12)
    data['Wind Spd (km/h) [t-24hr]'] = data['Wind Spd (km/h)'].shift(24)
    data['Wind Spd (km/h) [t-36hr]'] = data['Wind Spd (km/h)'].shift(36)
    data['Wind Spd (km/h) [t-48hr]'] = data['Wind Spd (km/h)'].shift(48)
    data.dropna(inplace=True)
        
    # Create binary indicators for hour and month
    hour_names = ['H' + str(x+1) for x in range(12)]
    one_hot_hour = to_categorical(data['Time'].apply(lambda x : int(str(x)[:-3]) % 12))
    hour_df = DataFrame(one_hot_hour, index=data.index, columns=hour_names)
    
    month_names = ['M' + str(x+1) for x in range(12)]
    one_hot_month = to_categorical(data['Month']-1)
    month_df = DataFrame(one_hot_month, index=data.index, columns=month_names)
    
    features = concat([data.iloc[:,2:], hour_df, month_df], axis=1)
    #features = data.iloc[:,2:]
    
    features.to_csv(project_dir + '\\data\\processed\\all_features.csv')
    logger.info('Features were created.')
    
#%%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = str(Path(__file__).resolve().parents[2])

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
    