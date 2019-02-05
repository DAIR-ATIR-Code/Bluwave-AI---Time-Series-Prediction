# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from pandas import DataFrame, read_csv
import glob


def load(logger):
    # Integrate all data into one set
    folders = glob.glob(str(project_dir / "raw/*/"))
    if (len(folders) == 0):
        logger.error('Raw data was not found.')
        raise FileNotFoundError('Raw data was not found.')
    raw = DataFrame()
    for f in folders:
        paths = glob.glob(str(Path(f) / "*.csv"))
        for p in paths:
            try:
                month = read_csv(p, skiprows=15, parse_dates=True, \
                                 infer_datetime_format=True, index_col=0)
                raw = raw.append(month)
            except Exception:
                logger.error(p + ' could not be read.')
                raise ValueError('DataFrame is empty.')
    logger.info('Raw data set was loaded.')
    return raw


def main():
    """ Integrates, cleans, and regularizes data from (../raw) into data
        ready for feature engineering, saved in (../interim).
    """
#%%
    logger = logging.getLogger(__name__)
    logger.info('Turning raw data into usable format.')

    raw = load(logger) 
    raw.to_csv(str(project_dir / "interim/integrated.csv"))
    
    # Drop unnecessary date and flag columns
    # Drop Hmdx, Wind Chill, and Weather columns due to sparseness
    select = raw.iloc[:,[1,3,4,6,8,10,12,14,16]]

    # Interpolate missing data
    clean = select.interpolate(method='linear', limit_direction='both')
    
    if (clean.empty):
        logger.error('No data remains after cleaning.')
        raise ValueError('Clean DataFrame is empty.')
    else:
        logger.info('Raw data set was cleaned.')
        
    clean.to_csv(str(project_dir / "interim/clean.csv"))
    logger.info('Cleaned data set was saved.')
    
    # Scale the measurements
    reg = clean.copy()
    minimum, maximum = reg.iloc[:,2:].min(), reg.iloc[:,2:].max()
    reg.iloc[:,2:] = (reg.iloc[:,2:] - minimum) / (maximum - minimum)
    
    reg.to_csv(str(project_dir / "interim/regularized.csv"))
    logger.info('Regularized data set was saved.')
   
#%%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2] / "data"

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()