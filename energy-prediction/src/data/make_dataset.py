# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from pandas import DataFrame, read_csv
import glob


def load(logger):
    # Integrate all data into one set
    paths = glob.glob(str(project_dir / "raw/*.csv"))
    paths.sort()
    if (len(paths) == 0):
        logger.error('Raw data was not found.')
        raise FileNotFoundError('Raw data was not found.')
    raw = DataFrame()
    for p in paths:
        try:
            year = read_csv(p, parse_dates=[0], thousands=',')
            raw = raw.append(year, sort=False)
        except Exception:
            logger.error(p + ' could not be read.')
            raise ValueError('DataFrame is empty.')
    logger.info('Raw data set was loaded.')
    return raw


def main():
    """ Integrates, cleans, and regularizes data from (../raw) into data
        ready for feature engineering, saved in (../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('Turning raw data into useable format.')

    raw = load(logger)
    raw['Hr_Start'] = raw['Hr_End'] - 1
    dt = raw.apply(lambda x : x['Date'].replace(hour=x['Hr_Start']), axis=1)
    raw.set_index(dt, inplace=True)
    raw.index.rename('Datetime', inplace=True)
    raw.to_csv(str(project_dir / "interim/integrated.csv"))
    
    # Drop RCP and RSP columns due to sparseness
    clean = raw.drop(columns=['Date', 'Hr_End', 'Hr_Start', 'Max_5min_RCP', 
                              'Max_5min_RSP', 'Min_5min_RCP', 'Min_5min_RSP'])
    
    if (clean.empty):
        logger.error('No data remains after cleaning.')
        raise ValueError('Clean DataFrame is empty.')
    else:
        logger.info('Raw data set was cleaned.')
        
    clean.to_csv(str(project_dir / "interim/clean.csv"))
    logger.info('Cleaned data set was saved.')
    
    # Scale the measurements
    reg = clean.copy()
    minimum, maximum = reg.min(), reg.max()
    reg = (reg - minimum) / (maximum - minimum)
    
    reg.to_csv(str(project_dir / "interim/regularized.csv"))
    logger.info('Regularized data set was saved.')
   

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2] / "data"

    main()