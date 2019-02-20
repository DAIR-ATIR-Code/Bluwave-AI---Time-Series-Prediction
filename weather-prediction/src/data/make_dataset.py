# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from pandas import DataFrame, read_csv
import glob


def load(logger):
    # Integrate all data into one set
    folders = glob.glob(str(project_dir / "raw/*/"))
    folders.sort()
    if (len(folders) == 0):
        logger.error('Raw data was not found.')
        raise FileNotFoundError('Raw data was not found.')
    raw = DataFrame()
    for f in folders:
        paths = glob.glob(str(Path(f) / "*.csv"))
        paths.sort()
        for p in paths:
            try:
                month = read_csv(p, skiprows=15, parse_dates=True,
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
    logger = logging.getLogger(__name__)
    logger.info('Turning raw data into usable format.')

    raw = load(logger) 
    raw.to_csv(str(project_dir / "interim/integrated.csv"))
    
    # Drop (a) redundant date and time columns, (b) unnecessary flag columns,
    # (c) sparse Hmdx, Wind Chill, and Weather columns
    select = raw.iloc[:, [4, 6, 8, 10, 12, 14, 16]]

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
    minimum, maximum = reg.min(), reg.max()
    reg = (reg - minimum) / (maximum - minimum)
    
    reg.to_csv(str(project_dir / "interim/regularized.csv"))
    logger.info('Regularized data set was saved.')
   

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2] / "data"

    main()