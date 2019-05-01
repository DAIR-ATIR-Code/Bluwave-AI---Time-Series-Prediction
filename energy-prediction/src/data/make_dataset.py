# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from pandas import DataFrame, read_csv, read_excel
import requests
import glob


def load(logger):
    # Integrate all data into one set, sorting by timestamp
    paths = glob.glob(str(project_dir / "raw/*.csv"))
    paths.sort()
    if (len(paths) == 0):
        # Download data from ISO NE website
        names = {'2016': 'https://www.iso-ne.com/static-assets/documents/2016/02/smd_hourly.xls',
                 '2017': 'https://www.iso-ne.com/static-assets/documents/2017/02/2017_smd_hourly.xlsx',
                 '2018': 'https://www.iso-ne.com/static-assets/documents/2018/02/2018_smd_hourly.xlsx'}
        try:
            for year in names.keys():
                data = requests.get(names[year], allow_redirects=True)
                rawname = "raw/" + year + "_smd_hourly"
                with open(str(project_dir / (rawname + ".xls")), 'wb') as f:
                    f.write(data.content)
                excel = read_excel(str(project_dir / (rawname + ".xls")), 1)
                excel.to_csv(str(project_dir / (rawname + ".csv")), index=False)
        except Exception:
            logger.error('Raw data was not found.')
            raise FileNotFoundError('Raw data was not found.')
        paths = glob.glob(str(project_dir / "raw/*.csv"))
        paths.sort()
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
        Accompanying notebook is 01-clean.
    """
    logger = logging.getLogger(__name__)
    logger.info('Turning raw data into usable format.')

    raw = load(logger)
    # Configure Datetime indexing
    raw['Hr_Start'] = raw['Hr_End'] - 1
    dt = raw.apply(lambda x: x['Date'].replace(hour=x['Hr_Start']), axis=1)
    raw.set_index(dt, inplace=True)
    raw.index.rename('Datetime', inplace=True)
    raw.to_csv(str(project_dir / "interim/integrated.csv"))
    
    # Drop (a) redundant date and time columns and
    #      (b) sparse RSP and RCP columns
    clean = raw.drop(columns=['Date', 'Hr_End', 'Hr_Start', 'Max_5min_RCP',
                              'Max_5min_RSP', 'Min_5min_RCP', 'Min_5min_RSP'])
    
    if (clean.empty):
        logger.error('No data remains after cleaning.')
        raise ValueError('Clean DataFrame is empty.')
    else:
        logger.info('Raw data set was cleaned.')
        
    clean.to_csv(str(project_dir / "interim/clean.csv"))
    logger.info('Cleaned data set was saved.')
    
    # Scale the measurements so that all values have similar magnitudes.
    # This helps optimization (gradient descent) work better.
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