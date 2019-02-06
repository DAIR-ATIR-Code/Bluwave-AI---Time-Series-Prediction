# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
#import click
from pandas import Series, read_csv
from tensorflow import keras


def load(logger):
    try:
        data = read_csv(str(project_dir / "data/processed/select_features.csv"),
                        parse_dates=True, infer_datetime_format=True,
                        index_col=0)
        logger.info('Select features data set was loaded.')
    except Exception:
        logger.error('data/processed/select_features.csv could not be read.')
        raise ValueError('DataFrame is empty.')
          
    try:
        clean_wspd = read_csv(str(project_dir / "data/interim/clean.csv"),
                              parse_dates=True, infer_datetime_format=True,
                              usecols=['Date/Time', 'Wind Spd (km/h)'],
                              index_col=0)
        logger.info('Pre-normalized data set was loaded.')
    except Exception:
        logger.error('data/interim/clean.csv could not be read.')
        raise ValueError('DataFrame is empty.')
        
    try:
        final_model = keras.models.load_model(str(project_dir /
                                                  "models/final_model.hdf5"))
        logger.info('Final model was loaded.')
    except Exception:
        logger.error('models/final_model.hdf5 could not be loaded.')
        raise ValueError('Model is unavailable.')
        
    return data, clean_wspd, final_model


#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())
#def main(input_filepath, output_filepath):
def main():
    """ Makes predictions using final trained model in (../models)
        and saves output to (../models).
    """
#%%
    logger = logging.getLogger(__name__)
    logger.info('Making predictions with trained model.')
    
    data, clean_wspd, model = load(logger)
    
    test_start = '2018-07'
    test = data.loc[test_start:]
    test_y = test.pop('Wind Spd (km/h)')

#    trained_predictions = Series(trained_model.predict(test)[:,0], \
#                                 index=test_y.index, name='Wind Spd (km/h)')
    final_predictions = Series(model.predict(test)[:, 0],
                               index=test_y.index, name='Wind Spd (km/h)')
    
    mse = model.evaluate(test, test_y, verbose=0)
    print('>>> Testing MSE of final model: \t{:.8f}'.format(mse))
    
    # Restore original scale of values
    minimum, maximum = float(clean_wspd.min()), float(clean_wspd.max())
    norm_predictions = final_predictions * (maximum - minimum) + minimum
    
    norm_predictions.to_csv(str(project_dir / "models/predictions.csv"),
                            header=False)
    logger.info('Model predictions saved.')
   
#%%
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename='out.log', level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()