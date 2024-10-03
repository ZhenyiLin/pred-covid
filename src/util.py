import pandas as pd
import numpy as np
import logging
logger = logging.getLogger(__name__)
from datetime import datetime, timedelta
import lightgbm as lgb
import src
import json
import joblib




def load_config():
    """
    load config file
    """
    with open('config.json', 'r') as file:
        config = json.load(file)
    formatted_json = json.dumps(config, indent=4)
    logger.info(f"Loaded CONFIG:\n{formatted_json}")
    return config

def save_model(model_dict, file_name):
    logger.info(f"Saving model to ./models/{file_name}")
    joblib.dump(model_dict, "./models/" + file_name)

def add_days_to_date(ref_date, days):
    date_obj = datetime.strptime(ref_date, '%Y-%m-%d')
    new_date = date_obj + timedelta(days=days)
    return new_date.strftime('%Y-%m-%d')


def train_test_split(df, ref_date='2022-07-01', pred_days=7, gap_days=0):
    """
    train test split
      +----------------------------------+-------------------+-----------------------------------------------------------+
      |2020-05-01  ------> train_end_date| --- gap days----> |Current day (ref_date)                                     |
      +----------------------------------+-------------------+-----------------------------------------------------------+
      |         TRAIN                    |Targets are unknown|                                                           |
      +----------------------------------+-------------------+-----------------------------------------------------------+
      |  CV1                             |                   |                                                           |
      |        CV2                       |                   |                                                           |
      |              ...                 |                   |                                                           |
      |                               CVN|                   |                                                           |
      +----------------------------------+-------------------+-----------------------------------------------------------+            
    """
    train_start_date = '2020-05-01'
    train_end_date = add_days_to_date(ref_date, -gap_days-1)
    test_start_date = add_days_to_date(train_end_date, 1)
    test_end_date = add_days_to_date(test_start_date, pred_days-1)
    
    
    logger.info(f"train / test split: \n  gap days = {gap_days}\n  {train_start_date} <= train <= {train_end_date}\n  {test_start_date} <= test <= {test_end_date}")
    train_df = df.loc[(train_start_date <= df['date']) & (df['date'] <= train_end_date) & df["is_valid_target"], :]
    test_df = df.loc[(test_start_date <= df['date']) & (df['date'] <= test_end_date) & df["is_valid_target"], :]
    logger.info(f"train size: {len(train_df)}; test_size: {len(test_df)}")
    return train_df, test_df


# def score(pred_df):
#     """
#     compute the aggregated MAE score
#       steps:
#         1 - compute MALE
#         2 - agg N days to get location MAE
#         3 - agg locations to get country MAE
#         4 - mean and std MAE
#     """
#     score_df = pred_df.copy()
#     score_df['mae'] = abs(np.log(1+score_df['prediction'])- np.log(1+score_df['target'])) * score_df['populations']
    
#     #score_df = score_df.groupby(['country_name', 'location_key'], as_index=False).agg(mae=('mae', 'mean'))
#     #score_df = score_df.groupby(['country_name'], as_index=False).agg(mae=('mae', 'mean'))
#     score = float(score_df['mae'].mean())
#     logger.info(f"evaluation score =  {score:.4f}")
    return score

def score(populations, preds, targets, verbose=-1):
    """
    score metrics
    """
    total_population = populations.sum()
    score = abs(np.log(1 + preds) - np.log( 1 + targets)) * populations 
    score = float(score.sum() / total_population)
    if verbose > 0:
        logger.info(f"evaluation score =  {score:.4f}")
    return score

def lgb_score(populations):
    def _custom(preds, eval_data):
        dataset_name = eval_data.params.get('name', 'unknown')
        if 'train' in dataset_name:
            populations = populations['train']
        else:
            populations = populations['valid']
        targets = eval_data.get_label()
        is_higher_better = False
        return 'wmale', score(populations, preds, targets), is_higher_better
    return _custom
    


def display_score(score_dict):
    logger.info(f"---- cv / test scores ----")
    n = len(score_dict) - 1
    for i in range(n):
        print(f"cv{i} score = {score_dict[f'cv{i}']:.4f}")
        
    cv_score_mean = np.mean([score_dict[f'cv{i}'] for i in range(n)])
    cv_score_std = np.std([score_dict[f'cv{i}'] for i in range(n)])
    print(f'cv score = {cv_score_mean:.4f}')
    print(f"test score = {score_dict['test']:.4f}")
    print(f"cv score std  = {cv_score_std:.4f}")


def split_dataset_by_level(levels=[1]):
    df = pd.read_csv("./dataset/aggregated.csv", 
                     usecols=['location_key', 'date', 'place_id','country_name', 'aggregation_level', 'new_confirmed', 'cumulative_confirmed', 
                              'new_persons_vaccinated', 'cumulative_persons_vaccinated',                                    
                              'population', 'gdp_usd', 'latitude', 'longitude',
                              'average_temperature_celsius', 'minimum_temperature_celsius', 'maximum_temperature_celsius',
                              'rainfall_mm', 'snowfall_mm', 'relative_humidity'
                                                                 ])
    #df.to_csv("./dataset/agg_small.csv", index=False)

    if 0 in levels:
        # save level 0
        df0 = df[df['aggregation_level']==0]
        df0.to_csv("./dataset/agg0.csv", index=False)

    if 1 in levels:
        # save level 1
        df1 = df[df['aggregation_level']==1]
        df1.to_csv("./dataset/agg1.csv", index=False)

    if 2 in levels:
        # save level 2
        df2 = df[df['aggregation_level']==2]
        df2.to_csv("./dataset/agg2.csv", index=False)

    if 3 in levels:
        # save level 3
        df3 = df[df['aggregation_level']==3]
        df3.to_csv("./dataset/agg3.csv", index=False)


def load_dataset(level=1):
    logger.info(f"loading level {level} dataset ...")
    df = pd.read_csv(f"./dataset/agg{level}.csv")
    df['country'] = df['location_key'].str.split("_").str[0]
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['new_confirmed', 'population'])
    df = df.sort_values(by=['location_key', 'date'])
    return df



def fill_missing_values(df):
    """
    fill missing values
    """
    logger.info("Filling missing values ...")
    # Fill by 0
    cols = ['new_confirmed', 'cumulative_confirmed', 'new_persons_vaccinated', 'cumulative_persons_vaccinated',
            'rainfall_mm', 'snowfall_mm']
    df[cols] = df[cols].fillna(0)
    
    
    # Fill by -1
    cols = ['population_rural', 'population_urban', 'population_density', 'human_development_index', 
            'population_age_00_09', 'population_age_10_19', 'population_age_20_29', 'population_age_30_39',
            'population_age_40_49', 'population_age_50_59', 'population_age_60_69', 'population_age_70_79', 
            'population_age_80_and_older', 'gdp_usd', ]
    df[cols] = df[cols].fillna(-1)
    
    
    # Fill missing geolocation
    location_latitude_mapping = {
        "CA_BC": 54,
        "CA_PE": 46.25,
        "JP_44": 33.2,
        "PT_18": 40.67,
        "SL_NW": 8.6
    }
    location_longitude_mapping = {
        "CA_BC": -125,
        "CA_PE": -63,
        "JP_44": 131.6,
        "PT_18": -7.9,
        "SL_NW": -12.5
    }
    df['latitude'] = df[['location_key', 'latitude']].apply(lambda r: location_latitude_mapping.get(r['location_key'], r['latitude']), axis=1)
    df['longitude'] = df[['location_key', 'longitude']].apply(lambda r: location_latitude_mapping.get(r['location_key'], r['longitude']), axis=1)

    return df

def truncate_target_outlier(df, n_times=10):
    """
    remove samples with invalid 'new_confirmed'
    """
    logger.info("truncating target outliers ...")
    df = df.loc[df['new_confirmed'] > 0, :].copy()
    df['new_confirmed_mean_by_location'] = df.groupby('location_key')['new_confirmed'].transform('mean')
    df = df.loc[df['new_confirmed'] <= n_times * df['new_confirmed_mean_by_location'], :]
    return df

