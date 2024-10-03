import sklearn.preprocessing
import pandas as pd
import numpy as np
import logging
import src
logger = logging.getLogger(__name__)

class CovidDataLoader:
    def __init__(self, level=1):
        """
        Load the Covid Dataset by level
            level 0: country
            level 1: state
            level 2: county
            level 3: city (very tiny group and may span level 1 & 2)
        """
        self.level = level
    def transform(self, df=None):
        logger.info(f"---- Covid Data Loader --------")
        logger.info(f"  loading dataset ...")
        df = pd.read_csv(f"./dataset/agg{self.level}.csv")
        logger.info(f"  cleaning dataset ...")
        df['country'] = df['location_key'].str.split("_").str[0]
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=['new_confirmed', 'population'])
        logger.info(f"  sorting dataset by location and date...")
        df = df.sort_values(by=['location_key', 'date'])
        return df


class TargetCreator:
    def __init__(self, shift_day=1):
        """
        create target variable as new_confirmed / population
        """
        self.shift_day = -shift_day

    def transform(self, df):
        logger.info(f"---- Target Creator (new_confirmed / population * 100) --------")
        logger.info(f" shifting {self.shift_day} day(s)...")
        df = src.util.truncate_target_outlier(df)
        df['new_infection_rate'] = df['new_confirmed']/df['population']*100
        df['cumsum_infection_rate'] = df.groupby('location_key')['new_confirmed'].cumsum()
        df['target'] = df['new_infection_rate'].shift(self.shift_day)
        df['target_date'] = df['date'].shift(self.shift_day)
        df['is_valid_target'] = (df['date'] - df['target_date']).dt.days == self.shift_day
        return df

#===================================================================
# Location Features
#===================================================================
class LocationRollingMean:
    def __init__(self, shift_day=0, rolling_days=[]):
        self.shift_day = shift_day
        self.rolling_days = rolling_days

    def transform(self, df):
        logger.info(f"---- Location Rolling Mean Feature --------")
        for rolling_day in self.rolling_days:
            output_col = f'loc_roll{rolling_day}_mean'
            logger.info(f"rolling {rolling_day} days -> {output_col}")
            df[output_col] = df.groupby(['location_key'])['new_confirmed']\
                               .transform(lambda x: x.shift(self.shift_day).rolling(rolling_day).mean()) / df['population']
        return df 

class LocationRollingStd:
    def __init__(self, shift_day=0, rolling_days=[]):
        self.shift_day = shift_day
        self.rolling_days = rolling_days

    def transform(self, df):
        logger.info(f"---- Location Rolling Std Feature --------")
        for rolling_day in self.rolling_days:
            output_col = f'loc_roll{rolling_day}_std'
            logger.info(f"rolling {rolling_day} days -> {output_col}")
            df[output_col] = df.groupby(['location_key'])['new_confirmed']\
                               .transform(lambda x: x.shift(self.shift_day).rolling(rolling_day).std()) / df['population']
        return df 

class LocationSameWeekDayRollingMean:
    def __init__(self, shift_day=0, rolling_days=[]):
        self.shift_day = shift_day
        self.rolling_days = rolling_days

    def transform(self, df):
        logger.info(f"---- Location Same Weekday Rolling Mean Feature --------")
        for rolling_day in self.rolling_days:
            output_col = f'loc_same_weekday_roll{rolling_day}_mean'
            logger.info(f"rolling {rolling_day} days -> {output_col}")
            df[output_col] = df.groupby(['location_key', 'weekday'])['new_confirmed']\
                               .transform(lambda x: x.shift(self.shift_day).rolling(rolling_day).mean()) / df['population']
        return df 


class LocationDurationDays:
    def __init__(self):
        pass

    def transform(self, df):
        logger.info(f"---- Location Duration in days Feature --------")
        df['location_duration_days'] = df.groupby('location_key')['date'].transform(lambda x: (x - x.min()).dt.days)
        return df


class LocationLag:
    def __init__(self, lag_days=[0, 1, 2, 3, 4, 5, 6, 7]):
        self.lag_days = lag_days

    def transform(self, df):
        logger.info(f"---- Location lags in {self.lag_days} --------")
        for lag_day in self.lag_days:
            df[f'loc_lag{lag_day}'] = df.groupby('location_key')['new_confirmed'].transform(lambda x: x.shift(lag_day)) / df['population'] * 100
        return df


class LocationRainRollingSum:
    def __init__(self, shift_day=0, rolling_days=[1, 7, 14]):
        self.shift_day = shift_day
        self.rolling_days = rolling_days

    def transform(self, df):
        logger.info(f"---- Location Rain Rolling Sum in {self.rolling_days} --------")
        for rolling_day in self.rolling_days:
            df[f'loc_rain_roll{rolling_day}_sum'] = df.groupby('location_key')['rainfall_mm']\
                                                      .transform(lambda x: x.shift(self.shift_day).rolling(rolling_day).sum())
        return df

#===================================================================
# Country Features
#===================================================================
class CountryPopulation:
    def __init__(self):
        pass

    def transform(self, df):
        logger.info(f"---- Country Population Feature --------")
        location_population_df = df.groupby(['country_name', 'location_key'], as_index=False)['population'].mean()
        country_population_df = location_population_df.groupby(['country_name'], as_index=False)['population'].sum()
        country_population_df = country_population_df.rename(columns={'population': 'country_population'})
        df = pd.merge(df, country_population_df, on=['country_name'], how='left')
        return df


class CountryRollingMean:
    def __init__(self, shift_day=0, rolling_days=[]):
        self.shift_day = shift_day
        self.rolling_days = rolling_days

    def transform(self, df):
        logger.info(f"---- Location Rolling Mean Feature --------")
        for rolling_day in self.rolling_days:
            output_col = f'loc_roll{rolling_day}_mean'
            logger.info(f"rolling {rolling_day} days -> {output_col}")
            df[output_col] = df.groupby(['country_name'])['new_confirmed']\
                               .transform(lambda x: x.shift(self.shift_day).rolling(rolling_day).mean()) / df['country_population']
        return df 

class CountryRollingStd:
    def __init__(self, shift_day=0, rolling_days=[]):
        self.shift_day = shift_day
        self.rolling_days = rolling_days

    def transform(self, df):
        logger.info(f"---- Location Rolling Std Feature --------")
        for rolling_day in self.rolling_days:
            output_col = f'loc_roll{rolling_day}_std'
            logger.info(f"rolling {rolling_day} days -> {output_col}")
            df[output_col] = df.groupby(['country_name'])['new_confirmed']\
                               .transform(lambda x: x.shift(self.shift_day).rolling(rolling_day).std()) / df['country_population']
        return df 



#=====================================================
# Date
#=====================================================
class Weekday:
    def __init__(self):
        pass

    def transform(self, df):
        logger.info(f"---- Week day --------")
        df['weekday'] = df['target_date'].dt.weekday
        return df

class IsSunday:
    def __init__(self):
        pass

    def transform(self, df):
        logger.info(f"---- Is Sunday--------")
        df['is_sunday'] = (df['target_date'].dt.weekday == 6).astype(int)
        return df


class GapDays:
    def __init__(self, gap_days=1):
        self.gap_days = gap_days

    def transform(self, df):
        logger.info(f"---- Gap days --------")
        df['gap_days'] = (df['target_date'] - df['date']).dt.days
        return df


#=====================================================
# Geo
#=====================================================
class DistanceFromCity:
    def __init__(self):
        pass

    def transform(self, df):
        logger.info(f"---- Distance from city--------")


        
        df['distance_from_cn_hb'] = np.sqrt((df['latitude'] - 31.2) ** 2 + (df['longitude'] - 112.3)**2)
        df['distance_from_cn_sh'] = np.sqrt((df['latitude'] - 31.2) ** 2 + (df['longitude'] - 121.4)**2)
        df['distance_from_us_la'] = np.sqrt((df['latitude'] - 31) ** 2 + (df['longitude'] + 92)**2)
        df['distance_from_us_ny'] = np.sqrt((df['latitude'] - 43) ** 2 + (df['longitude'] + 75)**2)
        df['distance_from_au_nsw'] = np.sqrt((df['latitude'] + 32) ** 2 + (df['longitude'] - 147)**2)
        df['distance_from_in_dl'] = np.sqrt((df['latitude'] - 28.6) ** 2 + (df['longitude'] - 77.2)**2)
        df['distance_from_in_mh'] = np.sqrt((df['latitude'] - 19) ** 2 + (df['longitude'] - 73)**2)
        df['distance_from_in_ka'] = np.sqrt((df['latitude'] - 15) ** 2 + (df['longitude'] - 76)**2)
        df['distance_from_it_21'] = np.sqrt((df['latitude'] - 45.25) ** 2 + (df['longitude'] - 7.9)**2)
        df['distance_from_br_sp'] = np.sqrt((df['latitude'] + 21.8) ** 2 + (df['longitude'] + 49.2)**2)
        df['distance_from_br_rj'] = np.sqrt((df['latitude'] + 22.36) ** 2 + (df['longitude'] + 42.72)**2)
        df['distance_from_city'] = df[['distance_from_cn_hb', 'distance_from_cn_sh', 'distance_from_us_la', 'distance_from_us_ny',
                                       'distance_from_au_nsw', 'distance_from_in_dl', 'distance_from_in_mh', 'distance_from_in_ka',
                                       'distance_from_it_21', 'distance_from_br_sp', 'distance_from_br_rj']].min(axis=1)
        return df


#=====================================================
# Temperature
#=====================================================
class TemperatureRollingMean:
    def __init__(self, shift_day=0, rolling_days=[7, 14, 28, 56]):
        self.rolling_days = rolling_days
        self.shift_day = shift_day

    def transform(self, df):
        logger.info(f"---- Location Temperature Rolling Mean in {self.rolling_days} --------")
        for rolling_day in self.rolling_days:
            df[f'temperature_roll{rolling_day}_mean'] = df.groupby(['location_key'])['average_temperature_celsius'].transform(
                lambda x: x.shift(self.shift_day).rolling(rolling_day).mean())
        return df

class TemperatureRollingStd:
    def __init__(self, shift_day=0, rolling_days=[7, 14, 28, 56]):
        self.rolling_days = rolling_days
        self.shift_day = shift_day

    def transform(self, df):
        logger.info(f"---- Location Temperature Rolling Std in {self.rolling_days} --------")
        for rolling_day in self.rolling_days:
            df[f'temperature_roll{rolling_day}_mean'] = df.groupby(['location_key'])['average_temperature_celsius'].transform(
                lambda x: x.shift(self.shift_day).rolling(rolling_day).std())
        return df
        
#=====================================================
# Relative Humidity
#=====================================================
class RelativeHumidityRollingMean:
    def __init__(self, shift_day=0, rolling_days=[7, 14, 28, 56]):
        self.rolling_days = rolling_days
        self.shift_day = shift_day

    def transform(self, df):
        logger.info(f"---- Location Relative Humidity Rolling Mean in {self.rolling_days} --------")
        for rolling_day in self.rolling_days:
            df[f'rh_roll{rolling_day}_mean'] = df.groupby(['location_key'])['relative_humidity'].transform(
                lambda x: x.shift(self.shift_day).rolling(rolling_day).mean())
        return df

class RelativeHumidityRollingStd:
    def __init__(self, shift_day=0, rolling_days=[7, 14, 28, 56]):
        self.rolling_days = rolling_days
        self.shift_day = shift_day

    def transform(self, df):
        logger.info(f"---- Location Relative Humidity Rolling Std in {self.rolling_days} --------")
        for rolling_day in self.rolling_days:
            df[f'rh_roll{rolling_day}_mean'] = df.groupby(['location_key'])['relative_humidity'].transform(
                lambda x: x.shift(self.shift_day).rolling(rolling_day).std())
        return df

#===================================================================
# Vaccine feature
#===================================================================
class VaccinatedRate:
    def __init__(self):
        pass

    def transform(self, df):
        logger.info(f"---- Vaccine Rate --------")
        df['new_vaccinated_rate'] = df["new_persons_vaccinated"] / df["population"]
        df['cumsum_vaccinated_rate'] = df.groupby('location_key')["new_persons_vaccinated"].cumsum()
        return df


#=====================================================
# Ratio feature
#=====================================================
class RatioFeature:
    def __init__(self):
        pass

    def transform(self, df):
        logger.info(f"---- Ratio Features --------")
        cols = df.columns
        for day in range(1, 100):
            if ('rainfall_nm' in cols) and (f"loc_rain_roll{day}_sum" in cols):
                df['loc_rain_roll{day}_sum_ratio'] = df['rainfall_nm'] / df[f'loc_rain_roll{day}_sum']

            if ('loc_lag0' in cols) and (f'loc_roll{day}_mean' in cols):
                df[f'loc_roll{day}_mean_ratio'] = df['loc_lag0'] / df[f'loc_roll{day}_mean']

            if ('average_temperature_celsius' in cols) and (f'temperature_roll{day}_mean' in cols):
                df[f'temperature_roll{day}_mean_ratio'] = df['average_temperature_celsius'] / df[f'temperature_roll{day}_mean']

            if ('relative_humidity' in cols) and (f'rh_roll{day}_mean' in cols):
                df[f'rh_roll{day}_mean_ratio'] = df['relative_humidity'] / df[f'rh_roll{day}_mean']
        return df


#=====================================================
# Encoder
#=====================================================
class OrdinalEncoder:
    def __init__(self):
        self.location_encoder = sklearn.preprocessing.OrdinalEncoder()
        self.country_encoder = sklearn.preprocessing.OrdinalEncoder()

    def fit(self, df):
        logger.info(f"---- Ordinal Encoder fit --------")
        self.location_encoder.fit(df[['location_key']])
        self.country_encoder.fit(df[['country_name']])

    def transform(self, df):
        logger.info(f"---- Ordinal Encoder transform --------")
        logger.info(f"  outputs are 'location_key_encoded' and 'country_encoded'")
        df['location_key_encoded'] = self.location_encoder.transform(df[['location_key']])
        df['country_encoded'] = self.country_encoder.transform(df[['country_name']])
        return df

class MinDateEncoder:
    def __init__(self):
        pass

    def transform(self, df):
        logger.info(f"---- MinDate Encoder --------")
        # Encode Location
        min_date_mapping = df.groupby('location_key')['date'].min().sort_values().reset_index()
        min_date_mapping['location_key_encoded'] = min_date_mapping.index
        df = df.merge(min_date_mapping[['location_key', 'location_key_encoded']], on='location_key', how='left')

        # Encode Country
        min_date_mapping = df.groupby('country_name')['date'].min().sort_values().reset_index()
        min_date_mapping['country_encoded'] = min_date_mapping.index
        df = df.merge(min_date_mapping[['country_name', 'country_encoded']], on='country_name', how='left')
        
        return df


#========================================================
# Missing Values
#========================================================
class MissingValue:
    def __init__(self, features):
        self.features = features

    def transform(self, df):
        logger.info(f"---- Handle Missing Values --------")
        df = df.replace([np.inf, -np.inf], np.nan)
        for feature in self.features:
            
            if 'roll' in feature:
                logger.info(f" dropping nan {feature} ...")
                df.dropna(subset=[feature], inplace=True)
                
            if feature in ['latitude', 'longitude', 'relative_humidity']:
                 df[feature] = df[feature].fillna(999)

            if feature in ['rainfall_mm', 'snowfall_mm']:
                df[feature] = df[feature].fillna(0)
        return df