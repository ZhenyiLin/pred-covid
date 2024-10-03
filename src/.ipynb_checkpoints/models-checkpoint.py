import pandas as pd
import logging
import lightgbm as lgb
import numpy as np
import src
logger = logging.getLogger(__name__)

class BaseModel:
    def __init__(self, pred_days):
        self.pred_days = pred_days

    def fit(self, train_df):
        pred_days = self.pred_days
        logger.info(f"---- Base Model fitting --------")
        latest_idx = train_df.groupby('location_key')['date'].idxmax()
        metadata_df = train_df.loc[latest_idx, ['location_key', 'target']]
        metadata_df.columns = ['location_key', 'prediction']
        self.metadata_df = metadata_df

    def transform(self, df):
        logger.info(f"---- Base Model predicting --------")
        df = pd.merge(df.loc[:, ['country_name', 'location_key', 'date', 'target', 'population']], self.metadata_df, how='left', on=['location_key'])
        df = df.fillna(0.0)
        return df


class LGBRollingForecastModel:
    def __init__(self, features, lgb_params, num_boost_round=500, importance_coverage=0.9, seed=10, verbose=-1):
        self.lgb_params = lgb_params
        self.features = features
        self.num_boost_round = num_boost_round
        self.importance_coverage = importance_coverage
        self.verbose = verbose
        self.seed = seed

    def fit(self, train_df):
        logger.info(f"---- LightGBM Rolling Forecast Model fitting --------")
        logger.info(f"    round 1: fit all {len(self.features)} features ...")
        # round 1: fit all features
        self._fit(train_df, self.features)
        self.top_features = self.feature_imporatance_df[self.feature_imporatance_df['importance_coverage']<=self.importance_coverage]['feature_name'].to_list()
        

        logger.info(f"    round 2: fit top {len(self.top_features)} features ...")
        # round 2: fit selected features
        self._fit(train_df, self.top_features)
        if self.verbose > 1:
            display(self.feature_imporatance_df)
        elif self.verbose > 0:
            display(self.feature_imporatance_df.iloc[:len(self.top_features)])

    def _fit(self, train_df, features):
        lgb_params = self.lgb_params
        lgb_params['seed'] = self.seed
        lgb_train_dataset = lgb.Dataset(train_df[features], label=train_df['target'], params={'name':'train'})
        lgb_estimator = lgb.train(lgb_params, 
                                  lgb_train_dataset, 
                                  feval = src.util.lgb_score({'train': train_df['population']}),
                                  num_boost_round = self.num_boost_round,
                                 )
        feature_imporatance_df = pd.DataFrame({
            'feature_name':lgb_estimator.feature_name(),
            'feature_importance':lgb_estimator.feature_importance()}
        ).sort_values('feature_importance',ascending=False)

        total_importance = feature_imporatance_df['feature_importance'].sum()
        feature_imporatance_df['importance_coverage'] = feature_imporatance_df['feature_importance'].cumsum() / total_importance
        
        self.lgb_estimator = lgb_estimator
        self.feature_imporatance_df = feature_imporatance_df
        

    def transform(self, df):
        logger.info(f"---- LightGBM Rolling Forecast Model transform --------")
        features = self.top_features
        df = df.copy()
        df['prediction'] = self.lgb_estimator.predict(df[features])
        df = df[['country_name', 'location_key', 'date', 'target_date', 'target', 'prediction', 'population']]
        return df


class LGBRecursiveForecastModel:
    def __init__(self, full_df, recursion_pipeline, features, lgb_params, num_boost_round=500, importance_coverage=0.9, seed=10, verbose=-1):
        self.lgb_params = lgb_params
        self.features = features
        self.num_boost_round = num_boost_round
        self.importance_coverage = importance_coverage
        self.verbose = verbose
        self.seed = seed
        self.full_df = full_df
        self.recursion_pipeline = recursion_pipeline

    def fit(self, train_df):
        logger.info(f"---- LightGBM Recursive Forecast Model fitting --------")
        logger.info(f"    round 1: fit all {len(self.features)} features ...")
        # round 1: fit all features
        self._fit(train_df, self.features)
        self.top_features = self.feature_imporatance_df[self.feature_imporatance_df['importance_coverage']<=self.importance_coverage]['feature_name'].to_list()
        

        logger.info(f"    round 2: fit top {len(self.top_features)} features ...")
        # round 2: fit selected features
        self._fit(train_df, self.top_features)
        if self.verbose > 1:
            display(self.feature_imporatance_df)
        elif self.verbose > 0:
            display(self.feature_imporatance_df.iloc[:len(self.top_features)])

    def _fit(self, train_df, features):
        lgb_params = self.lgb_params
        lgb_params['seed'] = self.seed
        lgb_train_dataset = lgb.Dataset(train_df[features], label=train_df['target'], params={'name':'train'})
        lgb_estimator = lgb.train(lgb_params, 
                                  lgb_train_dataset, 
                                  feval = src.util.lgb_score({'train': train_df['population']}),
                                  num_boost_round = self.num_boost_round,
                                 )
        feature_imporatance_df = pd.DataFrame({
            'feature_name':lgb_estimator.feature_name(),
            'feature_importance':lgb_estimator.feature_importance()}
        ).sort_values('feature_importance',ascending=False)

        total_importance = feature_imporatance_df['feature_importance'].sum()
        feature_imporatance_df['importance_coverage'] = feature_imporatance_df['feature_importance'].cumsum() / total_importance
        
        self.lgb_estimator = lgb_estimator
        self.feature_imporatance_df = feature_imporatance_df
        

    def transform(self, df):
        logger.info(f"---- LightGBM Recursive Forecast Model transform --------")
        features = self.top_features
        df = df.copy()

        # prepare rolling_df, on which we will fill target by our prediction
        src.logging_config.setup_logging(logging.ERROR)
        min_date = df['date'].min() - pd.DateOffset(days=56)
        max_date = df['date'].max()
        rolling_df = self.full_df[(self.full_df['date'] >= min_date) & (self.full_df['date'] <= max_date)].copy()
        sorted_dates = sorted(df['date'].unique())
    

        daily_score = {}
        for i, date in enumerate(sorted_dates):
            # get current day data
            daily_df = rolling_df[rolling_df['date']==date].copy()
        
            # predict next day
            daily_df['prediction'] = self.lgb_estimator.predict(daily_df[features])
            score = src.util.score(daily_df['population'], daily_df['prediction'], daily_df['target'])
            daily_score[i] = score
            print(date, score)
        
            # update df
            if 'prediction' not in df.columns:
                df = df.merge(daily_df[['date', 'location_key', 'prediction']], on=['date', 'location_key'], how='left')
            else:
                df = df.merge(daily_df[['date', 'location_key', 'prediction']], on=['date', 'location_key'], how='left', suffixes=('', '_new'))
                df['prediction'] = np.where(df['prediction_new'].notna(), df['prediction_new'], df['prediction'])
                df.drop(columns=['prediction_new'], inplace=True)
    
            # use next day prediction to update next day
            if i < len(sorted_dates) - 1:
                next_day_df = daily_df[['date', 'location_key', 'prediction', 'population']].copy()
                next_day_df['date'] = next_day_df['date'] + pd.DateOffset(days=1)
                next_day_df['new_confirmed'] = (next_day_df['prediction'] * next_day_df['population'] / 100).astype('int')
                next_day_df = next_day_df[['date', 'location_key', 'new_confirmed']].set_index(['date', 'location_key'])
                rolling_df.set_index(['date', 'location_key'], inplace=True)
                rolling_df.update(next_day_df)
                rolling_df.reset_index(inplace=True)
            
                # regenerate `new_confirmed` based features
                for t in self.recursion_pipeline:
                    rolling_df = t.transform(rolling_df)
        self.daily_score = daily_score   
        src.logging_config.setup_logging(logging.INFO)
        pred_df = df[['country_name', 'location_key', 'date', 'target', 'target_date', 'prediction', 'population']]
        return pred_df


class AveragedModel:
    def __init__(self, models):
        self.models = models

    def fit(self, train_df):
        logger.info(f"---- Averaged Model fitting --------")
        feature_importance_dfs = []
        for i, model in enumerate(self.models):
            logger.info(f"    fitting model {i}")
            model.fit(train_df) 

            # merge feature importance
            tmp = model.feature_imporatance_df[['feature_name', 'feature_importance']]
            tmp.columns=['feature_name', f'feature_importance_{i}']
            feature_importance_dfs.append(tmp)

        merged_feature_importance_df = feature_importance_dfs[0]
        for df in feature_importance_dfs[1:]:
            merged_feature_importance_df = pd.merge(merged_feature_importance_df, df, on='feature_name', how='outer')
        merged_feature_importance_df['feature_importance'] = merged_feature_importance_df.filter(like='feature_importance_').mean(axis=1).astype('int')
        self.feature_importance_df = merged_feature_importance_df
    

    def transform(self, df):
        logger.info(f"---- Averaged Model transformation --------")
        preds = []
        for model in self.models:
            pred_df = model.transform(df)
            preds.append(pred_df['prediction'])
        preds = pd.concat(preds, axis=1).mean(axis=1)
        pred_df['prediction'] = preds
        return pred_df


class CVModel:
    def __init__(self, cv_models):
        self.cv_models = cv_models
        
    def transform(self, df):
        preds = None
        cv_cnt = len(self.cv_models)
        for cv_id, cv_model in self.cv_models.items():
            pred_df = cv_model.transform(df)
            if preds is None:
                preds = pred_df['prediction'] 
            else:
                preds += pred_df['prediction']
        preds = preds / cv_cnt
        pred_df['prediction'] = preds
        return pred_df


class LGBRollingForecastByCountryModel:
    def __init__(self, features, lgb_params, num_boost_round=500):
        self.lgb_params = lgb_params
        self.features = features
        self.num_boost_round = num_boost_round

    def fit(self, train_df, valid_df):
        features = self.features
        lgb_params = self.lgb_params
        self.lgb_estimator = {}
        feature_importance = None
        country_cnt = train_df['country_name'].nunique()
        for country_name in train_df['country_name'].unique():
            train = train_df[train_df['country_name']==country_name].copy()
            valid = valid_df[valid_df['country_name']==country_name].copy()
            if len(valid) == 0:
                continue
            lgb_train_dataset = lgb.Dataset(train[features], label=train['target'], params={'name':'train'})
            lgb_valid_dataset = lgb.Dataset(valid[features], label=valid['target'], params={'name':'valid'})
            lgb_estimator = lgb.train(lgb_params, 
                                      lgb_train_dataset, 
                                      feval = src.util.lgb_score(train['population'], valid['population']),
                                      num_boost_round=self.num_boost_round
                                     )
            # display(pd.DataFrame({'feature name':lgb_estimator.feature_name(),
            #                      'feature importance':lgb_estimator.feature_importance()}).sort_values('feature importance',ascending=False))
            
            tmp = pd.DataFrame({'feature name':lgb_estimator.feature_name(),
                                'feature importance':lgb_estimator.feature_importance() / country_cnt})
            tmp = tmp.set_index('feature name')
            if feature_importance is None:
                feature_importance = tmp
            else:
                feature_importance += tmp
            self.lgb_estimator[country_name] = lgb_estimator
        self.feature_importance = feature_importance
        display(feature_importance.sort_values('feature importance',ascending=False))

    def transform(self, df):
        features = self.features
        dfs = []
        for country_name in df['country_name'].unique(): 
            tmp = df.loc[df['country_name']==country_name, :].copy()
            tmp['prediction'] = self.lgb_estimator[country_name].predict(tmp[features])
            tmp = tmp[['country_name', 'location_key', 'date', 'target', 'prediction', 'population']]
            dfs.append(tmp)
        return pd.concat(dfs, axis=0)