import sys
import os
import csv
import numpy as np
import pandas as pd
from datetime import datetime as dt

import config


from sklearn.decomposition import PCA, FastICA

SEED = 26
np.random.seed(SEED)

def timeit(func):
    def timed(*args, **kwargs):
        ts = time.time()
        print('Function', func.__name__, 'running...')

        result = func(*args, **kwargs)
        te = time.time()
        shape = result[0].shape
        print('Completed. Dataframe shape: ', shape, 'Time elapsed:', round((te -ts),1), 's')
        print()
        return result
    return timed

@timeit
def create_daily_return_lags(df, lags, na_method):
    
    df_daily = df[['date','fold','resp']].groupby(['date','fold']).mean().reset_index()
    
    features = []
    for lag in tqdm(lags):
        col_name = '_'.join(['resp_daily_lag',str(lag)])
        features.append(col_name)
        df_daily[col_name] = df_daily.groupby(['fold'])['resp'].transform(lambda x: x.shift(lag))
        
    df_daily = df_daily.drop(['resp','fold'], axis=1)
    df = pd.merge(df, df_daily, on=['date'], how='left')
    
    if na_method == -1:
        df[features] = df[features].fillna(-1)
    elif na_method == 'drop':
        df[features] = df[features].dropna()
    
    return df, features

@timeit
def create_last_trade_return_lags(df, lags, na_method):
    
    features = []
    for lag in tqdm(lags):
        col_name = '_'.join(['resp_lag',str(lag)])
        features.append(col_name)
        df[col_name] = df.groupby(['fold'])['resp'].transform(lambda x: x.shift(lag))
    
    if na_method == -1:
        df[features] = df[features].fillna(-1)
    elif na_method == 'drop':
        df[features] = df[features].dropna()
    
    return df, features

@timeit
def running_daily_total_trades(df):
            
    df['running_total_trades'] = (df[['date','fold','ts_id']]
                                  .sort_values('ts_id')
                                  .groupby(['date','fold'])
                                  .transform(lambda x: x.expanding().count())
                                  .astype('int64')
                                  .values)
    
    return df, ['running_total_trades']

@timeit
def create_day_features(df):
    '''
    create day of week features assuming starting monday and 5 day trading week
    '''
    df['dow'] = df['ts_id']%5
    return df, ['dow']

@timeit
def create_target(df, threshold=0):
    df['target'] = df['resp'].apply(lambda x: 1 if x > threshold else 0)
    return df, 'target'

@timeit
def create_pca_features(df, columns, n_components, col_prefix):
    
    df_train = df[df['fold']=='train_fold'][columns]
    df_train = df_train.fillna(0)
    
    pca = PCA(n_components=n_components)
    pca.fit(df_train.values)
    
    features = ["_".join([col_prefix,str(i)]) for i in range(1,n_components+1)]
    
    pca_df = pd.DataFrame(pca.transform(df[columns].fillna(0).values), 
                          columns=features)
    
    df = pd.concat([df, pca_df],axis=1)
    
    return df, features


if __name__ == '__main__':
    
    df = pd.read_csv(config.CV_FILE)
    
    original_features = [f"feature_{f}" for f in range(0,130)]

    df, daily_lag_features = create_daily_return_lags(df, 
                                                      lags=np.arange(1,30,1),
                                                      na_method=-1)

    df, lag_features = create_last_trade_return_lags(df, 
                                                     lags=np.arange(1,30,1),
                                                     na_method=-1)

    df, orig_pca_features = create_pca_features(df,
                                                columns=original_features,
                                                n_components=15,
                                                col_prefix='orginal_pca')

    df, lag_pca_features = create_pca_features(df,
                                               columns=lag_features+daily_lag_features,
                                               n_components=15,
                                               col_prefix='lag_pca')

    df, dow_features = create_day_features(df)
    df, running_total_trades_features = running_daily_total_trades(df)
    df, target_col = create_target(df, threshold=0)
    
    # fill main features missing values
    df[original_features] = df[original_features].fillna(0) # okay as scaled to 0 mean
    
    %%time
    df = df.drop(['resp_1','resp_2','resp_3','resp_4'], axis=1)
    df = df.set_index(['date','ts_id','weight','resp'])
    
    date = str(dt.today().date()).replace("-","_")
    df.to_parquet(f"../preprocessed/all_features_{date}.parquet",
                  engine='fastparquet',
                  compression='gzip')