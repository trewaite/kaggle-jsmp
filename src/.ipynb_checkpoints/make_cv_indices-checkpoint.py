import numpy as np
import pandas as pd

def create_time_series_folds(df, base_train_days, val_days=40):
    
    '''
    Create base train set and time series split fold on `date` column
    '''
    
    total_days = df['date'].nunique()
    
    total_validation_days = total_days - base_train_days
    
    folds = int(total_validation_days/val_days)
    
    print(f"Creating training set with {base_train_days} days and {folds} validation folds with {val_days} days...")
    
    df.loc[df['date'] < base_train_days, 'fold'] = 'train_fold'
    
    start_day = base_train_days
    for fold in range(0,folds):
        df.loc[(df['date'] >= start_day) & (df['date'] < start_day+val_days), 'fold'] = f'fold_{fold}'
        start_day += val_days
        
    print('Complete.')
        
    return df
    return df

if __name__ == '__main__':
    
    train_index = pd.read_csv('./inputs/train.csv', usecols=['date'])
    
    train_fold_ind = create_time_series_folds(train_index, 
                                              base_train_days = 300, 
                                              val_days = 40)
    
    train_fold_ind.to_csv('./preprocessed/train_fold_ind.csv',index=False)