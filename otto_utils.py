import os
from multiprocessing import Pool
import datetime
import gc
import itertools
import time

# import numpy as np
import pandas as pd
# import polars as pl
import cudf

################## Config ############
def iterate_dict(d)->dict:
    keys, values = zip(*d.items()) # Gets keys and values in deterministic order
    df_config = pd.DataFrame(list(itertools.product(*values)), columns=keys)
    df_config = df_config.to_dict('records')
    return df_config

def create_config(d)->pd.DataFrame:
    keys, values = zip(*d.items()) # Gets keys and values in deterministic order
    df_config = pd.DataFrame(list(itertools.product(*values)), columns=keys)
    df_config = df_config.reset_index(drop=True)
    return df_config

################## Helper Functions ############
def trange(df: pd.Series):
    start = pd.to_datetime(df.ts.min()*1e9)
    stop = pd.to_datetime(df.ts.max()*1e9)
    print(start)
    print(stop)
    print(stop-start)
    
def get_recall(submission, DO_LOCAL_VALIDATION, DATE, VERBOSE=True):
    if DO_LOCAL_VALIDATION:
        VERBOSE=True
        FN_GT = f'data/{DATE}_val_labels_gt.pkl'
        ground_truth = pd.read_pickle(FN_GT)

        # Revised recall calculation to use first 20 events after validation input timestamp
        sub_with_gt = submission.merge(ground_truth[['session', 'type', 'labels']],
                                       how='left',
                                       on=['session', 'type'])
        sub_with_gt = sub_with_gt[~sub_with_gt.labels_y.isna()]
        sub_with_gt['hits'] = sub_with_gt.parallel_apply(
            lambda row: len(set(row.labels_x).intersection(list(dict.fromkeys(row.labels_y))[:20])), axis=1)
        sub_with_gt['gt_count'] = sub_with_gt.labels_y.apply(len).clip(0,20)

        grp = sub_with_gt.groupby(['type'])
        recall_per_type = grp['hits'].sum() / grp['gt_count'].sum()
        val_score = (recall_per_type * pd.Series({'clicks': 0.10, 'carts': 0.30, 'orders': 0.60})).sum()
        if VERBOSE:
            print('======================')
            for col in ['clicks', 'carts', 'orders']:
                print(f'{col} Recall: {recall_per_type[col]:0.4f}')
            print('======================')
        print(f'Overall Recall: {val_score:0.5f}')
        if VERBOSE:
            print('======================')
        return recall_per_type['clicks'], recall_per_type['carts'], recall_per_type['orders'], val_score
    else:
        # For test submission
        sub = submission
        sub['session'] = sub['session'].astype(str)
        sub['session_type'] = sub.session.str.cat(sub.type, sep='_')
        sub = sub[['session_type', 'labels']]
        sub.to_csv('submission.zip', index=False)
        sub.head(3)
               
    
################## Covisit Functions ############
def clicks_covisit(
        df: pd.DataFrame, num_cand=20, DAY_RANGE=1, chunk_size=200_000, return_pandas=True
)-> pd.DataFrame:
    """
    # time weighting, cart/order -> cart/order
    # Use most recent 30 events
    # Day range = +- 1
    # cand_thres = 20
    """
    for idx in range(0, df.session.max(), chunk_size):
        df_chunk = df.loc[(idx<=df.session) & (df.session<(idx+chunk_size))]
        df_chunk = clicks_process(df_chunk, num_cand, DAY_RANGE)
        df_chunk = df_chunk.reset_index()
        if idx==0:
            tmp = df_chunk
        else:
            tmp = tmp.append(df_chunk)
    tmp = tmp.groupby(['aid_x','aid_y']).wgt.sum().reset_index()        
    tmp = tmp.sort_values(['aid_x','wgt'], ascending=[True, False])     
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
    tmp = tmp.loc[tmp.n < num_cand].drop('n', axis=1) # After, tmp_cols [aid_x, aid_y, wgt]
    tmp = tmp.reset_index(drop=True)
    if return_pandas:
        tmp = tmp.to_pandas()
    gc.collect()
    return tmp

def clicks_process(
    df: pd.DataFrame, num_cand, DAY_RANGE
)-> cudf.DataFrame:
    """ For each chunk, use most recent 30 sessions, Get aid_x, aid_y time weights (wgt) """
    df = cudf.from_pandas(df)
    # For each sessions, get top 30 most recent events
    df = df.sort_values(['session','ts'], ascending=[True, False])
    df = df.reset_index(drop=True)
    df['n'] = df.groupby('session').cumcount() # 'n' does order indexing for filter next
    df = df.loc[df.n < 30].drop('n', axis=1)
    
    # Create all aid_x, aid_y permutations
    df = df.merge(df, on='session') 
    
    # Filter based on +- DAY_RANGE, and remove duplicates
    df = df.loc[(df.ts_x-df.ts_y).abs() < DAY_RANGE*24*60*60]
    df = df.loc[(df.aid_x != df.aid_y)]
    df = df[['session','aid_x','aid_y','ts_x']].drop_duplicates(['session','aid_x','aid_y'])
    
    # Added for time weighting over 5 weeks -> Linear*3
    # Max / min timestamp = 1662328791 / 1659304800
    df['wgt'] = 1 + 3*(df.ts_x-1659304800) / (1662328791-1659304800)
    df['wgt'] = df.wgt.astype('float32')
    df = df[['aid_x','aid_y','wgt']]
    return df

def carts_covisit(
        df: pd.DataFrame, num_cand=15, DAY_RANGE=1, chunk_size=200_000, return_pandas=True
)-> pd.DataFrame:
    """
    # type weight, click/cart/order -> cart/order
    # Use first 30 events
    # Day range = +- 1
    # cand_thres = 15
    """
    for idx in range(0, df.session.max(), chunk_size):
        df_chunk = df.loc[(idx<=df.session) & (df.session<(idx+chunk_size))]
        df_chunk = carts_process(df_chunk, num_cand, DAY_RANGE)
        df_chunk = df_chunk.reset_index()
        if idx==0:
            tmp = df_chunk
        else:
            tmp = tmp.append(df_chunk)
            # 1-11-23 Moving this groupby out of loop helped a lot
    #         tmp = tmp.groupby(['aid_x', 'aid_y']).wgt.sum().reset_index()
    tmp = tmp.groupby(['aid_x', 'aid_y']).wgt.sum().reset_index()        
    tmp = tmp.sort_values(['aid_x', 'wgt'], ascending=[True, False])     
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
    tmp = tmp.loc[tmp.n<num_cand].drop('n', axis=1) # After, tmp_cols [aid_x, aid_y, wgt]
    tmp = tmp.reset_index(drop=True)
    if return_pandas:
        tmp = tmp.to_pandas()
    gc.collect()
    return tmp

def carts_process(
    df: pd.DataFrame, num_cand, DAY_RANGE, type_weights2={0: 0.5, 1: 9, 2:0.5}
)-> cudf.DataFrame:
    """ For each chunk, use most recent 30 sessions, Get aid_x, aid_y type weights (wgt) """
    df = cudf.from_pandas(df)
    # For each sessions, get top 30 most recent events
    df = df.sort_values(['session', 'ts'], ascending=[True, False])
    df = df.reset_index(drop=True)
    df['n'] = df.groupby('session').cumcount() # 'n' does order indexing for filter next
    df = df.loc[df.n<(30)].drop('n', axis=1)

    # Create all aid_x, aid_y permutations
    df = df.merge(df, on='session')
    
    # Filter based on +- DAY_RANGE, and remove duplicates
    df = df.loc[(df.ts_x-df.ts_y).abs() < DAY_RANGE*24*60*60]
    df = df.loc[(df.aid_x != df.aid_y)]
    df = df[['session', 'aid_x', 'aid_y', 'type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
    
    # Add type weighting
    df['wgt'] = df.type_y.map(type_weights2)
    df = df[['aid_x','aid_y','wgt']]
    return df

def b2b_covisit(
        df: pd.DataFrame, num_cand=15, DAY_RANGE=14, chunk_size=500_000, return_pandas=True
)-> pd.DataFrame:
    """
    # no weighting, cart/order -> cart/order
    # Use most recent 30 events
    # Day range = +- 14
    # cand_thres = 15
    """
    idx_reset = df.session.max()/chunk_size//10
    for ireset, idx in enumerate(range(0, df.session.max(), chunk_size)):
        df_chunk = df.loc[(idx<=df.session) & (df.session<(idx+chunk_size))]
        df_chunk = b2b_process(df_chunk, num_cand, DAY_RANGE)
        df_chunk = df_chunk.reset_index(drop=True)
        if idx==0:
            tmp = df_chunk
        else:
            tmp = tmp.append(df_chunk)
        if idx_reset==0 or ireset%idx_reset == 0:
            tmp = tmp.groupby(['aid_x', 'aid_y']).wgt.sum().reset_index()
    tmp = tmp.reset_index(drop=True)
    tmp = tmp.groupby(['aid_x', 'aid_y']).wgt.sum().reset_index()   
    tmp = tmp.sort_values(['aid_x', 'wgt'], ascending=[True, False])     
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
    tmp = tmp.loc[tmp.n<num_cand].drop('n', axis=1) # After, tmp_cols [aid_x, aid_y, wgt]
    tmp['wgt'] = tmp.wgt.astype('float32')
    tmp = tmp.reset_index(drop=True)
    if return_pandas:
        tmp = tmp.to_pandas()
    gc.collect()
    return tmp

def b2b_process(
    df: pd.DataFrame, num_cand, DAY_RANGE
)-> cudf.DataFrame:
    """ For each chunk, use most recent 30 sessions, for aid_x + aid_y pair, weight=1 (wgt) """
    df = cudf.from_pandas(df)
    # Change filter rule here. Type 1/2 = Carts / Orders
    df = df.loc[df['type'].isin([1,2])]
    
    # For each sessions, get top 30 most recent events
    df = df.sort_values(['session', 'ts'], ascending=[True, False])
    df = df.reset_index(drop=True)
    
    # USE TAIL OF SESSION
    df['n'] = df.groupby('session').cumcount() # 'n' does order indexing for filter next
    df = df.loc[df.n<(30)].drop('n', axis=1)
    
    # Create all aid_x, aid_y permutations
    df = df.merge(df, on='session')
    
    # Filter based on +- DAY_RANGE, and remove duplicates
    df = df.loc[(df.ts_x-df.ts_y).abs() < DAY_RANGE*24*60*60]
    df = df.loc[(df.aid_x != df.aid_y)]
    df = df[['session', 'aid_x', 'aid_y', 'type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
    
    # Wgt=1 for each aid_x, aid_y instance
    df['wgt'] = 1
    df['wgt'] = df.wgt.astype('float32')
    df = df[['aid_x','aid_y','wgt']]
    return df

def preprocess_covisits(df, date):
    print('*Start covisit preprocessing')
    if not os.path.exists('covisit'): os.mkdir('covisit')

    fn = f'covisit/{date}_top_20_clicks_data_datatest.pkl'
    if not os.path.exists(fn):
        print('Preprocess top_20_clicks')
        t1 = time.perf_counter()
        tmp = clicks_covisit(df, chunk_size=100_000)
        tmp = tmp.groupby('aid_x').aid_y.apply(list).to_dict()
        tdur = time.perf_counter()-t1
        print(f'Process time: {tdur/60:0.2f} mins')
        pd.to_pickle(tmp, fn, protocol=4)
    else:
        print('top_20_clicks already exists')

    fn = f'covisit/{date}_top_20_buys_data_datatest.pkl'
    if not os.path.exists(fn):
        print('Preprocess top_20_buys')
        t1 = time.perf_counter()
        tmp = carts_covisit(df, chunk_size=100_000)
        tmp = tmp.groupby('aid_x').aid_y.apply(list).to_dict()
        tdur = time.perf_counter()-t1
        print(f'Process time: {tdur/60:0.2f} mins')
        pd.to_pickle(tmp, fn, protocol=4)
    else:
        print('top_20_buys already exists')

    fn = f'covisit/{date}_top_20_buy2buy_data_datatest.pkl'
    if not os.path.exists(fn):
        print('Preprocess top_20_buy2buy')
        t1 = time.perf_counter()
        tmp = b2b_covisit(df)
        tmp = tmp.groupby('aid_x').aid_y.apply(list).to_dict()
        tdur = time.perf_counter()-t1
        print(f'Process time: {tdur/60:0.2f} mins')
        pd.to_pickle(tmp, fn, protocol=4)
    else:
        print('top_20_buy2buy already exists')
    print('*Finished covisit preprocessing')

################## Suggest Functions ############
def get_preds(t, func, chunk_size=64):
    with Pool(4) as p:
        s1 = p.map(func, t, chunksize=chunk_size)
    preds = []
    for x in s1:
        preds.append(x)
    return preds