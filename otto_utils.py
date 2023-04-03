import os
from multiprocessing import Pool
import datetime
import gc
import itertools
import time

import pandas as pd
import cudf

# Ranker imports
import numpy as np
import polars as pl
from lightgbm.sklearn import LGBMRanker
import lightgbm
from sklearn.model_selection import GroupKFold
from lightgbm.callback import early_stopping, log_evaluation

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

################## Ranker Feature Preprocesing ############
def get_cands_pl(preds, df_train, df_valid, atype, grps, DATE):
    s = df_valid.session.unique()
    df_train = pl.from_pandas(df_train)
    df_train = df_train.with_column(pl.col('session').cast(pl.Int32))
    df_valid = pl.from_pandas(df_valid)
    df_valid = df_valid.with_column(pl.col('session').cast(pl.Int32))
    
    dfc = pl.DataFrame({'user': s, 'item': preds})
    dfc = dfc.explode('item')
    dfc = dfc.with_column(pl.col('item').cast(pl.Int32))

    # Add column with orig20 pred features and order
    lrank20 = list(range(1,21))
    lorig20, lorder = [], []
    for p in preds:
        num_zeros = len(p)-20
        lorig20.extend([1]*20+[-1]*num_zeros)
        lorder.extend(lrank20 + [-1]*num_zeros)
    dfc = dfc.with_columns([
        pl.Series(name='orig20', values=lorig20).cast(pl.Int8),
        pl.Series(name='origOrder', values=lorder).cast(pl.Int8),
    ])
    
    item_feats = get_item_feats_pl(pl.concat([df_train, df_valid]))
    user_feats = get_user_feats_pl(df_valid)
    
    dfc = dfc.join(item_feats, on='item', how='left').fill_nan(-1)
    dfc = dfc.with_column(pl.col('user').cast(pl.Int32))
    dfc = dfc.join(user_feats, on='user', how='left').fill_nan(-1)
    
    fn = f'data/{DATE}_val_labels_gt.pkl'
    df_gt = pd.read_pickle(fn)
    df_gt = pl.from_pandas(df_gt)
    df_gt = df_gt.rename({'session': 'user'})
    df_gt = df_gt.with_column(pl.col('user').cast(pl.Int32))
    df_gt = df_gt.filter(df_gt['type']==atype)

    item_labels = df_gt.explode('labels')
    item_labels = item_labels.with_column(pl.col('labels').cast(pl.Int32))
    item_labels = item_labels.rename({'labels': 'item'})

    df_gt = df_gt[['user']]
    df_gt = df_gt.join(item_labels, on='user', how='left')
    df_gt = df_gt.with_column(pl.lit(1).alias(atype).cast(pl.Int8))
    df_gt = df_gt.drop('type')
    
    dfc = dfc.join(df_gt, on=['user', 'item'], how='left').fill_null(0)
    dfc = dfc.groupby('user').head(max(grps))
    
    dfc = dfc.with_columns([
        pl.col('item_buy_ratio').cast(pl.Float32),
        pl.col('user_user_count').cast(pl.UInt16),
        pl.col('user_item_count').cast(pl.UInt16),
        pl.col('user_buy_ratio').cast(pl.Float32)
    ])
    return dfc

def get_item_feats_pl(df):
    aid_feats = df.groupby('aid').agg([
        pl.count('aid').alias('item_item_count'),
        pl.n_unique('session').alias('item_user_count'),
        pl.mean('type').alias('item_buy_ratio'),
        # 1-29-23 Added ratios of item_clicks/carts/orders per aid and types
        pl.col('type').filter(pl.col('type')==0).count().alias('item_clicks'),
        pl.col('type').filter(pl.col('type')==1).count().alias('item_carts'),
        pl.col('type').filter(pl.col('type')==2).count().alias('item_orders'),
        pl.col('type').count().alias('item_events')
    ])
    aid_feats = aid_feats.with_columns([
        pl.col('item_clicks')/pl.col('item_events').alias('item_clicks'),
        pl.col('item_carts')/pl.col('item_events').alias('item_carts'),
        pl.col('item_orders')/pl.col('item_events').alias('item_orders'),
    ])
    aid_feats = aid_feats.with_columns([
        pl.col('item_clicks').cast(pl.Float32),
        pl.col('item_carts').cast(pl.Float32),
        pl.col('item_orders').cast(pl.Float32)
    ])
    aid_feats = aid_feats.rename({'aid': 'item'})
    return aid_feats

def get_user_feats_pl(df):
    user_feats = df.groupby('session').agg([
        pl.count('session').alias('user_user_count'),
        pl.n_unique('aid').alias('user_item_count'),
        pl.mean('type').alias('user_buy_ratio'),
        ## 1-30-23
        pl.count('aid').alias('num_events')
    ])
    user_feats = user_feats.rename({'session': 'user'})
    return user_feats

def apply(df, pipeline):
    for f in pipeline:
        df = f(df)
    return df

def add_action_num_reverse_chrono_pl(df):
    return df.select([
        pl.col('*'),
        pl.col('session').cumcount().reverse().over('session').alias('action_num_reverse_chrono')
    ])

def add_session_length_pl(df):
    return df.select([
        pl.col('*'),
        pl.col('session').count().over('session').alias('session_length')
    ])

def add_log_recency_score_pl(df):
    linear_interpolation = 0.1 + ((1-0.1) / (df['session_length']-1)) * (df['session_length']-df['action_num_reverse_chrono']-1)
    return df.with_columns(pl.Series(2**linear_interpolation - 1).alias('log_recency_score')).fill_nan(1)

def make_feats_radek_pl(df_valid):
    df_valid = pl.from_pandas(df_valid)
    pipeline = [add_session_length_pl, add_action_num_reverse_chrono_pl,
                add_log_recency_score_pl]
    df_valid = apply(df_valid, pipeline)
    
    df_valid = df_valid.rename({'session':'user', 'aid':'item'})
    df_valid = df_valid.with_columns([
        pl.col('session_length').cast(pl.Int16),
        pl.col('action_num_reverse_chrono').cast(pl.Int16),
        pl.col('log_recency_score').cast(pl.Float32),
    ])
    return df_valid

def make_all_feats_pl(dfc, df_val_r, atype):
    d_type2id = pd.read_pickle('data/d_type2id.pkl')
    df_val_r = df_val_r.filter(df_val_r['type']==d_type2id[atype])
    df_val_r = df_val_r.unique(subset=['user', 'item'], keep='last') # polars drop_duplicates
    df_val_r = df_val_r.with_column(pl.col('user').cast(pl.Int32))
    dfc = dfc.join(df_val_r, on=['user', 'item'], how='left');
    return dfc

def downsample_neg(dfc, atype, neg_frac):
    positives = dfc.loc[dfc[atype]==1]
    negatives = dfc.loc[dfc[atype]==0].sample(frac=neg_frac)
    print(f'Downsamp Befor: % ones: {(dfc[atype].value_counts()/sum(dfc[atype].value_counts()))[1]:.4f}')
    dfc = pd.concat([positives, negatives], axis=0, ignore_index=True)
    dfc = dfc.sort_values('user').reset_index(drop=True)
    print(f'Downsamp After: % ones: {(dfc[atype].value_counts()/sum(dfc[atype].value_counts()))[1]:.4f}')
    return dfc

################## Ranker Training ############
def get_preds_lgbm(test_cands, feats, atype, CFG):
    """
    Inference
    # 1) Make dataframe from test data
    # 2) Make features from 4 weeks train + 1 week test
    # 3) Use saved models to make predictions, ensemble, pick top 20, submit
    """
    preds = np.zeros(len(test_cands))
    for fold in range(CFG['num_splits']):
        if fold == CFG['num_avg']:
            break
        print(f'pred fold: {fold}')
        mdl = lightgbm.Booster(model_file=f'mdls/lgbm_fold{fold}_{atype}.txt')
        preds += mdl.predict(test_cands[feats])/CFG['num_avg']
        
    test_preds = test_cands[['user', 'item']].copy()
    test_preds['pred'] = preds
    # Post process
    # Users[0] = lowest num, pred[0] = highest pred
    test_preds = test_preds.sort_values(by=['user', 'pred'],
                                        ascending=[True, False])[['user', 'item']]
    test_preds = test_preds.reset_index(drop=True)
    test_preds = test_preds.groupby('user').head(20)
    test_preds = test_preds.groupby('user').agg(list)
    test_preds = test_preds.item.to_list()
    return test_preds

def train_lgbm(dfc, atype, feats, CFG):
    skf = GroupKFold(n_splits=CFG['num_splits'])
    splits = skf.split(dfc, dfc[atype], groups=dfc['user'])
    train_curves = []
    
    for fold,(train_idx, valid_idx) in enumerate(splits):
        if fold == CFG['num_avg']:
            break
        #         print(f'new: {fold}')
        X_train = dfc.loc[train_idx, feats]
        y_train = dfc.loc[train_idx, atype]
        X_valid = dfc.loc[valid_idx, feats]
        y_valid = dfc.loc[valid_idx, atype]
        
        train_group = dfc.loc[train_idx].groupby('user').user.count().to_numpy()
        valid_group = dfc.loc[valid_idx].groupby('user').user.count().to_numpy()
        groups = (train_group, valid_group)
        
        ranker = train_lgbm_one_fold(X_train, y_train, X_valid, y_valid, groups, CFG)
        
        metric_name = f"{CFG['metric']}@{CFG['eval_at']}"
        save_iter = len(ranker.evals_result_['valid_0'][metric_name])
        if not os.path.exists('mdls'):
            os.mkdir('mdls')
        ranker.booster_.save_model(f'mdls/lgbm_fold{fold}_{atype}.txt',
                                   num_iteration=save_iter)
#                                    num_iteration=ranker.best_iteration_)
        train_curves.append(ranker.evals_result_['valid_0'][metric_name])
    return ranker, (metric_name, train_curves)

def train_lgbm_one_fold(X_train, y_train, X_valid, y_valid, groups, CFG,):
    d_mdl = CFG['MDL']
    d_temp = d_mdl.copy()
    del d_temp['model']
    
    ranker = LGBMRanker(
        objective="lambdarank",
        metric=CFG['metric'],
        importance_type='gain',
        **d_temp
    )
    ranker.fit(
        X=X_train,
        y=y_train,
        group=groups[0],
        eval_set=[(X_valid, y_valid)],
        eval_group=[groups[1]],
        eval_at=[CFG['eval_at']],
        callbacks=[early_stopping(d_temp['early_stopping']),
                   log_evaluation(100)]
    )
    return ranker