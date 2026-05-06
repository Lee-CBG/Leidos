import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

def load_2(
    binary=True, 
    cutoffs=None, 
    include_pid=False, 
    path='Leidos_data_98_Abs_with_clinical_features_v3.csv',
    exclude_hiv=True,
    include_all_vaccines=True
):
    df = pd.read_csv(path)
    
    # Filter for Pfizer/Moderna if the flag is False
    if not include_all_vaccines:
        valid_vax = ['Pfizer', 'Moderna']
        df = df[df['V0_Vaccine'].isin(valid_vax) | df['B0_Vaccine'].isin(valid_vax)]
        
    group = 'B0_Group'
    which = group.split('_')[0]
    pre = df[df[group] == f'Pre {which}']
    post = df[df[group] == f'Post {which}']
    
    result = pd.merge(pre, post, on="PID", how="inner")
    
    # 1. Dynamically identify antibody columns
    ab_start_col = 'A.calcoaceticus LysM_x'
    ab_start_idx = list(result.columns).index(ab_start_col)
    ab_cols = [c for c in result.columns[ab_start_idx:] if '_x' in c]
    
    if exclude_hiv:
        ab_cols = [c for c in ab_cols if 'HIV' not in c]
        
    # 2. Explicitly define metadata features to use (Vaccine columns removed from here)
    cat_col_names = ['Cohort_x', 'Age_Group_x', 'Age_Group_1_x', 'Sex_x', 'Race_x']
    real_col_names = ['Age_x']
    
    # 3. Build x_cols deterministically
    x_cols = cat_col_names + real_col_names + ab_cols
    
    data = result[x_cols + ['Vaccine_Response_class_y', 'Vaccine_Response_raw_y', 'PID']].dropna()
    X = data[x_cols].values
    
    # 4. Calculate dynamic indices based on list lengths
    n_cat = len(cat_col_names)
    n_real = len(real_col_names)
    abs_start_idx = n_cat + n_real  
    
    if cutoffs is not None:
        thresholds = []
        cutoff_cols = [c[:-2] for c in ab_cols] 
        for c in cutoff_cols:
            thresholds.append(cutoffs[c])
        thresholds = np.array(thresholds)

        I = (X[:, abs_start_idx:] >= thresholds).astype(float)
        h = np.maximum(0.0, X[:, abs_start_idx:] - thresholds)
        X_aug = np.concatenate([X, I, h], axis=1)
        X = X_aug
        
    if binary:
        y = (data['Vaccine_Response_class_y'] == 'High').astype(int).values
    else:
        y = data['Vaccine_Response_raw_y'].values
    
    cat_cols_idx = list(range(n_cat))
    
    X_cat = X[:, cat_cols_idx]
    X_real = np.delete(X, cat_cols_idx, axis=1)
    
    encoder = OrdinalEncoder()
    X_cat_encoded = encoder.fit_transform(X_cat)
    
    X_processed = np.hstack([X_cat, X_cat_encoded, X_real])
    
    emb_list = list(X_processed[:, n_cat : n_cat*2].max(0).astype(int) + 1)
    real_size = X_processed.shape[1] - (n_cat * 2)
    
    if not include_pid:
        return X_processed, y, emb_list, real_size, encoder
    else:
        return X_processed, y, data['PID'].values, emb_list, real_size, encoder


def load_6(
    binary=True, 
    cutoffs=None, 
    include_pid=False, 
    path='Leidos_data_98_Abs_with_clinical_features_v3.csv',
    exclude_hiv=True,
    include_all_vaccines=True 
):
    df = pd.read_csv(path)
    
    # Filter for Pfizer/Moderna if the flag is False
    if not include_all_vaccines:
        valid_vax = ['Pfizer', 'Moderna']
        df = df[df['V0_Vaccine'].isin(valid_vax) | df['B0_Vaccine'].isin(valid_vax)]
        
    group = 'B0_Group'
    which = group.split('_')[0]
        
    post = df[df[group] == f'Post {which}']
    
    # 1. Dynamically identify antibody columns
    ab_start_col = 'A.calcoaceticus LysM'
    ab_start_idx = list(post.columns).index(ab_start_col)
    ab_cols = list(post.columns[ab_start_idx:])
    
    if exclude_hiv:
        ab_cols = [c for c in ab_cols if 'HIV' not in c]
        
    # 2. Explicitly define metadata features to use (Vaccine columns removed from here)
    cat_col_names = ['Cohort', 'Age_Group', 'Age_Group_1', 'Sex', 'Race']
    real_col_names = ['Age']
    
    # 3. Build x_cols deterministically
    x_cols = cat_col_names + real_col_names + ab_cols
    
    data = post[x_cols + ['Vaccine_Response_class', 'Vaccine_Response_raw', 'PID']].dropna()
    X = data[x_cols].values
    
    # 4. Calculate dynamic indices based on list lengths
    n_cat = len(cat_col_names)
    n_real = len(real_col_names)
    abs_start_idx = n_cat + n_real
    
    if cutoffs is not None:
        thresholds = []
        for c in ab_cols:
            thresholds.append(cutoffs[c])
        thresholds = np.array(thresholds)

        I = (X[:, abs_start_idx:] >= thresholds).astype(float)
        h = np.maximum(0.0, X[:, abs_start_idx:] - thresholds)
        X_aug = np.concatenate([X, I, h], axis=1)
        X = X_aug
    
    if binary:
        y = (data['Vaccine_Response_class'] == 'High').astype(int).values
    else:
        y = data['Vaccine_Response_raw'].values
    
    cat_cols_idx = list(range(n_cat))
    
    X_cat = X[:, cat_cols_idx]
    X_real = np.delete(X, cat_cols_idx, axis=1)

    encoder = OrdinalEncoder()
    X_cat_encoded = encoder.fit_transform(X_cat)
    
    X_processed = np.hstack([X_cat, X_cat_encoded, X_real])
    
    emb_list = list(X_processed[:, n_cat : n_cat*2].max(0).astype(int) + 1)
    real_size = X_processed.shape[1] - (n_cat * 2)
    
    if not include_pid:
        return X_processed, y, emb_list, real_size, encoder
    else:
        return X_processed, y, data['PID'].values, emb_list, real_size, encoder