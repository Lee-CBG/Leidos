import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def load_2(
    binary=True, 
    cutoffs=None, 
    include_pid=False, 
    path='Leidos_data_98_Abs_with_clinical_features_v2.csv',
    exclude_hiv=True
):
    df = pd.read_csv(path)
    
    group = 'B0_Group'
    which = group.split('_')[0]
    pre = df[df[group] == f'Pre {which}']
    post = df[df[group] == f'Post {which}']
    
    result = pd.merge(pre, post, on="PID", how="inner")
        
    if exclude_hiv:
            x_cols = ['Cohort_x'] + [i for i in result.columns[7:] if ('_x' in i and 'HIV' not in i)]
    else:
        x_cols = ['Cohort_x'] + [i for i in result.columns[7:] if '_x' in i] 
    
    data = result[x_cols + ['Vaccine_Response_class_y'] + ['Vaccine_Response_raw_y']].dropna()
    X = data[x_cols].values
    
    if cutoffs is not None:
        thresholds = []
        cutoff_cols = list(map(lambda z: z[:-2], x_cols))[6:]
        for c in cutoff_cols:
            thresholds.append(cutoffs[c])
        thresholds = np.array(thresholds)

        I = (X[:, 6:] >= thresholds).astype(float)
        h = np.maximum(0.0, X[:, 6:] - thresholds)
        X_aug = np.concatenate([X, I, h], axis=1)
        X = X_aug
        
    if binary:
        y = (data['Vaccine_Response_class_y'] == 'High').astype(int).values
    else:
        y = data['Vaccine_Response_raw_y'].values
    
    
    cat_cols = [0, 2, 3, 4, 5]
    X_cat = X[:, cat_cols]
    X_real = np.delete(X, cat_cols, axis=1)
    
    encoder = OrdinalEncoder()
    X_cat_encoded = encoder.fit_transform(X_cat)
    
    X_processed = np.hstack([X_cat, X_cat_encoded, X_real])
    

    emb_list = list(X_processed[:, 5:10].max(0).astype(int) + 1)
    real_size = X_processed.shape[1] - 10
    
    if not include_pid:
        return X_processed, y, emb_list, real_size, encoder
    else:
        a = result[['PID'] + x_cols + ['Vaccine_Response_class_y'] + ['Vaccine_Response_raw_y']].dropna()
        return X_processed, y, a['PID'].values, emb_list, real_size, encoder



def load_6(
    binary=True, 
    cutoffs=None, 
    include_pid=False, 
    path='Leidos_data_98_Abs_with_clinical_features_v2.csv',
    exclude_hiv=True
):
    df = pd.read_csv(path)
    group = 'B0_Group'
    which = group.split('_')[0]
        
    post = df[df[group] == f'Post {which}']
    
    data = post[['Cohort'] + list(post.columns[5:])].dropna()
    
    if exclude_hiv:
        x_cols = [i for i in ['Cohort'] + list(data.columns[3:]) if 'HIV' not in i]
    else:
        x_cols = ['Cohort'] + list(data.columns[3:]) 
    
    X = data[x_cols].values
    if cutoffs is not None:
        
        thresholds = []
        cutoff_cols = list(map(lambda z: z, x_cols))[6:]
        
        for c in cutoff_cols:
            thresholds.append(cutoffs[c])
        thresholds = np.array(thresholds)

        I = (X[:, 6:] >= thresholds).astype(float)                # same shape [batch_size, d]
        h = np.maximum(0.0, X[:, 6:] - thresholds)     # same shape
        X_aug = np.concatenate([X, I, h], axis=1)
        X = X_aug
    
    if binary:
        y = (data['Vaccine_Response_class'] == 'High').astype(int).values
    else:
        y = data['Vaccine_Response_raw'].values
    
    cat_cols = [0, 2, 3, 4, 5]
    
    X_cat = X[:, cat_cols]
    X_real = np.delete(X, cat_cols, axis=1)

    encoder = OrdinalEncoder()
    X_cat_encoded = encoder.fit_transform(X_cat)
    
    X_processed = np.hstack([X_cat, X_cat_encoded, X_real])
    
    
    emb_list = list(X_processed[:, 5:10].max(0).astype(int) + 1)
    real_size = X_processed.shape[1] - 10
    

    if not include_pid:
        return X_processed, y, emb_list, real_size, encoder
    else:
        a = post[['PID'] + ['Cohort'] + list(post.columns[5:])].dropna()
        return X_processed, y, a['PID'].values, emb_list, real_size, encoder


