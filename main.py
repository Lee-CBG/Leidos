import pandas as pd
from data import *
from setup import ml
import pickle


GLOBAL_CUTOFFS = {}
for a, b in pd.read_csv('cutoffs.csv').values:
    GLOBAL_CUTOFFS[a] = b


cutoff_setting = {
    1:{'qcut':True, 'cutoffs':None, 'setting': 'qcut'}, # not used
    2:{'qcut':False, 'cutoffs':GLOBAL_CUTOFFS, 'setting': 'binary'},
    3:{'qcut':False, 'cutoffs':None, 'setting': 'no-cutoff'}
}


results_6_excel = {}
results_8_excel = {}

COHORTS = ['Autoimmune: Other', 'Cancer: Other', 'HIV', 'Healthy Control',
           'IBD', 'Multiple Myeloma', 'Transplant']


how_many_seeds = 100


for seed in range(0, how_many_seeds):
    for task in [6, 8]:
        binary = True
        for clf in ['mlp', 'nc', 'svm']:
            for c in [2, 3]:
                for coeffs in [(4, 1)]:
                    for use_cat in [True, False]:
                        if use_cat:
                            use = 'Demographic'
                        else:
                            use = 'Ab raw'
                        # qcut = cutoff_setting[c]['qcut']
                        cutoffs = cutoff_setting[c]['cutoffs']
                        setting = cutoff_setting[c]['setting'] 
                        
                        if task == 6:
                            ## Post to post, universal model
                            X, y , emb_list, real_size, encoder = load_6(binary, cutoffs=cutoffs)
                            results_6_excel[f'{seed},{task},{setting},{clf},{coeffs},{use}'] = ml(
                                seed,
                                clf,
                                X, y , emb_list, real_size, encoder, dim=2, use_cat=use_cat, binary=binary, use_hmean=True, splits=5, coeffs=coeffs,
                                n_epochs=51, p=0.2, lr=1e-3 
                            )
                        elif task == 8:
                            ## Post to post, cohort-specific model
                            for cohort in COHORTS:
                                X, y , emb_list, real_size, encoder = load_6(binary, cutoffs=cutoffs)
                                idx = X[:, 0] == cohort
                                if (y[idx] == 0).sum() < 5:
                                    continue
                                results_8_excel[f'{seed},{cohort},{task},{setting},{clf},{coeffs},{use}'] = ml(
                                    seed,
                                    clf,
                                    X[idx], y[idx], emb_list, real_size, encoder, dim=2, use_cat=use_cat, binary=binary, use_hmean=True, splits=5, coeffs=coeffs,
                                    n_epochs=51, p=0.2, lr=1e-3                  
                                )




with open('final_models.pkl', 'wb') as f:
    pickle.dump({'dict6': results_6_excel, 'dict8': results_8_excel}, f)

    
