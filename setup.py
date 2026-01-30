from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from bnMLP import BottleneckMLP
import copy
from utils import *
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.metrics import roc_curve, auc, RocCurveDisplay, precision_recall_curve, average_precision_score


def ml(SEED, clf, X, y, emb_list, real_size, encoder, dim, use_cat, splits=10, binary=True, use_hmean=True, coeffs=(1, 1), p=0.2, n_epochs=300, lr=1e-2):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED)
    best_models = []
    metrics = []
    for_stratify = y if binary else (y>3.5).astype(int)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, for_stratify), start=1):
        #################################################### Dataset
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        if use_cat:
            start_from = len(emb_list)
        else:
            start_from = 11
        
        
        size = real_size if use_cat else real_size - 1
        X_train_tensor = torch.from_numpy(X_train[:, start_from:].astype(np.float32))
        y_train_tensor = torch.from_numpy(y_train).float()
        
        X_test_tensor = torch.from_numpy(X_test[:, start_from:].astype(np.float32))
        y_test_tensor = torch.from_numpy(y_test).float()
        
        
        # stratified split using sklearn
        train_idx, val_idx = train_test_split(
            np.arange(len(X_train_tensor)),
            test_size=0.2,
            stratify=y_train_tensor.numpy() if binary else (y_train_tensor.numpy()>3.5).astype(int),
            random_state=SEED
        )
        
        # create datasets
        train_dataset = TensorDataset(X_train_tensor[train_idx], y_train_tensor[train_idx])
        val_dataset   = TensorDataset(X_train_tensor[val_idx],  y_train_tensor[val_idx])
        
        # data loaders
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(SEED))
        val_loader   = DataLoader(val_dataset,   batch_size=len(val_dataset), shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor),
                                 batch_size=len(X_test_tensor), shuffle=False)
        #################################################### Model
        set_seed(SEED)
        
        m = BottleneckMLP(emb_list, size, start_from, 8, (128, 64, dim), p=p, use_cat=use_cat, binary=binary).to(device)
        
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0)
        lf = nn.BCELoss() if binary else nn.MSELoss()
        
        best = -1 if binary else 1e50
        best_a = -1
        best_b = -1
        best_state_dict = None
        ######################################### Train for some epochs ###########################################
        for epoch in range(n_epochs):
            epoch_loss = 0
            for x_batch, y_batch in train_loader:
                m.train()
                opt.zero_grad()
                y_hat, two_d = m(x_batch.to(device))
                loss = lf(y_hat.reshape(-1), y_batch.to(device).reshape(-1))
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            if epoch % 50 == 0:
                print('Fold:', fold)
                print('Epoch:', epoch+1)
                print('Train Loss:', epoch_loss / len(train_loader))
            with torch.no_grad():
                ######################################### validation after each epoch ###########################################
                m.eval()
                for x_batch, y_batch in val_loader:
                    
                    if clf == 'mlp':
                        logits, two_d = m(x_batch.to(device))
                    else:
                        with torch.no_grad():
                            m.eval()
                            x_for_clf = []
                            y_for_clf = []
                            #################### use train set for training a classifier 
                            #################### that will be evaluated on the val set
                            for x_batch_train, y_batch_train in train_loader:
                                logits, two_d = m(x_batch_train.to(device))
                                x_for_clf.extend(two_d)
                                y_for_clf.extend(y_batch_train)
                        x_for_clf = torch.stack(x_for_clf).cpu().numpy()
                        y_for_clf = torch.stack(y_for_clf).cpu().numpy()
                        if clf == 'svm':
                            one = (y_for_clf == 1).astype(float).sum()
                            zero = (y_for_clf == 0).astype(float).sum()
                            w = (one/zero)
                            delta = 1
                
                            classifier = SVC(kernel="rbf", class_weight={0: w, 1: 1}, probability=True)
                            ## fit on train set
                            classifier.fit(x_for_clf, y_for_clf)

                            ##### x_batch is val set
                            x_in = m(x_batch.to(device))[-1].cpu().numpy()
                            logits = classifier.decision_function(x_in)
                        elif clf == 'nc':
                            classifier = NearestCentroid(metric="euclidean")
                            ## fit on train set
                            classifier.fit(x_for_clf, y_for_clf)
                            ##### x_batch is val set
                            x_in = m(x_batch.to(device))[-1].cpu().numpy()
                            logits = classifier.predict(x_in)
                            
                    if binary:
                        if clf == 'mlp':
                            y_pred = (logits.reshape(-1) > 0.5).int().cpu()
                        elif clf == 'svm':
                            y_pred = torch.from_numpy(logits >= delta).int()
                        elif clf == 'nc':
                            y_pred = torch.from_numpy(logits).int()
                        a_val = (y_pred[y_batch==0] == y_batch[y_batch==0]).float().mean().item() + 1e-20
                        b_val = (y_pred[y_batch==1] == y_batch[y_batch==1]).float().mean().item() + 1e-20
                        ####################################### This is weighting zeros higher
                        
                        c_val = sum(coeffs) / ((coeffs[0]/a_val) + coeffs[1]/b_val)
                        
                        try:
                            if clf == 'svm':
                                logits = torch.sigmoid(torch.from_numpy(logits))
                            elif clf == 'nc':
                                D = pairwise_distances(x_in, classifier.centroids_, metric=classifier.metric)
                                pos_idx = np.where(classifier.classes_ == 1)[0][0]
                                logits = torch.from_numpy(-D[:, pos_idx])
                            auc_val = roc_auc_score(y_batch.cpu().numpy(), logits.reshape(-1).cpu().numpy())
                            sen_val = []
                            spec_val = []
                            for t in [0.75, 0.8, 0.85, 0.9, 0.95]:
                                sen_, spec_ = at_95(logits.cpu().numpy(), y_batch, t=t)
                                sen_val.append(sen_)
                                spec_val.append(spec_)
                        except:
                            pass
                    else:
                        if clf != 'mlp':
                            raise NotImplementedError
                        y_labeled = (y_batch > 3.5).int()
                        zero = y_labeled == 0
                        a_val = lf(logits[zero], y_batch[zero]) ** 0.5 
                        b_val = lf(logits[~zero], y_batch[~zero]) ** 0.5
                        c_val = 2 / ((1/a_val) + 1/b_val)

                    if use_hmean:
                       condition = (binary and c_val > best) or ((not binary) and c_val < best) 
                    else:
                        condition = (binary and spec_ > best) or ((not binary) and c_val < best)
                    if condition:
                        ######################################### Record test scores at best validation epoch ###########################################
                        
                        if binary:
                            if use_hmean:
                                best = c_val
                                best_a = a_val
                                best_b = b_val
                            else:
                                best = spec_val
                        else:
                            best = c_val
                        
                        best_state_dict = copy.deepcopy(m.state_dict())
                        m.eval()
                        for x_batch, y_batch in test_loader:
                            if clf == 'mlp':
                                
                                logits, two_d = m(x_batch.to(device))
                            elif clf == 'svm':
                                x_in = m(x_batch.to(device))[-1].cpu().numpy()
                                logits = classifier.decision_function(x_in)
                            elif clf == 'nc':
                                x_in = m(x_batch.to(device))[-1].cpu().numpy()
                                logits = classifier.predict(x_in)
                                
                            if binary:
                                if clf == 'mlp':
                                    y_pred = (logits.reshape(-1) > 0.5).int().cpu()
                                elif clf == 'svm':
                                    y_pred = torch.from_numpy(logits >= delta).int()
                                elif clf == 'nc':
                                    
                                    y_pred = torch.from_numpy(logits).int()
                                a_test = (y_pred[y_batch==0] == y_batch[y_batch==0]).float().mean().item() + 1e-20
                                b_test = (y_pred[y_batch==1] == y_batch[y_batch==1]).float().mean().item() + 1e-20
                                c_test = 2 / ((1/a_test) + 1/b_test)
                                if clf == 'svm':
                                    logits = torch.sigmoid(torch.from_numpy(logits))
                                elif clf == 'nc':
                                    D = pairwise_distances(x_in, classifier.centroids_, metric=classifier.metric)
                                    pos_idx = np.where(classifier.classes_ == 1)[0][0]
                                    logits = torch.from_numpy(-D[:, pos_idx])
                                    # logits = torch.sigmoid(torch.from_numpy(logits))
                                
                                auc_test = roc_auc_score(y_batch.cpu().numpy(), logits.cpu().numpy())
                                sen_test = []
                                spec_test = []
                                for t in [0.75, 0.8, 0.85, 0.9, 0.95]:
                                    sen_, spec_ = at_95(logits.cpu().numpy(), y_batch, t=t)
                                    sen_test.append(sen_)
                                    spec_test.append(spec_)
                                
                            else:
                                if clf != 'mlp':
                                    raise NotImplementedError
                                y_labeled = (y_batch > 3.5).int()
                                zero = y_labeled == 0
                                a_test = lf(logits[zero], y_batch[zero]) ** 0.5 
                                b_test = lf(logits[~zero], y_batch[~zero]) ** 0.5
                                c_test = 2 / ((1/a_test) + 1/b_test)
                                auc_test = 0
            if epoch % 50 == 0:
                if not binary:
                    print(f'Fold {fold}, Test Metric: ({a_test:.4f}, {b_test:.4f}, {c_test:.4f}, {auc_test:.4f}), Val Best mean: {best}')
                else:
                    print(f'Val Best mean: ({best:.4f}, {best_a:.4f}, {best_b:.4f})')
                    print(f'Test Metric at best Validation Epoch: ({a_test:.4f}, {b_test:.4f}, {c_test:.4f}, {auc_test:.4f}, {sen_test[0]:.4f}, {spec_test[0]:.4f})')
                    
                print(f'{splits}-fold Average: {torch.tensor(metrics).mean(0)}')
                print('----------------------'*2)
        best_models.append(best_state_dict)
        
        if binary:
            metrics.append([a_test, b_test, c_test, auc_test, *spec_test, *sen_test])
            
        else:
            metrics.append([a_test, b_test, c_test, auc_test])
        print('============================================='*2)

    return best_models, skf, torch.tensor(metrics).mean(0)


