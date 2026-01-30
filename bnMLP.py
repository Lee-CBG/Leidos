import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class BottleneckMLP(nn.Module):
        def __init__(self, num_embeddings_list, input_size, start_from, emb_dim, dims=(32, 16, 2), p=0.5, use_cat=True, binary=True):
            super(BottleneckMLP, self).__init__()
            self.start_from = start_from
            self.use_cat = use_cat
            self.binary = binary
            if use_cat:
                self.embeddings = nn.ModuleList([
                    nn.Embedding(num_embeddings, emb_dim)
                    for num_embeddings in num_embeddings_list
                ])
            
            layers = []

            if use_cat:
                first_dim = input_size + len(num_embeddings_list) * emb_dim
            else:
                first_dim = input_size
            layers.append(nn.BatchNorm1d(first_dim))
            self.drop = nn.Dropout(p)
    
            prev_size = first_dim
            # for i, d in enumerate(dims[:-1]):
            for i, d in enumerate(dims):
                layers.append(nn.Linear(prev_size, d))
                layers.append(nn.BatchNorm1d(d))
                # if i != len(dims) - 2:
                if i != len(dims) - 1:
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(p)) # I indented this recently
                
                prev_size = d
    
            self.encoder = nn.Sequential(*layers)
            self.classifier = nn.Sequential(
                # do not use dropout here
                nn.ReLU(),
                # nn.Linear(dims[-2], dims[-1]),
                nn.BatchNorm1d(dims[-1]),
                # nn.ReLU(),
                # nn.Dropout(p),
                
                nn.Linear(dims[-1], 1),
                # nn.Dropout(p)
            )

        def forward(self, x, return_features=False):
            _x = []
            
            if self.use_cat:
                for i, emb in enumerate(self.embeddings):
                    _x.append(self.drop(emb(x[:, i].long())))
                
                _x = torch.cat([torch.hstack(_x), x[:, self.start_from:]], dim=-1).float()
            
                z = self.encoder(_x)
            else:
                z = self.encoder(x)
            if return_features:
                return z
            if self.binary:
                return torch.sigmoid(self.classifier(z)), z
            return self.classifier(z), z
                


