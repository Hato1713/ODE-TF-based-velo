#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch.nn as nn
import torch.utils
import torch.optim


# In[4]:


class WX_MLP_Decoder(nn.Module):
    """
    Decode the influence on target gene expression (WX: (1, )) by all associated TFs (X: tensor(nTF, ))
    """
    def __init__(self, input_dim, hidden_dims = [32, 32], output_dim = 1):
        super.__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        self.hidden_layer = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                nn.Linear(hidden_dims[i], hidden_dims[i+1])
            )

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        self.activation = nn.Sigmoid()

    def to(self, device):
        super().to(device)
        return self
        
    def forward(self, X):
        X = self.input_layer(X)
        importance_weights = self.input_layer.weight.detach()
        
        for layer in self.hidden_layers:
            X = self.activation(X)
            X = layer(X)
            
        X = output_layer(X)

        return X

    # def get_tf_importance(self):
    #     weights = self.input_layer.weight.detach()
    #     importance = torch.norm(weights, dim=0)
    #     return importance.cpu().numpy()


# In[ ]:




