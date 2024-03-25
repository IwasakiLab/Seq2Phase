import torch
from torch import nn, optim
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class EarlyStopping:

    def __init__(self, patience=5, verbose=False, path='checkpoint_model.pth'):
        self.patience = patience    
        self.verbose = verbose      
        self.counter = 0            
        self.best_score = None      
        self.early_stop = False     
        self.val_loss_min = np.Inf   
        self.path = path             

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:  
            self.best_score = score   
            self.checkpoint(val_loss, model)  
        elif score <= self.best_score:  
            self.counter += 1   
            if self.verbose:  
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')  
            if self.counter >= self.patience:  
                self.early_stop = True
        else:  
            self.best_score = score  
            self.checkpoint(val_loss, model)  
            self.counter = 0  

    def checkpoint(self, val_loss, model):
        if self.verbose:  
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)  
        self.val_loss_min = val_loss  
        
def training_loop(n_epochs, optimizer, model, loss, mask_train, x_train,  y_train):
    loss=loss
    
    n_samples=x_train.shape[0]
    n_val=int(n_samples*0.2)

    shuffled_ind=torch.randperm(n_samples)

    train_ind=shuffled_ind[:-n_val] 
    val_ind=shuffled_ind[-n_val:]
    
    x_val=x_train[val_ind]
    y_val=y_train[val_ind]
    x_train=x_train[train_ind]
    y_train=y_train[train_ind]
    
    x_train=x_train
    y_train=y_train
    
    x_val=x_val
    y_val=y_val

    patience=10
    earlystopping = EarlyStopping(patience=patience, verbose=False)
    for epoch in range(1, n_epochs+1):
        model.train()
        
        y_train_pred=model.forward(x_train)
        loss_train=loss(y_train_pred, y_train)
        
        model.eval()
        with torch.no_grad():
            y_val_pred=model.forward(x_val)
            loss_val=loss(y_val_pred, y_val)

        earlystopping(loss_val, model) 
        if earlystopping.early_stop: 
            break
            
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        
class FNN2(nn.Module):
    def __init__(self, embeddings_dim=1024, dropout=0.25):
        super(FNN2, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(32,2)
        )


    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        o = self.linear(x)  
        return o


class NN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_epochs=500, lr=0.03):
        self.n_epochs = n_epochs
        self.lr = lr
        self.model = None
        self.optim = None
        self.loss = nn.CrossEntropyLoss()

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        X_tensor = torch.tensor(X, dtype=torch.float)
        y_tensor = torch.tensor(y, dtype=torch.long)
        n_dim = X.shape[1]
        self.model=FNN2(embeddings_dim=n_dim)
        self.optim = optim.Adam(self.model.parameters(), lr=self.lr)
        training_loop(
            n_epochs=self.n_epochs,
            optimizer=self.optim,
            model=self.model,
            loss=self.loss,
            mask_train=None,
            x_train=X_tensor,
            y_train=y_tensor,
        )
        return self

    def predict(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float)
            self.model.eval()
            y_pred = self.model(X_tensor)
            _, predicted = torch.max(y_pred, 1)
            return predicted.numpy()

    def predict_proba(self, X):
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float)
            self.model.eval()
            y_pred = self.model(X_tensor)
            probas = nn.Softmax(dim=1)(y_pred)
            return probas.numpy()