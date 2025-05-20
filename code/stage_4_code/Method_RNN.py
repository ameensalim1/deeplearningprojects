'''
Concrete MethodModule class for an RNN-based text classifier.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import torch
from torch import nn
import numpy as np
from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy # Assuming this is compatible or will be made so
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

class Method_RNN(method, nn.Module):
    data = None 
    
    # RNN Specific Hyperparameters (will be passed in __init__)
    # vocab_size: int
    # embedding_dim: int
    # hidden_dim: int
    # num_classes: int
    # rnn_type: str # 'RNN', 'LSTM', 'GRU'
    # num_rnn_layers: int
    # bidirectional: bool
    # dropout_prob_embed: float
    # dropout_prob_rnn: float
    # dropout_prob_fc: float
    # pad_idx: int

    # Training Hyperparameters (can be part of kwargs or set as defaults)
    max_epoch = 100 # Reduced for quicker runs, adjust as needed
    learning_rate = 1e-3
    
    # Performance tracking
    train_losses: list
    epoch_accuracies: list # Renamed from test_accs for clarity during training epochs

    def __init__(
        self,
        mName,
        mDescription,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int, # Should be 2 for sentiment (pos/neg)
        rnn_type: str = 'LSTM',
        num_rnn_layers: int = 1,
        bidirectional: bool = True,
        dropout_prob_embed: float = 0.5,
        dropout_prob_rnn: float = 0.5, # Only applied if num_rnn_layers > 1
        dropout_prob_fc: float = 0.5,
        pad_idx: int = 0, # Default PAD index, should match Dataset_Loader
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=None,
        loss_fn=None
    ):     
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        object.__setattr__(self, 'mName', mName)
        object.__setattr__(self, 'mDescription', mDescription)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.rnn_type = rnn_type.upper()
        self.num_rnn_layers = num_rnn_layers
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx

        # --- Layers ---
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout_prob_embed)

        rnn_dropout_val = dropout_prob_rnn if num_rnn_layers > 1 else 0
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_rnn_layers,
                               bidirectional=bidirectional, dropout=rnn_dropout_val, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_rnn_layers,
                              bidirectional=bidirectional, dropout=rnn_dropout_val, batch_first=True)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_rnn_layers,
                              bidirectional=bidirectional, dropout=rnn_dropout_val, batch_first=True, nonlinearity='tanh')
        else:
            raise ValueError("Unsupported RNN type. Choose from 'RNN', 'LSTM', 'GRU'.")

        self.fc_dropout = nn.Dropout(dropout_prob_fc)
        
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)
        
        # --- Optimizer and Loss ---
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {"lr": self.learning_rate}
        # For binary classification with logits, CrossEntropyLoss is suitable (expects raw scores)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss() 

        # --- History ---
        self.train_losses = []
        self.epoch_accuracies = []


    def forward(self, text_indices_batch):
        # text_indices_batch: (batch_size, seq_len)
        
        embedded = self.embedding(text_indices_batch)  # (batch_size, seq_len, embedding_dim)
        embedded = self.embed_dropout(embedded)
        
        rnn_output, hidden = self.rnn(embedded)
        # rnn_output: (batch_size, seq_len, num_directions * hidden_dim)
        # hidden (for LSTM): (h_n, c_n) where h_n is (num_layers * num_directions, batch_size, hidden_dim)
        # hidden (for GRU/RNN): h_n of shape (num_layers * num_directions, batch_size, hidden_dim)

        if self.rnn_type == 'LSTM':
            if self.bidirectional:
                final_hidden = torch.cat((hidden[0][-2,:,:], hidden[0][-1,:,:]), dim=1)
            else:
                final_hidden = hidden[0][-1,:,:]
        else: # GRU or RNN
            if self.bidirectional:
                final_hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            else:
                final_hidden = hidden[-1,:,:]
        
        dropped_out = self.fc_dropout(final_hidden)
        logits = self.fc(dropped_out) # (batch_size, num_classes)
        return logits

    def train_model(self, X_train: torch.Tensor, y_train: torch.Tensor, batch_size: int = 64): # Changed method name from 'train' to avoid conflict
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)

        optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)
        loss_fn = self.loss_fn
        evaluator = Evaluate_Accuracy('training evaluator','')

        for epoch in range(self.max_epoch):
            self.train()
            running_loss = 0.0

            for Xb, yb in train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)

                optimizer.zero_grad()
                logits = self.forward(Xb)
                loss   = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * Xb.size(0)

            avg_loss = running_loss / len(train_loader.dataset)
            self.train_losses.append(avg_loss)

            # log & evaluate every 10 epochs
            if epoch % 10 == 0 or epoch == self.max_epoch - 1:
                self.eval()
                with torch.no_grad():
                    all_logits = self.forward(X_train)
                    preds = all_logits.argmax(1).cpu().numpy()
                    trues = y_train.cpu().numpy()
                evaluator.data = {'true_y': trues, 'pred_y': preds}
                metrics = evaluator.evaluate()
                self.epoch_accuracies.append(metrics['accuracy'])

                print(f"Epoch {epoch:>2}  loss={avg_loss:.4f}  "
                      f"acc={metrics['accuracy']:.4f}")

    def test_model(self, X_test): 
        self.eval() 
        with torch.no_grad():
            y_pred_logits = self.forward(X_test)
            y_pred_proba = torch.softmax(y_pred_logits, dim=1)
            y_pred_labels = y_pred_proba.max(1)[1]
        return y_pred_labels 

    def run(self):
        print('method running...')
        print('--start training...')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device) 
        self.device = device
        print(f"Using device: {device}")

        X_train = torch.LongTensor(self.data['train']['X']).to(self.device)
        y_train = torch.LongTensor(self.data['train']['y']).to(self.device)

        self.train_model(X_train, y_train, batch_size=64)

        
        print('--start testing...')
        pred_y_np = np.array([])
        true_y_np = np.array([])

        if 'test' in self.data and self.data['test']['X'] is not None and self.data['test']['X'].size > 0:
            X_test_np = self.data['test']['X']
            true_y_np = self.data['test']['y']
            X_test_tensor = torch.LongTensor(X_test_np).to(device)
            pred_y_tensor = self.test_model(X_test_tensor) 
            if pred_y_tensor is not None:
                pred_y_np = pred_y_tensor.cpu().numpy() 
            else:
                print("Warning: test_model returned None, pred_y_np will be empty.")
        else:
            print("Warning: No test data or empty test data. Skipping testing.")

        return {'pred_y': pred_y_np, 'true_y': true_y_np}
    