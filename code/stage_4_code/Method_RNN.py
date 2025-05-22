'''
Concrete MethodModule class for an RNN-based text classifier.
'''
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import torch
from torch import nn
import numpy as np
from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import (
    Evaluate_Accuracy,
) # Assuming this is compatible
from torch.utils.data import TensorDataset, DataLoader


class Method_RNN(method, nn.Module):
    data = None
    # device attribute will be set in run()
    device = None

    # RNN Specific Hyperparameters are passed in __init__
    # Training Hyperparameters
    max_epoch = 100
    learning_rate = 1e-3 # Default, can be overridden by optimizer_kwargs

    # Performance tracking
    train_losses: list
    epoch_accuracies: list

    def __init__(
        self,
        mName,
        mDescription,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_classes: int,
        rnn_type: str = "LSTM",
        num_rnn_layers: int = 1,
        bidirectional: bool = True,
        dropout_prob_embed: float = 0.5,
        dropout_prob_rnn: float = 0.5,
        dropout_prob_fc: float = 0.5,
        pad_idx: int = 0,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=None,
        loss_fn=None,
    ):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.mName = mName
        self.mDescription = mDescription

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.rnn_type = rnn_type.upper()
        self.num_rnn_layers = num_rnn_layers
        self.bidirectional = bidirectional
        self.pad_idx = pad_idx

        # --- Layers ---
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_idx
        )
        self.embed_dropout = nn.Dropout(dropout_prob_embed)

        rnn_dropout_val = dropout_prob_rnn if num_rnn_layers > 1 else 0
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                num_layers=num_rnn_layers,
                bidirectional=bidirectional,
                dropout=rnn_dropout_val,
                batch_first=True,
            )
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                embedding_dim,
                hidden_dim,
                num_layers=num_rnn_layers,
                bidirectional=bidirectional,
                dropout=rnn_dropout_val,
                batch_first=True,
            )
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(
                embedding_dim,
                hidden_dim,
                num_layers=num_rnn_layers,
                bidirectional=bidirectional,
                dropout=rnn_dropout_val,
                batch_first=True,
                nonlinearity="tanh",
            )
        else:
            raise ValueError(
                "Unsupported RNN type. Choose from 'RNN', 'LSTM', 'GRU'."
            )

        self.fc_dropout = nn.Dropout(dropout_prob_fc)

        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)

        # --- Optimizer and Loss ---
        self.optimizer_cls = optimizer_cls
        default_lr = {"lr": self.learning_rate}
        if optimizer_kwargs:
            default_lr.update(optimizer_kwargs)
        self.optimizer_kwargs = default_lr
        
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

        # --- History ---
        self.train_losses = []
        self.epoch_accuracies = []

    def forward(self, text_indices_batch):
        embedded = self.embedding(text_indices_batch)
        embedded = self.embed_dropout(embedded)
        rnn_output, hidden = self.rnn(embedded)

        if self.rnn_type == "LSTM":
            h_n = hidden[0]
        else:
            h_n = hidden

        if self.bidirectional:
            final_hidden = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        else:
            final_hidden = h_n[-1, :, :]

        dropped_out = self.fc_dropout(final_hidden)
        logits = self.fc(dropped_out)
        return logits

    def train_model(
        self, X_train: torch.Tensor, y_train: torch.Tensor, batch_size: int = 64
    ):
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False
        )

        optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)
        loss_fn = self.loss_fn
        evaluator = Evaluate_Accuracy("training evaluator", "")

        for epoch in range(self.max_epoch):
            self.train()
            running_loss = 0.0
            num_samples_epoch = 0

            for Xb, yb in train_loader:
                optimizer.zero_grad()
                logits = self.forward(Xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * Xb.size(0)
                num_samples_epoch += Xb.size(0)

            avg_loss = running_loss / num_samples_epoch if num_samples_epoch > 0 else 0
            self.train_losses.append(avg_loss)

            if epoch % 10 == 0 or epoch == self.max_epoch - 1:
                self.eval()
                all_preds_list = []
                all_trues_list = []
                
                eval_ds_epoch = TensorDataset(X_train, y_train)
                eval_loader_epoch = DataLoader(eval_ds_epoch, batch_size=batch_size, shuffle=False)

                with torch.no_grad():
                    for X_eval_b, y_eval_b in eval_loader_epoch:
                        logits_b = self.forward(X_eval_b)
                        preds_b = logits_b.argmax(dim=1)
                        all_preds_list.append(preds_b.cpu())
                        all_trues_list.append(y_eval_b.cpu())
                
                if not all_preds_list:
                    print(f"Epoch {epoch+1:>2}/{self.max_epoch}  loss={avg_loss:.4f}  acc=N/A (no eval data processed)")
                    self.epoch_accuracies.append(0.0)
                    continue

                all_preds_np = torch.cat(all_preds_list).numpy()
                all_trues_np = torch.cat(all_trues_list).numpy()
                
                evaluator.data = {"true_y": all_trues_np, "pred_y": all_preds_np}
                metrics = evaluator.evaluate()
                self.epoch_accuracies.append(metrics["accuracy"])

                print(
                    f"Epoch {epoch+1:>2}/{self.max_epoch}  loss={avg_loss:.4f}  "
                    f"acc={metrics['accuracy']:.4f}"
                )
            else:
                 print(f"Epoch {epoch+1:>2}/{self.max_epoch}  loss={avg_loss:.4f}")


    def test_model(self, X_test: torch.Tensor, batch_size: int = 64):
        self.eval()
        all_pred_labels_list = []

        if X_test is None or X_test.size(0) == 0:
            return torch.empty(0, dtype=torch.long)

        test_ds = TensorDataset(X_test)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for (Xb,) in test_loader:
                logits_b = self.forward(Xb)
                pred_labels_b = logits_b.argmax(dim=1)
                all_pred_labels_list.append(pred_labels_b.cpu())
        
        if not all_pred_labels_list:
            return torch.empty(0, dtype=torch.long)

        all_pred_labels = torch.cat(all_pred_labels_list)
        return all_pred_labels

    def run(self):
        print('method running...')
        print('--start training...')
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # If MPS not available (device is CPU), then check for CUDA
        if device.type == 'cpu': # Check device type for comparison
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(device) 
        self.device = device # Store the determined device
        print(f"Using device: {self.device}")

        if self.data is None or "train" not in self.data:
            print("Error: Training data not found in self.data.")
            return {"pred_y": np.array([]), "true_y": np.array([])}

        X_train_np = self.data["train"]["X"]
        y_train_np = self.data["train"]["y"]
        # Move data to the determined device
        X_train = torch.LongTensor(X_train_np).to(self.device)
        y_train = torch.LongTensor(y_train_np).to(self.device)

        train_batch_size = 64 
        self.train_model(X_train, y_train, batch_size=train_batch_size)
        print("--training complete--")
        
        print("--start testing...")
        pred_y_np = np.array([])
        true_y_np = np.array([])

        if "test" in self.data and self.data["test"]["X"] is not None and self.data["test"]["X"].size > 0:
            X_test_np = self.data["test"]["X"]
            true_y_np = self.data["test"]["y"]
            X_test_tensor = torch.LongTensor(X_test_np).to(self.device) # Move to device
            
            test_batch_size = 64 
            pred_y_tensor = self.test_model(X_test_tensor, batch_size=test_batch_size)
            
            if pred_y_tensor is not None and pred_y_tensor.numel() > 0 :
                pred_y_np = pred_y_tensor.cpu().numpy()
            else:
                print("Warning: test_model returned None or empty tensor, pred_y_np will be empty.")
                if isinstance(true_y_np, (list, np.ndarray)) and len(true_y_np) > 0: # Check if true_y_np is not empty
                     pred_y_np = np.array([-1] * len(true_y_np)) 
        else:
            print("Warning: No test data or empty test data. Skipping testing.")

        if not isinstance(true_y_np, np.ndarray):
            true_y_np = np.array(true_y_np)

        return {"pred_y": pred_y_np, "true_y": true_y_np}
