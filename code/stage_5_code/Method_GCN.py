'''
Concrete MethodModule class for Graph Convolutional Network (GCN)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from code.base_class.method import method


class GraphConvolutionalLayer(nn.Module):
    """
    Single Graph Convolutional Layer implementation following Kipf & Welling (2016)
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionalLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, input, adj):
        # Input shape: (num_nodes, in_features)
        # Adj shape: (num_nodes, num_nodes)
        support = torch.mm(input, self.weight)  # (num_nodes, out_features)
        output = torch.spmm(adj, support)  # (num_nodes, out_features)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    """
    Graph Convolutional Network with two GC layers
    """
    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolutionalLayer(nfeat, nhid)
        self.gc2 = GraphConvolutionalLayer(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class Method_GCN(method):
    """
    GCN method implementation for node classification
    """
    def __init__(self, mName=None, mDescription=None):
        super(Method_GCN, self).__init__(mName, mDescription)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.hidden_dim = 16
        self.dropout = 0.5
        self.learning_rate = 0.01
        self.weight_decay = 5e-4
        self.epochs = 200
        self.patience = 10  # Early stopping patience
        
        # Training tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def create_model(self, nfeat, nclass):
        """Create GCN model"""
        model = GCN(nfeat=nfeat, nhid=self.hidden_dim, nclass=nclass, dropout=self.dropout)
        return model.to(self.device)

    def train_epoch(self, model, optimizer, features, adj, labels, idx_train):
        """Train for one epoch"""
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = self.accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()
        return loss_train.item(), acc_train

    def validate(self, model, features, adj, labels, idx_val):
        """Validate model"""
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = self.accuracy(output[idx_val], labels[idx_val])
        return loss_val.item(), acc_val

    def test(self, model, features, adj, labels, idx_test):
        """Test model"""
        model.eval()
        with torch.no_grad():
            output = model(features, adj)
            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = self.accuracy(output[idx_test], labels[idx_test])
            pred_test = output[idx_test].max(1)[1].type_as(labels)
        return loss_test.item(), acc_test, pred_test

    def accuracy(self, output, labels):
        """Calculate accuracy"""
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def train_model(self, data):
        """Main training loop"""
        # Extract data
        features = data['graph']['X'].to(self.device)
        adj = data['graph']['utility']['A'].to(self.device)
        labels = data['graph']['y'].to(self.device)
        idx_train = data['train_test_val']['idx_train'].to(self.device)
        idx_val = data['train_test_val']['idx_val'].to(self.device)
        idx_test = data['train_test_val']['idx_test'].to(self.device)

        # Create model
        nfeat = features.shape[1]
        nclass = labels.max().item() + 1
        model = self.create_model(nfeat, nclass)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # Training loop
        best_val_acc = 0
        patience_counter = 0
        
        print(f"Training GCN model...")
        print(f"Features: {nfeat}, Classes: {nclass}, Nodes: {features.shape[0]}")
        print(f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")
        
        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(model, optimizer, features, adj, labels, idx_train)
            
            # Validate
            val_loss, val_acc = self.validate(model, features, adj, labels, idx_val)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc.item())
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc.item())
            
            # Print progress
            if epoch % 20 == 0:
                print(f'Epoch {epoch:04d}: train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, '
                      f'val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model for testing
        model.load_state_dict(best_model_state)
        
        # Final test
        test_loss, test_acc, pred_test = self.test(model, features, adj, labels, idx_test)
        
        print(f"Test set results: loss= {test_loss:.4f}, accuracy= {test_acc:.4f}")
        
        return model, test_acc, pred_test, labels[idx_test]

    def save_learning_curves(self, dataset_name, save_path):
        """Save learning curves plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss', color='blue')
        ax1.plot(self.val_losses, label='Val Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{dataset_name} - Training/Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy', color='blue')
        ax2.plot(self.val_accuracies, label='Val Accuracy', color='red')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{dataset_name} - Training/Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Learning curves saved to {save_path}")

    def run(self):
        """Main run method called by the framework"""
        print('GCN method running...')
        print('--start training...')
        
        model, test_acc, pred_test, true_test = self.train_model(self.data)
        
        print('--training completed')
        
        return {
            'pred_y': pred_test.cpu().numpy(),
            'true_y': true_test.cpu().numpy(),
            'test_accuracy': test_acc.item(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'model': model
        } 