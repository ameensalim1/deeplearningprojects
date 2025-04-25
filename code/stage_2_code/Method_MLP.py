'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import torch
from torch import nn
import numpy as np
import pandas as pd
from code.base_class.method import method
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from sklearn.metrics import precision_score, recall_score, f1_score

class Method_MLP(method, nn.Module):
    data = None
    # Defines the number of epochs for training the model
    max_epoch = 500
    # Learning rate for the optimizer
    learning_rate = 1e-3
    # Add attributes for dynamic layer sizes
    n_features = None
    n_classes = None

    train_losses: list
    test_accs: list

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    # Modified __init__ to accept n_features and n_classes
    def __init__(self, mName, mDescription, n_features, n_classes):

    def __init__(
        self,
        mName,
        mDescription,
        n_features,
        n_classes,
        hidden_dims=(784, 256, 128),
        activation=nn.ReLU,
        dropout=0.3,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs=None,
        loss_fn=None
    ):     
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # Store dimensions
        # self.train_losses = []
        # self.test_accs = []
        # self.n_features = n_features
        # self.n_classes = n_classes
        # Define layers dynamically
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # self.fc_layer_1 = nn.Linear(self.n_features, self.n_features * 2) # Example: Hidden layer size = 2 * features
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        # self.activation_func_1 = nn.ReLU()
        # self.fc_layer_2 = nn.Linear(self.n_features * 2, self.n_classes) # Output layer size = n_classes
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        # Using LogSoftmax + NLLLoss is often more numerically stable than Softmax + CrossEntropyLoss
        # self.activation_func_2 = nn.LogSoftmax(dim=1)

        self.n_features      = n_features
        self.n_classes       = n_classes
        self.hidden_dims     = hidden_dims
        self.activation_cls  = activation
        self.dropout_prob    = dropout
        self.optimizer_cls   = optimizer_cls
        self.optimizer_kwargs= optimizer_kwargs or {"lr": self.learning_rate}
        self.loss_fn         = loss_fn or nn.CrossEntropyLoss()

        # history
        self.train_losses = []
        self.test_accs    = []

        # Build layers from hidden_dims list
        layers = []
        prev = in_dim = n_features
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                activation(),
                nn.Dropout(dropout)
            ]
            prev = h
        layers += [nn.Linear(prev, n_classes)]  # last linear → logits
        self.net = nn.Sequential(*layers)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        # outout layer result
        # self.fc_layer_2(h) will be a nx2 tensor
        # n (denotes the input instance number): 0th dimension; 2 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        y_pred = self.activation_func_2(self.fc_layer_2(h))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)
        loss_function = self.loss_fn
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # Convert pandas Series/DataFrame to numpy array before tensor conversion
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.LongTensor(y.values)


        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            # Use the already converted tensors
            y_pred = self.forward(X_tensor)
            # convert y to torch.tensor as well
            # Use the already converted tensor
            y_true = y_tensor
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)
            # record training loss
            self.train_losses.append(train_loss.item())
            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()

            if epoch%100 == 0:
                # accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                # accuracy = accuracy_evaluator.evaluate()
                accuracy_evaluator.data = {
                    'true_y': y_true,
                    'pred_y': y_pred.max(1)[1].numpy()
                }
                eval_dict = accuracy_evaluator.evaluate()
                accuracy = eval_dict['accuracy']
                self.test_accs.append(accuracy)
                # Calculate other metrics
                # Convert tensors to numpy for sklearn metrics
                y_true_np = y_true.numpy()
                y_pred_np = y_pred.max(1)[1].numpy()
                # Use average='weighted' for multi-class scenarios, handles label imbalance
                # Set zero_division=0 to return 0 for metrics when denominator is 0
                precision = precision_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
                recall = recall_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
                f1 = f1_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
                print(f'Epoch: {epoch}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Loss: {train_loss.item():.4f}')

    def test(self, X):
        # do the testing, and result the result
        # Convert pandas DataFrame to numpy array before tensor conversion
        X_tensor = torch.FloatTensor(X.values)
        y_pred = self.forward(X_tensor)
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        # Ensure true_y is also numpy array for consistency
        true_y = self.data['test']['y'].values
        return {'pred_y': pred_y, 'true_y': true_y}