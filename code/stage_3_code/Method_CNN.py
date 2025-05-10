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
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy 
from sklearn.metrics import precision_score, recall_score, f1_score

class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 50 # CNNs can be slower; adjust as needed
    learning_rate = 1e-3
    
    # CNN specific parameters, to be set during __init__
    input_channels: int
    num_classes: int
    image_h: int
    image_w: int
    
    train_losses: list
    # Let's rename test_accs to train_acc_epochs for clarity if it logs training accuracy per epoch
    train_acc_epochs: list 

    def __init__(self, mName: str, mDescription: str, 
                 input_channels: int, num_classes: int, image_size: tuple, # (height, width)
                 optimizer_cls=torch.optim.Adam, loss_fn=None):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.mName = mName
        self.mDescription = mDescription
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.image_h, self.image_w = image_size

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = {"lr": self.learning_rate}
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

        self.train_losses = []
        self.train_acc_epochs = []
        
        # Example CNN Architecture: Adjust based on your dataset (MNIST, CIFAR, ORL)

        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves H, W

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Halves H, W again

        # Calculate the flattened size after conv and pool layers dynamically
        conv_output_h = self.image_h // 4
        conv_output_w = self.image_w // 4
        # Handle cases where image dimensions are not perfectly divisible by 4
        if conv_output_h == 0 or conv_output_w == 0:
            raise ValueError(f"Image dimensions ({self.image_h}x{self.image_w}) too small for 2 pooling layers.")
        self.flattened_size = 32 * conv_output_h * conv_output_w
        
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, self.num_classes)
        # CrossEntropyLoss combines LogSoftmax and NLLLoss, so no final activation here.

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(f"[INFO] Model moved to device: {self.device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x expected as (N, C, H, W)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, self.flattened_size) # Flatten
        x = self.relu3(self.fc1(x))
        x = self.fc2(x) # Output logits
        return x

    def train_model(self, X_tensor: torch.Tensor, y_tensor: torch.LongTensor): # Renamed from 'train' to avoid conflict
        optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)
        loss_function = self.loss_fn
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        for epoch in range(self.max_epoch):
            self.train() # PyTorch nn.Module method to set training mode
            
            y_pred_logits = self.forward(X_tensor)
            train_loss = loss_function(y_pred_logits, y_tensor)
            self.train_losses.append(train_loss.item())
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == self.max_epoch -1: # Log less frequently, but ensure last epoch is logged
                self.eval() # Set model to evaluation mode for accuracy calculation
                with torch.no_grad():
                    y_pred_labels = y_pred_logits.max(1)[1]
                
                accuracy_evaluator.data = {
                    'true_y': y_tensor.cpu().numpy(),
                    'pred_y': y_pred_labels.cpu().numpy()
                }
                eval_dict = accuracy_evaluator.evaluate()
                accuracy = eval_dict['accuracy']
                self.train_acc_epochs.append(accuracy)
                
                y_true_np = y_tensor.cpu().numpy()
                y_pred_np = y_pred_labels.cpu().numpy()
                
                precision = precision_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
                recall = recall_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
                f1 = f1_score(y_true_np, y_pred_np, average='weighted', zero_division=0)
                print(f'Epoch: {epoch}, Train Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Loss: {train_loss.item():.4f}')
            self.train() # Set back to training mode if not last iteration

    def test_model(self, X_tensor: torch.Tensor) -> np.ndarray: # Renamed from 'test'
        self.eval() # PyTorch nn.Module method to set evaluation mode
        with torch.no_grad():
            y_pred_logits = self.forward(X_tensor)
        y_pred_labels = y_pred_logits.max(1)[1]
        return y_pred_labels.cpu().numpy()

    def run(self):
        print(f'{self.method_name} method running...')
        
        train_X_data = self.data['train']['X'] # Expected to be (N, C, H, W) numpy array from Dataset_Loader
        train_y_data = self.data['train']['y'] # Expected to be (N,) numpy array
        test_X_data = self.data['test']['X']   # Expected to be (N, C, H, W) numpy array
        test_y_true_data = self.data['test']['y'] # Expected to be (N,) numpy array

        # Convert numpy arrays to PyTorch tensors
        # Dataset_Loader should provide X data in (N, C, H, W) format.
        if not isinstance(train_X_data, np.ndarray) or train_X_data.ndim != 4:
            raise ValueError(f"train_X_data must be a 4D numpy array (N, C, H, W), got {type(train_X_data)} with shape {getattr(train_X_data, 'shape', 'N/A')}")
        if not isinstance(test_X_data, np.ndarray) or test_X_data.ndim != 4:
            raise ValueError(f"test_X_data must be a 4D numpy array (N, C, H, W), got {type(test_X_data)} with shape {getattr(test_X_data, 'shape', 'N/A')}")

        train_X_tensor = torch.FloatTensor(train_X_data).to(self.device)
        train_y_tensor = torch.LongTensor(train_y_data.values if isinstance(train_y_data, pd.Series) else train_y_data).to(self.device)
        test_X_tensor = torch.FloatTensor(test_X_data).to(self.device)

        print('--start training...')
        self.train_model(train_X_tensor, train_y_tensor)
        
        print('--start testing...')
        pred_y_np = self.test_model(test_X_tensor)
        
        true_y_np = test_y_true_data.values if isinstance(test_y_true_data, pd.Series) else test_y_true_data
        return {'pred_y': pred_y_np, 'true_y': true_y_np}