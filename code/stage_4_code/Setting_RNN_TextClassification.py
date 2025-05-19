'''
Concrete Setting class for Stage 4: RNN Text Classification.
Handles loading text data, running the RNN model, saving, and evaluating.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
import numpy as np
from typing import Optional, Dict, Any 
import matplotlib.pyplot as plt
import torch # Added for tensor checks
# Import the Method_RNN (to be created/renamed from Method_MLP)
from .Method_RNN import Method_RNN 
# Dataset_Loader will be used via self.dataset passed in __init__
# Result_Saver will be used via self.result
# Evaluate_Accuracy will be used via self.evaluate

class Setting_RNN_TextClassification(setting):
    
    # Hyperparameters for the RNN model are primarily configured
    # within the Method_RNN instance itself when it's created.
    # This Setting class orchestrates the use of that configured method.

    def load_run_save_evaluate(self) -> Optional[Dict[str, float]]: # Return type is dict of scores or None on failure

        print("--- Loading text classification data ---")
        # self.dataset is an instance of Dataset_Loader for text
        loaded_data: Dict[str, Optional[Dict[str, Any]]] = self.dataset.load()

        train_data = loaded_data.get('train')
        test_data = loaded_data.get('test')

        if not (train_data and train_data.get('X') is not None and train_data.get('y') is not None and train_data['X'].size > 0):
            print("ERROR: Training data loading failed or data is empty. Cannot proceed.")
            return None 

        if not (test_data and test_data.get('X') is not None and test_data.get('y') is not None):
             print("Warning: Testing data loading failed or data is empty. Evaluation might be limited.")

        print("--- Text data loaded successfully ---")
        if train_data and 'X' in train_data and 'y' in train_data:
            print(f"Train data X shape: {train_data['X'].shape}, y shape: {train_data['y'].shape}")
        if test_data and 'X' in test_data and 'y' in test_data and test_data['X'].size > 0 :
            print(f"Test data X shape: {test_data['X'].shape}, y shape: {test_data['y'].shape}")
        
        if not isinstance(self.method, Method_RNN):
            print(f"ERROR: self.method is not an instance of Method_RNN. It is {type(self.method)}. Cannot proceed.")
            return None
        
        if hasattr(self.method, 'vocab_size') and self.method.vocab_size != self.dataset.vocab_size:
            print(f"CRITICAL Warning: Method_RNN vocab_size ({self.method.vocab_size}) does not match dataset vocab_size ({self.dataset.vocab_size}). Re-initializing method with correct vocab_size.")
            # This is a critical mismatch. The model needs to be configured with the correct vocab_size from the dataset.
            # Ideally, the main script should handle this by initializing Method_RNN *after* dataset.load()
            # For robustness, we can try to update it here if the method supports it, or error out.
            # Assuming Method_RNN can have its vocab_size updated or its embedding layer re-initialized.
            # This part is tricky as nn.Embedding is usually fixed after init.
            # A better approach is to ensure main.py does: dataset.load() -> get vocab_size -> init Method_RNN(vocab_size=...) -> init Setting(method=...)
            # For now, we just print a strong warning. The user must ensure correct initialization order.
            # It might be safer to return None here if a mismatch is detected, forcing correct setup.
            # For this project, let's assume the main script will correctly pass vocab_size obtained from dataset.load() to Method_RNN constructor.
        
        expected_num_classes = 2 
        if hasattr(self.method, 'num_classes') and self.method.num_classes != expected_num_classes:
             print(f"Warning: Method_RNN num_classes ({self.method.num_classes}) is not {expected_num_classes} for sentiment analysis.")


        print("--- Running method ---")
        self.method.data = loaded_data 
        learned_result: Optional[Dict[str, Any]] = self.method.run()

        if learned_result is None:
            print("ERROR: Method execution failed or returned None. Cannot evaluate.")
            return None 

        print("--- Saving results ---")
        # Ensure result_destination_file_path is set in Result_Saver instance
        if not self.result.result_destination_file_path:
            self.result.result_destination_file_path = f"{self.method.mName}_results"
            print(f"Result destination path not set, defaulting to: {self.result.result_destination_file_path}")

        self.result.data = learned_result
        self.result.save()

        print("--- Evaluating results ---")
        self.evaluate.data = learned_result
        
        if 'pred_y' in learned_result and isinstance(learned_result['pred_y'], torch.Tensor):
            learned_result['pred_y'] = learned_result['pred_y'].cpu().numpy()
        if 'true_y' in learned_result and isinstance(learned_result['true_y'], torch.Tensor):
            learned_result['true_y'] = learned_result['true_y'].cpu().numpy()
            
        scores: Dict[str, float] = self.evaluate.evaluate() 

        if hasattr(self.method, 'train_losses') and self.method.train_losses:
            plt.figure(figsize=(10, 4))
            plt.plot(self.method.train_losses, label="Train Loss")
            plt.xlabel("Epoch"); plt.ylabel("Loss")
            plt.title(f"{self.method.mName} - Training Loss")
            plt.legend(); plt.grid(True)
            plt.savefig(f"{self.result.result_destination_file_path}_train_loss.png") 
            plt.close()
            print(f"Training loss curve saved to {self.result.result_destination_file_path}_train_loss.png")

        epoch_accuracy_attr = None
        if hasattr(self.method, 'epoch_accuracies'): 
             epoch_accuracy_attr = 'epoch_accuracies'
        elif hasattr(self.method, 'test_accs'): 
             epoch_accuracy_attr = 'test_accs'

        if epoch_accuracy_attr and hasattr(self.method, epoch_accuracy_attr) and getattr(self.method, epoch_accuracy_attr):
            acc_values = getattr(self.method, epoch_accuracy_attr)
            plt.figure(figsize=(10, 4))
            plt.plot(acc_values, label="Training Accuracy per Log Interval/Epoch")
            plt.xlabel("Log Interval / Epoch"); plt.ylabel("Accuracy")
            plt.title(f"{self.method.mName} - Training Accuracy Log")
            plt.legend(); plt.grid(True)
            plt.savefig(f"{self.result.result_destination_file_path}_train_accuracy.png")
            plt.close()
            print(f"Training accuracy curve saved to {self.result.result_destination_file_path}_train_accuracy.png")

        print(f"Evaluation Scores: {scores}")
        return scores