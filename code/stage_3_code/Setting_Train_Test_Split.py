'''
Concrete Setting class for Stage 2: Uses pre-defined train/test sets.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
# Removed train_test_split import
import numpy as np
from typing import Tuple, Optional, Dict, Any # Added for type hints
import matplotlib.pyplot as plt
# Ensure your Method classes are imported
from code.stage_3_code.Method_MLP import Method_MLP
from code.stage_3_code.Method_CNN import Method_CNN

class Setting_Train_Test_Split(setting):
    # Removed fold attribute as it's not used for train/test split scenario

    def load_run_save_evaluate(self) -> Any: # Return type might be dict of scores now

        print("--- Loading data ---")
        # The self.dataset instance should be an instance of the (modified) Dataset_Loader,
        # initialized with the specific dataset_name (e.g., "MNIST") in your main script.
        loaded_data: Dict[str, Optional[Dict[str, Any]]] = self.dataset.load()

        train_data = loaded_data.get('train')
        test_data = loaded_data.get('test')

        if not (train_data and train_data.get('X') is not None and train_data.get('y') is not None and train_data['X'].size > 0):
            print("ERROR: Training data loading failed or data is empty. Cannot proceed.")
            return None, None 

        if not (test_data and test_data.get('X') is not None and test_data.get('y') is not None and test_data['X'].size > 0):
             print("ERROR: Testing data loading failed or data is empty. Cannot proceed.")
             return None, None

        print("--- Data loaded successfully ---")
        print(f"Train data X shape: {train_data['X'].shape}, y shape: {train_data['y'].shape}")
        print(f"Test data X shape: {test_data['X'].shape}, y shape: {test_data['y'].shape}")

        # If the method is CNN, ensure it has been initialized correctly.
        # The Method_CNN instance (self.method) should have been created in your main script
        # using image_channels, num_classes, image_size obtained from self.dataset *after* loading.
        # Example:
        # In main script:
        #   my_dataset = Dataset_Loader(dName="MNIST", dDescription="MNIST data", dataset_name="MNIST")
        #   my_dataset.load() # This populates image_channels, image_height, etc.
        #   my_cnn_method = Method_CNN(...,
        #                              input_channels=my_dataset.image_channels,
        #                              num_classes=my_dataset.num_classes,
        #                              image_size=(my_dataset.image_height, my_dataset.image_width))
        #   self.method = my_cnn_method (when setting up Setting_Train_Test_Split)

        if isinstance(self.method, Method_CNN):
            # Verify the method has the correct parameters from the dataset loader
            if not (self.method.input_channels == self.dataset.image_channels and
                    self.method.num_classes == self.dataset.num_classes and
                    self.method.image_h == self.dataset.image_height and
                    self.method.image_w == self.dataset.image_width):
                print("Warning: CNN method parameters might not match loaded dataset parameters.")
                print(f"  Method: C={self.method.input_channels}, Classes={self.method.num_classes}, H={self.method.image_h}, W={self.method.image_w}")
                print(f"  Dataset: C={self.dataset.image_channels}, Classes={self.dataset.num_classes}, H={self.dataset.image_height}, W={self.dataset.image_width}")
                # Ideally, you'd raise an error or re-initialize the method here, but re-init is complex.
                # Best to ensure correct initialization in the main script.

        elif isinstance(self.method, Method_MLP):
            # If MLP, data needs to be flattened if it's image data
            if train_data['X'].ndim > 2: # e.g. (N, C, H, W)
                print("Flattening data for MLP...")
                num_train_samples = train_data['X'].shape[0]
                num_test_samples = test_data['X'].shape[0]
                train_data['X'] = train_data['X'].reshape(num_train_samples, -1)
                test_data['X'] = test_data['X'].reshape(num_test_samples, -1)
                print(f"  New train X shape for MLP: {train_data['X'].shape}")
                print(f"  New test X shape for MLP: {test_data['X'].shape}")
            
            # Ensure MLP is initialized with correct n_features and n_classes
            # This should also happen in the main script.
            # n_features = train_data['X'].shape[1]
            # n_classes = self.dataset.num_classes # Or derive from y labels


        print("--- Running method ---")
        self.method.data = loaded_data 
        learned_result: Dict[str, Any] = self.method.run()

        if learned_result is None:
            print("ERROR: Method execution failed or returned None. Cannot proceed.")
            return None, None # Or an empty dict

        print("--- Saving results ---")
        self.result.data = learned_result
        self.result.save()

        print("--- Evaluating results ---")
        self.evaluate.data = learned_result
        scores: Dict[str, float] = self.evaluate.evaluate() # evaluate() returns a dict

        # Plotting (ensure method has these attributes)
        if hasattr(self.method, 'train_losses') and self.method.train_losses:
            plt.figure(figsize=(10, 4))
            plt.plot(self.method.train_losses, label="Train Loss")
            plt.xlabel("Epoch"); plt.ylabel("Loss")
            plt.title(f"{self.method.mName} - Training Loss")
            plt.legend(); plt.grid(True)
            plt.savefig(f"{self.method.mName}_train_loss.png")
            plt.close()

        # Use the renamed attribute for clarity
        epoch_accuracy_attr = 'train_acc_epochs' if hasattr(self.method, 'train_acc_epochs') else 'test_accs'
        if hasattr(self.method, epoch_accuracy_attr) and getattr(self.method, epoch_accuracy_attr):
            plt.figure(figsize=(10, 4))
            plt.plot(getattr(self.method, epoch_accuracy_attr), label="Training Accuracy per Log Interval")
            plt.xlabel("Log Interval (e.g., every 10 epochs)"); plt.ylabel("Accuracy")
            plt.title(f"{self.method.mName} - Training Accuracy Log")
            plt.legend(); plt.grid(True)
            plt.savefig(f"{self.method.mName}_train_accuracy_log.png")
            plt.close()

        print(f"Evaluation Scores: {scores}")
        return scores # Return the dictionary of scores