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

class Setting_Train_Test_Split(setting):
    # Removed fold attribute as it's not used for train/test split scenario

    def load_run_save_evaluate(self) -> Tuple[Optional[float], Optional[float]]:

        # Load dataset
        print("--- Loading data ---")
        # Ensure the dataset loader returns the expected dictionary structure
        loaded_data: Dict[str, Optional[Dict[str, Any]]] = self.dataset.load()

        # Check if data loaded successfully and contains required keys
        train_data = loaded_data.get('train')
        test_data = loaded_data.get('test')

        if train_data is None or 'X' not in train_data or 'y' not in train_data:
            print("ERROR: Training data loading failed or missing 'X'/'y'. Cannot proceed.")
            return None, None # Indicate failure

        if test_data is None or 'X' not in test_data or 'y' not in test_data:
             print("ERROR: Testing data loading failed or missing 'X'/'y'. Cannot proceed.")
             return None, None # Indicate failure

        print("--- Data loaded successfully ---")
        print(f"Train data X shape: {train_data['X'].shape}")
        print(f"Train data y shape: {train_data['y'].shape}")
        print(f"Test data X shape: {test_data['X'].shape}")
        print(f"Test data y shape: {test_data['y'].shape}")


        # No train/test split needed as we are using pre-defined sets

        # Run MethodModule
        print("--- Running method ---")
        # Assign the loaded data structure directly to the method
        # The method's run() should be adapted to use this structure: data['train']['X'], data['train']['y'], data['test']['X']
        self.method.data = loaded_data
        learned_result: Dict[str, Any] = self.method.run() # Method's run uses self.data internally

        # Check if method execution produced results
        if learned_result is None:
            print("ERROR: Method execution failed or returned None. Cannot proceed.")
            return None, None

        # Save raw ResultModule
        print("--- Saving results ---")
        self.result.data = learned_result
        # self.result.fold_count is not set as we are not using KFold
        self.result.save()

        self.evaluate.data = learned_result
        
        scores: Dict[str, float] = self.evaluate.evaluate()

        plt.figure()
        plt.plot(self.method.train_losses, label="Train loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.legend()
        plt.savefig("train_loss.png")

        plt.figure()
        plt.plot(self.method.test_accs, label="Test accuracy")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("test_acc.png")

        return scores