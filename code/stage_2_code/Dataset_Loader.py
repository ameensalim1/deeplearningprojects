'''
Concrete IO class for a specific dataset in Stage 2.
Loads train and test data from CSV files using pandas.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pandas as pd
import os
from typing import Dict, Optional, Any # Added for type hinting

class Dataset_Loader(dataset):
    data: Optional[Dict[str, Optional[Dict[str, Any]]]] = None # More specific type hint
    dataset_source_folder_path: str = 'data/stage_2_data'
    dataset_source_train_file_name: str = 'train.csv'
    dataset_source_test_file_name: str = 'test.csv'

    def __init__(self, dName: str = "Stage 2 Dataset", dDescription: str = "Loads train/test CSV data"):
        super().__init__(dName, dDescription)

    def load(self) -> Dict[str, Optional[Dict[str, Any]]]:

        print('--- Loading Stage 2 Data ---')
        train_data: Optional[Dict[str, Any]] = None
        test_data: Optional[Dict[str, Any]] = None

        train_file_path = os.path.join(self.dataset_source_folder_path, self.dataset_source_train_file_name)
        test_file_path = os.path.join(self.dataset_source_folder_path, self.dataset_source_test_file_name)

        # Load Training Data
        try:
            print(f"Loading training data from: {train_file_path}")
            train_df = pd.read_csv(train_file_path)
            if train_df.empty:
                 print("Warning: Training data file is empty.")
                 train_X = pd.DataFrame() # Empty DataFrame
                 train_y = pd.Series(dtype='object') # Empty Series, adjust dtype if known
            elif train_df.shape[1] < 2:
                 print("Warning: Training data needs at least 2 columns (features + label).")
                 train_X = train_df # Assign all columns to X if only 1 col exists
                 train_y = pd.Series(dtype='object') # Empty Series for y
            else:
                train_y = train_df.iloc[:, 0]
                train_X = train_df.iloc[:, 1:]
            train_data = {'X': train_X, 'y': train_y}
            print("Training data loaded successfully.")
            print(f"  Features (X_train): {train_X.shape}")
            print(f"  Labels (y_train): {train_y.shape}")
        except FileNotFoundError:
            print(f"ERROR: Training file not found at {train_file_path}")
        except Exception as e:
            print(f"ERROR loading training data: {e}")

        # Load Testing Data
        try:
            print(f"Loading testing data from: {test_file_path}")
            test_df = pd.read_csv(test_file_path)
            if test_df.empty:
                 print("Warning: Testing data file is empty.")
                 test_X = pd.DataFrame()
                 test_y = pd.Series(dtype='object')
            elif test_df.shape[1] < 2:
                 print("Warning: Testing data needs at least 2 columns (features + label).")
                 test_X = test_df
                 test_y = pd.Series(dtype='object')
            else:
                 test_y = test_df.iloc[:, 0]
                 test_X = test_df.iloc[:, 1:]
            test_data = {'X': test_X, 'y': test_y}
            print("Testing data loaded successfully.")
            print(f"  Features (X_test): {test_X.shape}")
            print(f"  Labels (y_test): {test_y.shape}")
        except FileNotFoundError:
            print(f"ERROR: Testing file not found at {test_file_path}")
        except Exception as e:
            print(f"ERROR loading testing data: {e}")

        # Store loaded data in the class instance
        self.data = {'train': train_data, 'test': test_data}

        print('--- Data Loading Complete ---')
        # Return the structured data
        return self.data # Return the data structure directly