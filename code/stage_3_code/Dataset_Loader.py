'''
Concrete IO class for a specific dataset in Stage 3.
Loads train and test data from CSV files using pandas.
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pandas as pd
import os
from typing import Dict, Optional, Any, Tuple
import pickle 
import numpy as np 

class Dataset_Loader(dataset):
    data: Optional[Dict[str, Optional[Dict[str, np.ndarray]]]] = None # X, y will be numpy arrays
    dataset_source_folder_path: str = 'data/stage_3_data/' # Ensure trailing slash or use os.path.join

    # New attributes for image datasets
    dataset_name: str # e.g., "MNIST", "CIFAR", "ORL"
    image_channels: Optional[int] = None
    image_height: Optional[int] = None
    image_width: Optional[int] = None
    num_classes: Optional[int] = None
    actual_img_shape_from_load: Optional[Tuple[int, ...]] = None # (C, H, W)

    def __init__(self, dName: str, dDescription: str, dataset_name: str): # dataset_name is the pickle file base name
        super().__init__(dName, dDescription)
        self.dataset_name = dataset_name.upper() # Store and use this to find the pickle file

    def _process_pickle_split(self, data_split: list, split_name: str) -> Dict[str, np.ndarray]:
        all_images = []
        all_labels = []

        if not data_split:
            print(f"Warning: No data in '{split_name}' split for {self.dataset_name}.")
            return {'X': np.array([]), 'y': np.array([])}

        # Dynamically determine image properties from the first image if not set
        if self.image_channels is None or self.image_height is None or self.image_width is None:
            first_image_raw = data_split[0]['image']
            # Normalize to [0,1] for shape deduction if needed, and then process
            temp_img = first_image_raw / 255.0 if first_image_raw.max() > 1.0 else first_image_raw

            if self.dataset_name == "MNIST" or self.dataset_name == "ORL":
                # Grayscale: (H, W)
                if temp_img.ndim == 2:
                    self.image_channels = 1
                    self.image_height, self.image_width = temp_img.shape
                    self.actual_img_shape_from_load = (self.image_channels, self.image_height, self.image_width)
                elif temp_img.ndim == 3 and temp_img.shape[2] == 3:
                    self.image_channels = 1
                    self.image_height, self.image_width = temp_img.shape[0:2]
                    self.actual_img_shape_from_load = (self.image_channels, self.image_height, self.image_width)
                else:
                    raise ValueError(f"{self.dataset_name} images expected to be 2D (H,W), got {temp_img.ndim}D")
            elif self.dataset_name == "CIFAR":
                # Color: (H, W, C)
                if temp_img.ndim == 3 and temp_img.shape[2] == 3:
                    self.image_channels = 3
                    self.image_height, self.image_width = temp_img.shape[0], temp_img.shape[1]
                    self.actual_img_shape_from_load = (self.image_channels, self.image_height, self.image_width) # Store as (C,H,W)
                else:
                    raise ValueError(f"{self.dataset_name} images expected to be 3D (H,W,C) with C=3, got shape {temp_img.shape}")
            else:
                raise ValueError(f"Unknown dataset_type {self.dataset_name} for automatic shape deduction.")
            print(f"Deduced for {self.dataset_name}: C={self.image_channels}, H={self.image_height}, W={self.image_width}")


        for instance in data_split:
            image_matrix_raw = instance['image'] # numpy array
            image_label = instance['label']
            if self.dataset_name == "ORL":
                image_label -= 1  # Make labels zero-based for CrossEntropyLoss

            # Normalize image data to [0, 1]
            image_matrix = image_matrix_raw / 255.0 if image_matrix_raw.max() > 1.0 else image_matrix_raw.copy()

            if self.dataset_name == "ORL": # Gray scale, but with RGB values
                if image_matrix.ndim == 3 and image_matrix.shape[2] == 3:
                    image_matrix = image_matrix[:, :, 0:1] # Use only one channel, since they are all the same value
            
            # Reshape to (C, H, W)
            if self.dataset_name == "MNIST" or self.dataset_name == "ORL": # Grayscale
                if image_matrix.ndim == 2: # (H, W)
                    image_matrix = np.expand_dims(image_matrix, axis=0) # (1, H, W)
                elif image_matrix.ndim == 3 and image_matrix.shape[2] == 1:  # (H, W, 1)
                    image_matrix = np.transpose(image_matrix, (2, 0, 1))  # → (1, H, W)
                elif image_matrix.ndim == 3 and image_matrix.shape[0] == 1: # Already (1, H, W)
                    pass
                else:
                    raise ValueError(f"Unexpected image shape for {self.dataset_name}: {image_matrix.shape}")
            elif self.dataset_name == "CIFAR": # Color
                if image_matrix.ndim == 3 and image_matrix.shape[2] == 3: # (H, W, C)
                    image_matrix = np.transpose(image_matrix, (2, 0, 1)) # (C, H, W)
                elif image_matrix.ndim == 3 and image_matrix.shape[0] == 3: # Already (C, H, W)
                    pass
                else:
                    raise ValueError(f"Unexpected image shape for {self.dataset_name}: {image_matrix.shape}")
            
            # Final check for consistency with deduced shape
            if image_matrix.shape != self.actual_img_shape_from_load:
                raise ValueError(f"Processed image shape {image_matrix.shape} mismatch with deduced {self.actual_img_shape_from_load}")

            all_images.append(image_matrix)
            all_labels.append(image_label)
        
        X_data = np.array(all_images, dtype=np.float32)
        y_data = np.array(all_labels, dtype=np.int64) # Labels usually integer type

        if self.num_classes is None and y_data.size > 0: # Determine num_classes from this split
            self.num_classes = len(np.unique(y_data))

        return {'X': X_data, 'y': y_data}

    def load(self) -> Dict[str, Optional[Dict[str, np.ndarray]]]:
        print(f'--- Loading Image Data for: {self.dataset_name} ---')
        
        pickle_file_path = os.path.join(self.dataset_source_folder_path, self.dataset_name)

        train_data_processed: Optional[Dict[str, np.ndarray]] = None
        test_data_processed: Optional[Dict[str, np.ndarray]] = None

        try:
            print(f"Loading full dataset from pickle: {pickle_file_path}")
            with open(pickle_file_path, 'rb') as f:
                full_raw_data = pickle.load(f) 
            
            if 'train' not in full_raw_data or 'test' not in full_raw_data:
                print(f"ERROR: Pickle file {pickle_file_path} must contain 'train' and 'test' keys.")
                self.data = {'train': None, 'test': None}
                return self.data

            # Process training data
            print("Processing training data...")
            train_data_processed = self._process_pickle_split(full_raw_data['train'], "train")
            if train_data_processed['X'].size > 0:
                 print(f"  Training X shape: {train_data_processed['X'].shape}, y unique labels: {len(np.unique(train_data_processed['y']))}")

            # Process testing data
            print("Processing testing data...")
            test_data_processed = self._process_pickle_split(full_raw_data['test'], "test")
            if test_data_processed['X'].size > 0:
                 print(f"  Testing X shape: {test_data_processed['X'].shape}, y unique labels: {len(np.unique(test_data_processed['y']))}")
            
            # Ensure num_classes is consistent if derived from both splits or globally set
            if self.num_classes is None and train_data_processed and test_data_processed: # Fallback
                unique_labels = np.unique(np.concatenate((train_data_processed['y'], test_data_processed['y'])))
                self.num_classes = len(unique_labels)
            print(f"Final Dataset Info: C={self.image_channels}, H={self.image_height}, W={self.image_width}, Classes={self.num_classes}")

        except FileNotFoundError:
            print(f"ERROR: Dataset pickle file not found at {pickle_file_path}")
            self.data = {'train': None, 'test': None} # Ensure data is set even on error
            return self.data
        except Exception as e:
            print(f"ERROR loading or processing dataset from {pickle_file_path}: {e}")
            self.data = {'train': None, 'test': None}
            return self.data

        self.data = {'train': train_data_processed, 'test': test_data_processed}
        print('--- Data Loading Complete ---')
        return self.data