'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.result import result
import pickle
import os

class Result_Saver(result):
    data = None
    # Removed fold_count attribute
    # fold_count = None
    result_destination_folder_path = str = ''
    result_destination_file_name = str = ''

    def save(self):
        """Saves the evaluation result data to a pickle file."""
        print('saving results...')
        if self.result_destination_folder_path is None or self.result_destination_file_name is None:
            print("Error: Result destination folder or file name not set.")
            return
        # Construct filename without fold_count, ensure .pkl extension
        file_path = os.path.join(self.result_destination_folder_path, self.result_destination_file_name)
        if not file_path.endswith('.pkl'):
             file_path += '.pkl' # Add extension if missing

        # Ensure the directory exists
        os.makedirs(self.result_destination_folder_path, exist_ok=True)

        try:
            with open(file_path, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"Results saved successfully to: {file_path}")
        except IOError as e:
            print(f"Error saving results to {file_path}: {e}")
        except pickle.PicklingError as e:
            print(f"Error pickling data: {e}")