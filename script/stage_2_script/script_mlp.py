from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import pandas as pd

#---- Multi-Layer Perceptron script ----
if __name__ == '__main__':
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- Data loading and inspection ---------------------
    print("--- Loading Data ---")
    data_obj = Dataset_Loader('Stage 2 Data', '')
    # Updated path for Stage 2 data - Correct relative path from project root
    data_obj.dataset_source_folder_path = 'data/stage_2_data/'
    loaded_data = data_obj.load()

    if loaded_data is None or loaded_data['train'] is None or loaded_data['test'] is None:
        print("ERROR: Data loading failed. Exiting script.")
        exit()

    X_train = loaded_data['train']['X']
    y_train = loaded_data['train']['y']

    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
         print("ERROR: Training data is not in the expected pandas format. Exiting.")
         exit()
    if X_train.empty or y_train.empty:
        print("ERROR: Training data is empty. Exiting.")
        exit()


    # Determine n_features and n_classes from the loaded training data
    print(f"Unique labels found in y_train: {np.unique(y_train)}")
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    print(f"--- Data Loaded: Features={n_features}, Classes={n_classes} ---")
    # ------------------------------------------------------

    # ---- Method, Result, Evaluate object initialization ----
    method_obj = Method_MLP('multi-layer perceptron', '', n_features, n_classes)

    result_obj = Result_Saver('saver', '')
    # Updated path for Stage 2 results - Correct relative path from project root
    result_obj.result_destination_folder_path = 'result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- Setting initialization --------------------------
    setting_obj = Setting_Train_Test_Split('train test split setup', '')
    # ------------------------------------------------------

    # ---- Running section ---------------------------------
    print('************ Start ************')
    setting_obj.dataset = data_obj
    setting_obj.method = method_obj
    setting_obj.result = result_obj
    setting_obj.evaluate = evaluate_obj

    eval_score, _ = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    if eval_score is not None:
        print(f'MLP Accuracy: {eval_score:.4f}')
    else:
        print('MLP Execution failed.')
    print('************ Finish ************')
    # ------------------------------------------------------