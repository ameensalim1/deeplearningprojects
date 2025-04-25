from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

#---- Multi-Layer Perceptron script ----
if __name__ == '__main__':
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- Data loading and inspection ---------------------
    print("--- Loading Data ---")
    data_obj = Dataset_Loader('Stage 2 Data', 'Loads train/test CSVs') 
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
    method_obj = Method_MLP(
        "MLP", "Pytorch Classifier",
        n_features, n_classes,
        hidden_dims      = (512, 256, 128),
        activation       = nn.SiLU,
        dropout          = 0.4,
        optimizer_cls    = optim.AdamW,
        optimizer_kwargs = {"lr": 1e-3, "weight_decay": 1e-4},
        loss_fn          = nn.CrossEntropyLoss()
    )

    result_obj = Result_Saver('saver', '')
    # Updated path for Stage 2 results - Correct relative path from project root
    result_obj.result_destination_folder_path = 'result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    evaluate_obj = Evaluate_Accuracy() 
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
    accuracy = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')


    for metric, value in accuracy.items():
        print(f"{metric}: {value:.4f}")


    configs = [
        {"hidden_dims":(512,256,128),"activation":nn.ReLU,  "dropout":0.3, "optimizer_cls":optim.Adam},
        {"hidden_dims":(512,256,128),"activation":nn.SiLU,  "dropout":0.4, "optimizer_cls":optim.AdamW},
        {"hidden_dims":(256,128),      "activation":nn.ReLU,  "dropout":0.5, "optimizer_cls":optim.SGD, "optimizer_kwargs":{"lr":1e-2,"momentum":0.9}},
    ]

    results = []
    for cfg in configs:
        method = Method_MLP("MLP", "tunable MLP", n_features, n_classes, **cfg)
        setting_obj.method = method
        scores = setting_obj.load_run_save_evaluate()
        results.append({**cfg, **scores})

    import pandas as pd
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    df.to_csv("tune_results.csv", index=False)

    # if scores:

        # for metric, value in scores.items():
            # print(f"{metric}: {value:.4f}")

    # else:
        # print('MLP Execution failed.')
    # print('************ Finish ************')
    # ------------------------------------------------------