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
        # Baseline - Standard MLP
        {
            "hidden_dims": (512, 256, 128),
            "activation": nn.ReLU,
            "dropout": 0.3,
            "optimizer_cls": optim.Adam,
            "optimizer_kwargs": {"lr": 1e-3}
        },
        # Deep architecture with SiLU
        {
            "hidden_dims": (1024, 512, 256, 128),
            "activation": nn.SiLU,
            "dropout": 0.4,
            "optimizer_cls": optim.AdamW,
            "optimizer_kwargs": {"lr": 1e-3, "weight_decay": 1e-4}
        },
        # Shallow but wide architecture
        {
            "hidden_dims": (1024, 512),
            "activation": nn.ReLU,
            "dropout": 0.5,
            "optimizer_cls": optim.SGD,
            "optimizer_kwargs": {"lr": 1e-2, "momentum": 0.9}
        },
        # Residual-like architecture (more layers, less dropout)
        {
            "hidden_dims": (256, 256, 256, 256),
            "activation": nn.GELU,
            "dropout": 0.2,
            "optimizer_cls": optim.AdamW,
            "optimizer_kwargs": {"lr": 5e-4, "weight_decay": 1e-5}
        },
        # Compact architecture with strong regularization
        {
            "hidden_dims": (128, 64),
            "activation": nn.ELU,
            "dropout": 0.6,
            "optimizer_cls": optim.Adam,
            "optimizer_kwargs": {"lr": 2e-3}
        }
    ]

    print("\n************ Architecture Comparison ************")
    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n{'='*50}")
        print(f"Training Architecture {i} of {len(configs)}")
        print(f"{'='*50}")
        print("\nConfiguration Details:")
        print(f"- Architecture Name: Arch_{i}")
        print(f"- Network Structure: {' -> '.join(str(x) for x in cfg['hidden_dims'])} -> {n_classes}")
        print(f"- Total Layers: {len(cfg['hidden_dims']) + 1}")
        print(f"- Activation Function: {cfg['activation'].__name__}")
        print(f"- Dropout Rate: {cfg['dropout']}")
        print(f"- Optimizer: {cfg['optimizer_cls'].__name__}")
        if 'optimizer_kwargs' in cfg:
            print("- Optimizer Settings:")
            for k, v in cfg['optimizer_kwargs'].items():
                print(f"  • {k}: {v}")
        print(f"\nStarting training...")
        
        method = Method_MLP("MLP", f"Architecture_{i}", n_features, n_classes, **cfg)
        setting_obj.method = method
        scores = setting_obj.load_run_save_evaluate()
        
        print(f"\nTraining Complete - Results:")
        # print(f"- Accuracy: {scores['accuracy']:.4f}")
        # print(f"- F1 Score: {scores['f1_score']:.4f}")
        # print(f"- Precision: {scores['precision']:.4f}")
        # print(f"- Recall: {scores['recall']:.4f}")
        
        results.append({
            "Architecture": f"Arch_{i}",
            "Hidden_Dims": str(cfg['hidden_dims']),
            "Activation": cfg['activation'].__name__,
            "Dropout": cfg['dropout'],
            "Optimizer": cfg['optimizer_cls'].__name__,
            **scores
        })

    # Create results DataFrame and save
    df = pd.DataFrame(results)
    print("\n" + "="*80)
    print("************ Final Results Comparison ************")
    print("="*80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print("\n" + df.to_string(index=False))
    
    # Save results to CSV
    results_path = 'result/stage_2_result/mlp_architecture_comparison.csv'
    df.to_csv(results_path, index=False)
    print(f"\nDetailed results saved to: {results_path}")

    # Find best architecture
    best_acc_idx = df['accuracy'].idxmax()
    print("\n" + "="*80)
    print("************ Best Architecture Summary ************")
    print("="*80)
    best_arch = df.iloc[best_acc_idx]
    print(f"\nBest Architecture: {best_arch['Architecture']}")
    print(f"Configuration:")
    print(f"- Hidden Layers: {best_arch['Hidden_Dims']}")
    print(f"- Activation: {best_arch['Activation']}")
    print(f"- Dropout: {best_arch['Dropout']}")
    print(f"- Optimizer: {best_arch['Optimizer']}")
    # print(f"\nPerformance Metrics:")
    # print(f"- Accuracy: {best_arch['accuracy']:.4f}")
    # print(f"- F1 Score: {best_arch['f1_score']:.4f}")
    # print(f"- Precision: {best_arch['precision']:.4f}")
    # print(f"- Recall: {best_arch['recall']:.4f}")
    # ------------------------------------------------------