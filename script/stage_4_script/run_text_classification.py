import sys
import os

_project_root_path_fix = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _project_root_path_fix not in sys.path:
    sys.path.insert(0, _project_root_path_fix)

import torch
import argparse # For CUDA and MPS flags

if _project_root_path_fix not in sys.path:
    sys.path.insert(0, _project_root_path_fix)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_RNN import Method_RNN
from code.stage_4_code.Setting_RNN_TextClassification import (
    Setting_RNN_TextClassification,
)
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy


def main(use_cuda: bool, use_mps: bool):
    # ---- Hyperparameters for RNN ----
    embedding_dim = 100
    hidden_dim = 128
    rnn_type = "LSTM"  # 'RNN', 'LSTM', or 'GRU'
    num_rnn_layers = 1
    bidirectional = True
    dropout_prob_embed = 0.5
    dropout_prob_rnn = 0.2  # Applied if num_rnn_layers > 1
    dropout_prob_fc = 0.5
    max_epoch = 20
    learning_rate = 0.001
    max_seq_length = 250

    # ---- Dataset Setup ----
    print("Initializing Dataset Loader...")
    try:
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
    except ImportError:
        print(
            "CRITICAL: NLTK basic modules not found. Please install NLTK and download resources."
        )
        print("$ pip install nltk")
        print(
            "Then in Python: import nltk; nltk.download('punkt'); nltk.download('stopwords')"
        )
        return

    dataset_params = {
        "dName": "MovieReviewSentiment",
        "dDescription": "IMDb Movie Review Sentiment Classification",
        "max_seq_length": max_seq_length,
    }
    text_dataset = Dataset_Loader(**dataset_params)

    print("Loading data (this will also build vocabulary)...")
    loaded_data = text_dataset.load()

    if not loaded_data or not loaded_data["train"]["X"].size > 0:
        print("Failed to load data or training data is empty. Exiting.")
        return

    # ---- Method Setup ----
    print("Initializing RNN Method...")
    method_params = {
        "mName": f"Method_RNN_{rnn_type}",
        "mDescription": f"RNN ({rnn_type}, EmbDim:{embedding_dim}, HidDim:{hidden_dim}, Layers:{num_rnn_layers}, Bi:{bidirectional}) for Text Classification",
        "vocab_size": text_dataset.vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_classes": 2,  # Binary sentiment: positive/negative
        "rnn_type": rnn_type,
        "num_rnn_layers": num_rnn_layers,
        "bidirectional": bidirectional,
        "dropout_prob_embed": dropout_prob_embed,
        "dropout_prob_rnn": dropout_prob_rnn,
        "dropout_prob_fc": dropout_prob_fc,
        "pad_idx": text_dataset.vocab[text_dataset.PAD_TOKEN],
        "optimizer_kwargs": {"lr": learning_rate},
    }
    rnn_method = Method_RNN(**method_params)
    rnn_method.max_epoch = max_epoch

    # ---- Device Setup (CUDA, MPS, CPU) ----
    # Priority: CUDA > MPS > CPU
    # Method_RNN also has internal device logic, but we explicitly set it here.
    
    selected_device_str = "cpu" # Default

    if use_cuda:
        if torch.cuda.is_available():
            selected_device_str = "cuda"
            print("CUDA is available. Requesting to use CUDA.")
        else:
            print("CUDA was requested, but is not available.")
            if not use_mps: # If MPS not also requested, will fall to CPU
                 print("Will attempt to use CPU.")
    if use_mps:
        if torch.backends.mps.is_available():
            selected_device_str = "mps"
            print("Apple Metal (MPS) is available. Requesting to use MPS.")
        else:
            print("Apple Metal (MPS) was requested, but is not available or PyTorch version is too old.")
            print("Will attempt to use CPU.")

    if selected_device_str == "cpu":
        print("Using CPU.")
    
    final_device = torch.device(selected_device_str)
    rnn_method.to(final_device)
    # The Setting_RNN_TextClassification will also need to be aware of the device
    # for moving data batches. Ensure its 'run' method uses rnn_method.device.

    # ---- Result Saver ----
    print("Initializing Result Saver...")
    results_folder = os.path.join(
        project_root, "results", "stage_4_results", "text_classification"
    )
    os.makedirs(results_folder, exist_ok=True)
    result_file_name = f"{rnn_method.method_name}_results_LR{learning_rate}_Epochs{max_epoch}_Device{selected_device_str.upper()}"
    result_save_path = os.path.join(results_folder, result_file_name)

    result_saver = Result_Saver(
        "TextClassificationResults", f"Results for {rnn_method.method_name}"
    )
    result_saver.result_destination_folder_path = results_folder
    result_saver.result_destination_file_name = result_file_name

    # ---- Evaluator ----
    print("Initializing Evaluator...")
    evaluator_params = {
        "eName": "SentimentEvaluator",
        "eDescription": "Accuracy, Precision, Recall, F1 for Sentiment Classification",
    }
    evaluator = Evaluate_Accuracy(**evaluator_params)

    # ---- Setting (Experiment Runner) ----
    print("Initializing Setting for RNN Text Classification...")
    text_classification_setting = Setting_RNN_TextClassification(
        "RNNTextClassificationExperiment",
        "Running RNN for IMDb sentiment analysis",
    )
    # Pass the device to the setting, or ensure it uses method.device
    # Assuming Setting_RNN_TextClassification's run method will use rnn_method.device
    # to move data batches to the correct device. If not, you might need to pass
    # final_device to its prepare() or __init__() method.
    text_classification_setting.prepare(
        text_dataset, rnn_method, result_saver, evaluator
    )

    # ---- Run Everything ----
    print(f"Starting the experiment on device: {final_device}...")
    final_scores = text_classification_setting.load_run_save_evaluate()

    if final_scores:
        print("--- Experiment Finished ---")
        print("Final Evaluation Scores:", final_scores)
        print(f"Results and plots saved in: {results_folder}")
        print(
            f"Specifically, main results file: {result_save_path}.pkl (and associated plots)"
        )
    else:
        print("--- Experiment Encountered an Error ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate RNN for text classification."
    )
    parser.add_argument(
        "--cuda", action="store_true", help="Enable CUDA if available"
    )
    parser.add_argument(
        "--mps",
        action="store_true",
        help="Enable Apple Metal (MPS) if available (for M1/M2/M3 Macs)",
    )
    args = parser.parse_args()

    main(use_cuda=args.cuda, use_mps=args.mps)
