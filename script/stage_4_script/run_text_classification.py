import torch
import argparse # For CUDA flag
import sys
import os

# Adjust system path to include the project root for imports
# This assumes 'script/stage_4_script/' is one level down from project root where 'code/' exists.
# You might need to adjust this depending on your exact project structure and how you run the script.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # Goes up two levels
sys.path.insert(0, project_root)

from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_RNN import Method_RNN
from code.stage_4_code.Setting_RNN_TextClassification import Setting_RNN_TextClassification
from code.stage_4_code.Result_Saver import Result_Saver
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy

def main(use_cuda: bool):
    # ---- Hyperparameters for RNN ----
    # These can be tuned
    embedding_dim = 100
    hidden_dim = 128
    rnn_type = 'LSTM'  # 'RNN', 'LSTM', or 'GRU'
    num_rnn_layers = 1
    bidirectional = True
    dropout_prob_embed = 0.5
    dropout_prob_rnn = 0.2 # Applied if num_rnn_layers > 1
    dropout_prob_fc = 0.5
    max_epoch = 20 # Adjust as needed; 100 was in Method_RNN, can override here or there
    learning_rate = 0.001
    
    # Max sequence length for padding/truncating
    max_seq_length = 250 # Dataset_Loader default is 200, ensure consistency or pass to it

    # ---- Dataset Setup ----
    print("Initializing Dataset Loader...")
    # Note: Dataset_Loader prints NLTK download instructions if modules are missing.
    # Consider adding a check here to ensure NLTK resources are actually available before proceeding.
    try:
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        # This is a basic check; actual download might still be needed if punkt/stopwords aren't there.
    except ImportError:
        print("CRITICAL: NLTK basic modules not found. Please install NLTK and download resources.")
        print("$ pip install nltk")
        print("Then in Python: import nltk; nltk.download('punkt'); nltk.download('stopwords')")
        return

    dataset_params = {
        'dName': "MovieReviewSentiment",
        'dDescription': "IMDb Movie Review Sentiment Classification",
        'max_seq_length': max_seq_length
    }
    text_dataset = Dataset_Loader(**dataset_params)
    
    print("Loading data (this will also build vocabulary)...")
    # load() will call _build_vocab, which sets text_dataset.vocab, .vocab_size, .idx2word
    # It's crucial that this happens BEFORE Method_RNN is initialized.
    loaded_data = text_dataset.load() 

    if not loaded_data or not loaded_data['train']['X'].size > 0:
        print("Failed to load data or training data is empty. Exiting.")
        return

    # ---- Method Setup ----
    print("Initializing RNN Method...")
    method_params = {
        'mName': f"Method_RNN_{rnn_type}",
        'mDescription': f"RNN ({rnn_type}, EmbDim:{embedding_dim}, HidDim:{hidden_dim}, Layers:{num_rnn_layers}, Bi:{bidirectional}) for Text Classification",
        'vocab_size': text_dataset.vocab_size, # Crucial: Get from loaded dataset
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'num_classes': 2, # Binary sentiment: positive/negative
        'rnn_type': rnn_type,
        'num_rnn_layers': num_rnn_layers,
        'bidirectional': bidirectional,
        'dropout_prob_embed': dropout_prob_embed,
        'dropout_prob_rnn': dropout_prob_rnn,
        'dropout_prob_fc': dropout_prob_fc,
        'pad_idx': text_dataset.vocab[text_dataset.PAD_TOKEN], # Crucial: Get from loaded dataset
        'optimizer_kwargs': {'lr': learning_rate}
        # 'max_epoch': max_epoch # Can be set in Method_RNN or overridden here if Method_RNN accepts it in init
    }
    rnn_method = Method_RNN(**method_params)
    rnn_method.max_epoch = max_epoch # Override if not in init or to be explicit

    # Override device based on CUDA flag (Method_RNN also has internal device logic)
    # This ensures the model is explicitly moved if CUDA is requested and available.
    if use_cuda and torch.cuda.is_available():
        print("CUDA is available. Requesting to use CUDA.")
        rnn_method.to(torch.device("cuda"))
    elif use_cuda and not torch.cuda.is_available():
        print("CUDA was requested, but is not available. Using CPU.")
        rnn_method.to(torch.device("cpu"))
    else:
        print("Using CPU.")
        rnn_method.to(torch.device("cpu"))


    # ---- Result Saver ----
    print("Initializing Result Saver...")
    # Output results to a specific folder for stage 4
    results_folder = os.path.join(project_root, "results", "stage_4_results", "text_classification")
    os.makedirs(results_folder, exist_ok=True)
    result_file_name = f"{rnn_method.mName}_results_LR{learning_rate}_Epochs{max_epoch}"
    result_save_path = os.path.join(results_folder, result_file_name)

    result_saver_params = {
        'rName': "TextClassificationResults",
        'rDescription': f"Results for {rnn_method.mName}",
        'destination_folder_path': results_folder, # Pass the folder
        'destination_file_name': result_file_name  # Pass the file name (without .pkl)
    }
    result_saver = Result_Saver(**result_saver_params)

    # ---- Evaluator ----
    print("Initializing Evaluator...")
    evaluator_params = {
        'eName': "SentimentEvaluator",
        'eDescription': "Accuracy, Precision, Recall, F1 for Sentiment Classification"
    }
    evaluator = Evaluate_Accuracy(**evaluator_params) # Assuming Evaluate_Accuracy is suitable

    # ---- Setting (Experiment Runner) ----
    print("Initializing Setting for RNN Text Classification...")
    setting_params = {
        'sName': "RNNTextClassificationExperiment",
        'sDescription': "Running RNN for IMDb sentiment analysis",
        'dataset': text_dataset,
        'method': rnn_method,
        'result_saver': result_saver,
        'evaluator': evaluator
    }
    text_classification_setting = Setting_RNN_TextClassification(**setting_params)

    # ---- Run Everything ----
    print("Starting the experiment: Load, Run, Save, Evaluate...")
    final_scores = text_classification_setting.load_run_save_evaluate()

    if final_scores:
        print("--- Experiment Finished ---")
        print("Final Evaluation Scores:", final_scores)
        print(f"Results and plots saved in: {results_folder}")
        print(f"Specifically, main results file: {result_save_path}.pkl (and associated plots)")
    else:
        print("--- Experiment Encountered an Error ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate RNN for text classification.")
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA if available')
    args = parser.parse_args()
    
    main(use_cuda=args.cuda) 