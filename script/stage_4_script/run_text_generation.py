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
from code.stage_4_code.Gen_Dataset_Loader import Gen_Dataset_Loader
from code.stage_4_code.Method_RNN_Generate import Method_RNN_Generate
from code.stage_4_code.Setting_RNN_TextGeneration import Setting_RNN_TextGeneration
from code.stage_4_code.Result_Saver import Result_Saver

def main(use_cuda: bool, seed_text: str, max_length: int):
    # ── Hyperparameters ──────────────────────────────────────────────────────────
    embedding_dim     = 100
    hidden_dim        = 256
    rnn_type          = 'LSTM'   # 'RNN', 'LSTM', or 'GRU'
    num_rnn_layers    = 2
    bidirectional     = False
    dropout_embed     = 0.3
    dropout_rnn       = 0.3
    dropout_fc        = 0.3
    max_epoch         = 50
    learning_rate     = 0.001
    batch_size        = 64

    # ── Dataset Loader ────────────────────────────────────────────────────────────
    print("Initializing text‐generation dataset loader…")
    gen_data = Gen_Dataset_Loader(
        dName="StoryGen",
        dDescription="RNN text generation dataset",
        max_seq_length=50
    )
    data_splits = gen_data.load()
    if data_splits['train']['X'].size == 0:
        print("No data found; aborting.")
        return

    # ── Model Setup ───────────────────────────────────────────────────────────────
    print("Building generation model…")
    gen_method = Method_RNN_Generate(
        mName=f"RNNGen_{rnn_type}",
        mDescription="RNN‐based text generator",
        vocab_size    = gen_data.vocab_size,
        embedding_dim = embedding_dim,
        hidden_dim    = hidden_dim,
        rnn_type      = rnn_type,
        num_rnn_layers= num_rnn_layers,
        bidirectional = bidirectional,
        dropout_prob_embed=dropout_embed,
        dropout_prob_rnn  =dropout_rnn,
        dropout_prob_fc   =dropout_fc,
        pad_idx       =gen_data.vocab[gen_data.PAD_TOKEN],
        optimizer_kwargs={'lr': learning_rate},
    )
    gen_method.max_epoch = max_epoch

    # ---- Device Setup (CUDA, MPS, CPU) ----
    selected_device_str = "cpu" # Default

    if use_cuda:
        if torch.cuda.is_available():
            selected_device_str = "cuda"
            print("CUDA is available. Using CUDA.")
        else:
            print("CUDA was requested, but is not available.")
    
    if selected_device_str == "cpu": # Only check for MPS if CUDA wasn't selected or isn't available
        if torch.backends.mps.is_available():
            selected_device_str = "mps"
            print("Apple Metal (MPS) is available. Using MPS.")
        else:
            if use_cuda: # If CUDA was requested but not available, and MPS also not available
                print("MPS is not available. Falling back to CPU.")
            else: # If neither CUDA nor MPS were specifically requested or available
                print("Neither CUDA nor MPS available/selected. Using CPU.")
    
    device = torch.device(selected_device_str)
    gen_method.to(device)
    # Ensure the method object knows its device for internal data movement
    if hasattr(gen_method, 'device'):
        gen_method.device = device
    else:
        # If Method_RNN_Generate doesn't have a device attribute by design,
        # this might indicate a different way it expects device info or handles data.
        # For now, we'll attempt to set it.
        print("Warning: Method_RNN_Generate might not have a 'device' attribute. Attempting to set it.")
        gen_method.device = device

    print(f"Using device: {device}")

    # ── Result Saver ─────────────────────────────────────────────────────────────
    results_folder = os.path.join(project_root, "results", "stage_4_results", "text_generation")
    os.makedirs(results_folder, exist_ok=True)
    save_name = f"{gen_method.mName}_seed_{seed_text.replace(' ','_')}_len{max_length}"

    saver = Result_Saver(save_name, "Generated text and model checkpoints")
    saver.result_destination_folder_path = results_folder
    saver.result_destination_file_name = save_name


    # ── Setting / Runner ─────────────────────────────────────────────────────────
    print("Initializing text‐generation setting…")
    setting = Setting_RNN_TextGeneration("RNNTextGenExperiment", "Train + generate text with RNN")
    # .prepare takes: dataset, method, result_saver, (and evaluator—but text‐gen doesn't use one)
    setting.prepare(gen_data, gen_method, saver, None)

    # ── Run train + generate ─────────────────────────────────────────────────────
    print(" Starting training & generation…")
    scores_and_text = setting.load_run_save_evaluate(seed_text=seed_text, max_generate_len=max_length)

    # show generated sample
    generated = scores_and_text.get("generated_text", "")
    print("\n── Generated text ─────────────────────────")
    print(generated)
    print("─────────────────────────────────────────────")
    print(f"All outputs saved under: {results_folder}/{save_name}.pkl")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train & run RNN text generation.")
    p.add_argument("--cuda", action="store_true", help="use CUDA if available")
    p.add_argument("--seed", type=str, default="Once upon a", help="starting seed text")
    p.add_argument("--max_len", type=int, default=200, help="max tokens to generate")
    args = p.parse_args()
    main(use_cuda=args.cuda, seed_text=args.seed, max_length=args.max_len)
