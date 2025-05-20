import torch
import argparse
import sys
import os

# make sure project root is on PYTHONPATH
current_dir   = os.path.dirname(os.path.abspath(__file__))
project_root  = os.path.dirname(os.path.dirname(current_dir))
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

    # move to GPU if requested
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    gen_method.to(device)
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
    # .prepare takes: dataset, method, result_saver, (and evaluator—but text‐gen doesn’t use one)
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
