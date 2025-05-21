from code.base_class.setting import setting
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import torch
import os

class Setting_RNN_TextGeneration(setting):
    """
    Concrete Setting class for Stage 4: RNN Text Generation.
    Handles loading generation text data, training the generator, producing a new sequence,
    saving, and (optionally) evaluating the generated text.
    """

    def load_run_save_evaluate(self, seed_text: Optional[str] = None, max_generate_len: int = 100) -> Optional[Dict[str, Any]]:
        print("--- Loading text generation data ---")
        loaded_data = self.dataset.load()

        # Expect the generation loader to provide 'train': {'X': np.ndarray, 'y': np.ndarray}
        train_split = loaded_data['train']
        print(f"Train sequences: {train_split['X'].shape}")
        self.method.data = loaded_data


        # prepare mini-batch data for next-token training
        seqs = train_split['X']
        X = seqs[:, :-1]
        Y = seqs[:, 1:]
        X_tensor = torch.LongTensor(X).to(self.method.device)
        Y_tensor = torch.LongTensor(Y).to(self.method.device)

        # train
        print("--- Training generation model ---")
        self.method.train_model(X_tensor, Y_tensor, batch_size=64)

        # Generate new text
        print("--- Generating new text ---")
        if isinstance(seed_text, str):
            tokens = self.dataset._clean_and_tokenize(seed_text)
            seed_ids = [self.dataset.vocab.get(t, self.dataset.vocab[self.dataset.UNK_TOKEN])
                        for t in tokens]
        else:
            seed_ids = seed_text or [seqs[0,0]]
        gen_ids = self.method.generate(seed_ids, max_generate_len)
        generated = [self.dataset.idx2word[i] for i in gen_ids]

        # Package results
        result = {
            'seed_text': seed_text,
            'generated_text': generated
        }

        # Save results
        print("--- Saving generation results ---")
        self.result.data = result
        self.result.save()

        # Plot training loss
        if hasattr(self.method, 'train_losses') and self.method.train_losses:
            plt.figure(figsize=(6,4))
            plt.plot(self.method.train_losses)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f"{self.method.mName} Training Loss")
            plt.grid(True)
            base = os.path.join(self.result.result_destination_folder_path,
                        self.result.result_destination_file_name)
            plt.savefig(f"{base}_gen_loss.png")
            plt.close()

        print("--- Generation complete ---")
        return result
