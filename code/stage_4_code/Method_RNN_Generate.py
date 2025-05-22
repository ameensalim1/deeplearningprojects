from typing import List, Dict, Any
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from code.base_class.method import method
from code.stage_4_code.Method_RNN import Method_RNN
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy

class Method_RNN_Generate(Method_RNN):
    """
    RNN-based language model for text generation.
    Extends Method_RNN but predicts next-token at each position.
    """
    def __init__(
        self,
        mName: str,
        mDescription: str,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        rnn_type: str = 'LSTM',
        num_rnn_layers: int = 1,
        bidirectional: bool = False,
        dropout_prob_embed: float = 0.5,
        dropout_prob_rnn: float = 0.0,
        dropout_prob_fc: float = 0.5,
        pad_idx: int = 0,
        optimizer_cls=torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] = None,
        loss_fn=None
    ):
        # we treat generation as a next-token classifier over full vocab
        super().__init__(
            mName,
            mDescription,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=vocab_size,
            rnn_type=rnn_type,
            num_rnn_layers=num_rnn_layers,
            bidirectional=bidirectional,
            dropout_prob_embed=dropout_prob_embed,
            dropout_prob_rnn=dropout_prob_rnn,
            dropout_prob_fc=dropout_prob_fc,
            pad_idx=pad_idx,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            loss_fn=loss_fn
        )
        object.__setattr__(self, 'mName', mName)
        object.__setattr__(self, 'mDescription', mDescription)
        # self.device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\") # Removed: Device will be set by .to() or externally
        # self.to(self.device) # Removed: .to() will be called from the main script
        
        # The self.device attribute will be set when .to(device) is called on the instance.
        # Initialize it to None or a default, and it will be updated.
        self.device = None 

        # override num_classes for generation
        self.loss_fn = loss_fn or nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass for generation: returns raw logits for each time step
        x: (batch_size, seq_len)
        returns: (batch_size, seq_len, vocab_size)
        """
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        outputs, _ = self.rnn(embedded)
        dropped = self.fc_dropout(outputs)
        logits = self.fc(dropped)  # (B, T, V)
        return logits

    def train_model(self, X: torch.LongTensor, Y: torch.LongTensor, batch_size: int = 64):
        """Train with teacher forcing on next-token prediction."""
        # Ensure self.device is set correctly (should be by the main script calling .to(device) and setting gen_method.device)
        if self.device is None:
            # Fallback or error if device wasn't set, though it should have been.
            print("Warning: self.device is None in train_model. Defaulting to CPU. This might be incorrect.")
            self.device = torch.device("cpu")
            # Or raise an error: raise RuntimeError("Device not set for Method_RNN_Generate before training.")

        dataset = TensorDataset(X, Y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        optimizer = self.optimizer_cls(self.parameters(), **self.optimizer_kwargs)

        for epoch in range(self.max_epoch):
            self.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.forward(xb)              # (B, T, V)
                B, T, V = logits.shape
                loss = self.loss_fn(logits.view(B*T, V), yb.view(B*T))
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * B
            avg = total_loss / len(loader.dataset)
            self.train_losses.append(avg)
            if epoch % 10 == 0 or epoch == self.max_epoch-1:
                print(f"Epoch {epoch:>3} | Gen Loss: {avg:.4f}")

    def generate(self, start_sequence: List[int], gen_length: int) -> List[int]:
        """
        Autoregressively generate new tokens.
        start_sequence: list of token indices (seed)
        gen_length: how many tokens to append
        returns full sequence of indices
        """
        self.eval()
        seq = start_sequence.copy()
        hidden = None

        with torch.no_grad():
            for _ in range(gen_length):
                # Ensure device is correctly set for generation inputs as well
                if self.device is None:
                    print("Warning: self.device is None in generate. Defaulting to CPU.")
                    self.device = torch.device("cpu")

                inp = torch.LongTensor([seq]).to(self.device)       # (1, len(seq))
                embedded = self.embedding(inp)                       # (1, L, E)
                embedded = self.embed_dropout(embedded)
                if hidden is None:
                    outputs, hidden = self.rnn(embedded)
                else:
                    outputs, hidden = self.rnn(embedded, hidden)
                last_logits = self.fc(self.fc_dropout(outputs[:, -1, :]))  # (1, V)
                next_idx = last_logits.argmax(-1).item()
                seq.append(next_idx)
        return seq
