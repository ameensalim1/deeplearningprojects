from code.base_class.dataset import dataset
import os
import re
from typing import Dict, Optional, Any, List, Tuple
import numpy as np
from collections import Counter
import nltk

try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
except ImportError:
    raise ImportError("Please install NLTK and download resources: nltk.download('punkt'), nltk.download('stopwords')")


class Gen_Dataset_Loader(dataset):
    """
    Concrete IO class for text generation dataset in Stage 4.
    Reads raw text, builds vocabulary, and generates sliding-window sequences.
    """
    data: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    dataset_source_folder_path: str = 'data/stage_4_data/text_generation/'

    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(
        self,
        dName: str,
        dDescription: str,
        max_seq_length: int = 30,
        min_freq: int = 1
    ):
        super().__init__(dName, dDescription)
        self.dName = dName
        self.dDescription = dDescription
        # ensure NLTK data is available
        for resource in ['punkt', 'stopwords']:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource=='punkt' else f'corpora/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)

        self.max_seq_length = max_seq_length
        self.min_freq = min_freq
        self.vocab: Dict[str, int] = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word: Dict[int, str] = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.vocab_size: int = 2
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()

    def _clean_and_tokenize(self, text: str) -> List[str]:
        text = text.lower()
        # keep only letters, numbers and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # simple split on whitespace
        tokens = text.split()
        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words]
        return tokens


    def load(self) -> Dict[str, Dict[str, np.ndarray]]:
        print(f"--- Loading Text Generation Data: {self.dName} ---")
        # 1. Read all text files into one long token list
        all_tokens: List[str] = []
        path = self.dataset_source_folder_path
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Generation data folder not found: {path}")
        
        for fname in os.listdir(path):
            file_path = os.path.join(path, fname)
            # skip directories, hidden files, and docx
            if fname != "data":
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                raw = f.read()
            toks = self._clean_and_tokenize(raw)
            all_tokens.extend(toks)

        if not all_tokens:
            raise RuntimeError("No text loaded for generation dataset.")

        # 2. Build vocabulary from tokens
        counter = Counter(all_tokens)
        idx = len(self.vocab)
        for word, cnt in counter.items():
            if cnt >= self.min_freq and word not in self.vocab:
                self.vocab[word] = idx
                self.idx2word[idx] = word
                idx += 1
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary size for generation: {self.vocab_size}")

        # 3. Create sliding-window sequences
        X_seqs = []
        Y_seqs = []
        N = self.max_seq_length
        for i in range(len(all_tokens) - N):
            inp = all_tokens[i : i + N]
            targ = all_tokens[i + 1 : i + N + 1]
            X_seqs.append([ self.vocab.get(w, self.vocab[self.UNK_TOKEN]) for w in inp ])
            Y_seqs.append([ self.vocab.get(w, self.vocab[self.UNK_TOKEN]) for w in targ ])
        X = np.array(X_seqs, dtype=np.int32)
        Y = np.array(Y_seqs, dtype=np.int32)

        # We won't use a separate test split here, so leave it empty
        self.data = {
            'train': {'X': X, 'y': Y},
            'test':  {'X': np.array([], dtype=np.int32), 'y': np.array([], dtype=np.int32)}
        }
        print(f"  Generated {X.shape[0]} training examples of length {N}.")
        return self.data
