'''
Concrete IO class for text classification dataset in Stage 4.
Loads and preprocesses text data (movie reviews).
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import os
import re
from typing import Dict, Optional, Any, List, Tuple, Set
import numpy as np
from collections import Counter
import nltk
# It's good practice to try importing and provide a message if NLTK components are missing.
try:
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    # from nltk.stem import PorterStemmer # Optional: for stemming
except ImportError:
    print("NLTK not found or specific modules missing. Please install NLTK and download 'punkt' and 'stopwords'.")
    print("Run: pip install nltk")
    print("Then in Python: import nltk; nltk.download('punkt'); nltk.download('stopwords')")
    # Depending on strictness, you might raise an error or allow the program to continue if NLTK isn't critical for all paths.

# Define a type alias for vocabulary for clarity
Word2IdxType = Dict[str, int]
Idx2WordType = Dict[int, str]


class Dataset_Loader(dataset):
    # --- Re-defined attributes for text data ---
    data: Optional[Dict[str, Dict[str, np.ndarray]]] = None # {'train': {'X': text_sequences, 'y': labels}, 'test': ...}
    dataset_source_folder_path: str = 'data/stage_4_data/text_classification/'
    
    vocab: Optional[Word2IdxType] = None
    idx2word: Optional[Idx2WordType] = None
    vocab_size: Optional[int] = None
    max_seq_length: int = 200 # Default, can be adjusted or determined from data
    
    # Special tokens
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>" # For words not in vocab during test time

    # NLTK resources (initialized in __init__ or a helper method)
    stop_words: Optional[Set[str]] = None
    # stemmer: Optional[PorterStemmer] = None # Optional

    def __init__(self, dName: str, dDescription: str, max_seq_length: int = 200):
        super().__init__(dName, dDescription)
        self.dName = dName
        self.dDescription = dDescription

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)

        self.max_seq_length = max_seq_length
        

        # now safely load stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            self.stop_words = set()
        # Initialize vocab and other attributes that will be built during load()
        self.vocab = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1} # Initialize with PAD and UNK
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.vocab_size = 2 # Start with PAD and UNK

    def _clean_text(self, text: str) -> List[str]:
        """
        Cleans a single text string:
        1. Lowercase
        2. Remove punctuation (keeps alphanumeric and spaces)
        3. Tokenize
        4. Remove stopwords
        5. Optional: Stemming (if self.stemmer is enabled)
        """
        text = text.lower()
        # Remove punctuation - keep letters, numbers, and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        try:
            tokens = word_tokenize(text)
        except (LookupError, OSError):
            # If NLTK’s punkt isn’t available, fall back to simple split
            tokens = text.split()
        
        # Remove stopwords
        if self.stop_words: # self.stop_words should be initialized in __init__
            tokens = [token for token in tokens if token not in self.stop_words and token.strip()] # also remove empty strings

        # Optional: Stemming
        # if self.stemmer:
        #     tokens = [self.stemmer.stem(token) for token in tokens]
            
        return tokens

    def _load_split_data(self, split_name: str) -> Tuple[List[List[str]], List[int]]:
        """
        Loads all text files for a given split ('train' or 'test'),
        cleans them, and returns tokenized texts and labels.
        """
        all_tokenized_texts: List[List[str]] = []
        all_labels: List[int] = []
        
        split_path = os.path.join(self.dataset_source_folder_path, split_name) # e.g., data/stage_4_data/text_classification/train
        
        for sentiment_label, sentiment_str in enumerate(['neg', 'pos']): # 0 for neg, 1 for pos
            sentiment_path = os.path.join(split_path, sentiment_str)
            if not os.path.isdir(sentiment_path):
                print(f"Warning: Directory not found {sentiment_path}")
                continue
                
            for filename in os.listdir(sentiment_path):
                if filename.endswith(".txt"):
                    file_path = os.path.join(sentiment_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            raw_text = f.read()
                        
                        cleaned_tokens = self._clean_text(raw_text)
                        if cleaned_tokens: # Only add if cleaning resulted in some tokens
                            all_tokenized_texts.append(cleaned_tokens)
                            all_labels.append(sentiment_label)
                    except Exception as e:
                        print(f"Error reading or processing file {file_path}: {e}")
                        
        return all_tokenized_texts, all_labels

    def _build_vocab(self, all_tokenized_train_texts: List[List[str]], min_freq: int = 5):
        """
        Builds the vocabulary from all tokenized training texts.
        Filters words by minimum frequency.
        Updates self.vocab, self.idx2word, and self.vocab_size.
        Assumes PAD_TOKEN and UNK_TOKEN are already in self.vocab at index 0 and 1.
        """
        token_counts = Counter()
        for tokens in all_tokenized_train_texts:
            token_counts.update(tokens)
        
        # Start adding words to vocab from index 2 (0=PAD, 1=UNK) 
        current_idx = len(self.vocab) 
        for token, count in token_counts.items():
            if count >= min_freq:
                if token not in self.vocab: # Ensure not to overwrite PAD/UNK or add duplicates
                    self.vocab[token] = current_idx
                    self.idx2word[current_idx] = token
                    current_idx += 1
        
        self.vocab_size = len(self.vocab)
        print(f"Vocabulary built. Size: {self.vocab_size} (including PAD & UNK). Filtered by min_freq={min_freq}.")

    def _texts_to_padded_sequences(self, tokenized_texts: List[List[str]]) -> np.ndarray:
        """
        Converts a list of tokenized texts to a 2D numpy array of padded integer sequences.
        """
        sequences = []
        for tokens in tokenized_texts:
            seq = [self.vocab.get(token, self.vocab[self.UNK_TOKEN]) for token in tokens]
            
            # Pad or truncate
            if len(seq) < self.max_seq_length:
                seq.extend([self.vocab[self.PAD_TOKEN]] * (self.max_seq_length - len(seq)))
            else:
                seq = seq[:self.max_seq_length]
            sequences.append(seq)
            
        return np.array(sequences, dtype=np.int32) # Usually int32 for embedding layers

    # --- Helper methods for text processing will go here ---
    # _clean_text(self, text: str) -> List[str]: # This line is a placeholder comment and will be effectively replaced
    # _build_vocab(self, all_tokens: List[List[str]], min_freq: int = 5): # This line is a placeholder comment and will be effectively replaced
    # _texts_to_padded_sequences(self, tokenized_texts: List[List[str]]) -> np.ndarray: # This line is a placeholder comment and will be effectively replaced

    # --- Main load() method will be re-implemented ---
    # def load(self) -> Dict[str, Dict[str, np.ndarray]]:

    def load(self) -> Dict[str, Dict[str, np.ndarray]]:
        print(f'--- Loading Text Classification Data: {self.dName} ---')

        # 1. Load raw tokenized texts and labels for train and test splits
        print("Loading and cleaning training data...")
        train_tokenized_texts, y_train_list = self._load_split_data('train')
        if not train_tokenized_texts:
            print("ERROR: No training data loaded. Check paths and data.")
            self.data = {'train': {'X': np.array([]), 'y': np.array([])}, 
                         'test': {'X': np.array([]), 'y': np.array([])}}
            return self.data
        
        print("Loading and cleaning testing data...")
        test_tokenized_texts, y_test_list = self._load_split_data('test')
        if not test_tokenized_texts:
            print("Warning: No testing data loaded. Proceeding with training data only for vocab building.")
            # We can still build vocab from train, but test set will be empty.

        # 2. Build vocabulary using ONLY training texts
        # Ensure vocab is fresh if load is called multiple times (though typically not)
        self.vocab = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1} 
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.vocab_size = 2
        self._build_vocab(train_tokenized_texts, min_freq=5) # Using a min_freq of 5 as an example

        # 3. Convert tokenized texts to padded sequences
        print("Converting training texts to sequences...")
        X_train_seq = self._texts_to_padded_sequences(train_tokenized_texts)
        y_train = np.array(y_train_list, dtype=np.int64)

        X_test_seq = np.array([]) # Initialize as empty
        y_test = np.array([])    # Initialize as empty
        if test_tokenized_texts: # Only process if test data was loaded
            print("Converting testing texts to sequences...")
            X_test_seq = self._texts_to_padded_sequences(test_tokenized_texts)
            y_test = np.array(y_test_list, dtype=np.int64)
        
        self.data = {
            'train': {'X': X_train_seq, 'y': y_train},
            'test': {'X': X_test_seq, 'y': y_test}
        }

        print(f"  Training X shape: {X_train_seq.shape}, y shape: {y_train.shape}")
        if X_test_seq.size > 0:
            print(f"  Testing X shape: {X_test_seq.shape}, y shape: {y_test.shape}")
        print(f"  Max sequence length: {self.max_seq_length}")
        print(f"  Vocabulary size: {self.vocab_size}")
        print('--- Text Data Loading Complete ---')
        return self.data

    # The old _process_pickle_split method is no longer needed for text data.
    # It can be removed. If you need to support both image and text datasets
    # with the same loader, you'd need a more complex conditional logic,
    # but for this stage, focusing on text is appropriate.