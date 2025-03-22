import re
import numpy as np
import torch


class Text_Tokenizer:

    def __init__(self, max_vocab_size, device):
        self.max_vocab_size = max_vocab_size
        self.device = device
        self.id2word = ["<PAD>", "<UNK>"]
        self.word2id = {w: i for i, w in enumerate(self.id2word)}

    def _get_word_id(self, word):
        if word not in self.word2id:
            if len(self.word2id) >= self.max_vocab_size:
                return self.word2id["<UNK>"]

            self.id2word.append(word)
            self.word2id[word] = len(self.word2id)

        return self.word2id[word]

    def _tokenize(self, text):
        # Simple tokenizer: strip out all non-alphabetic characters.
        text = re.sub("[^a-zA-Z0-9\- ]", " ", text)
        word_ids = list(map(self._get_word_id, text.split()))
        return word_ids
    
    def __len__(self):
        return len(self.id2word)
    
    def __call__(self, texts):
        texts = list(map(self._tokenize, texts))
        max_len = max(len(l) for l in texts)
        padded = np.ones((len(texts), max_len)) * self.word2id["<PAD>"]

        for i, text in enumerate(texts):
            padded[i, :len(text)] = text

        padded_tensor = torch.from_numpy(padded).type(torch.long).to(self.device)
        return padded_tensor