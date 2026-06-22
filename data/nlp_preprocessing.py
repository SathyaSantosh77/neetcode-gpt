import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        all_sentences = positive + negative
        unique_words = sorted(list(set(" ".join(all_sentences).split())))
        # 2. Encode each sentence by replacing words with their IDs
        vocab = {word: i for i, word in enumerate(unique_words, 1)}
        tensors = []
        # 3. Combine positive + negative into one list of tensors
        for sentence in all_sentences:
            s = [float(vocab[word]) for word in sentence.split()]
            tensors.append(torch.tensor(s))
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        dataset_tensor = nn.utils.rnn.pad_sequence(
            tensors,
            batch_first = True,
            padding_value = 0.0
        )

        return dataset_tensor