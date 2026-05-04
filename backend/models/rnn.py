import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np


CORPUS = [
    "hello how are you",
    "hello how are you doing",
    "hello how are you today",
    "hello how are we learning",
    "how are you doing today",
    "how are you feeling today",
    "we are learning sequence models",
    "we are learning deep learning",
    "sequence models predict next words",
    "rnn models learn from previous words",
    "deep learning models learn patterns",
    "students learn sequence learning by prediction",
    "next word prediction shows sequence learning",
]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


class SimpleRNNLanguageModel:
    def __init__(self, corpus: List[str], hidden_size: int = 28, seed: int = 7) -> None:
        tokenized = [tokenize(line) for line in corpus]
        vocabulary = sorted({word for line in tokenized for word in line})
        self.vocabulary = ["<unk>"] + vocabulary
        self.word_to_index = {word: index for index, word in enumerate(self.vocabulary)}
        self.hidden_size = hidden_size
        self.rng = np.random.default_rng(seed)

        vocab_size = len(self.vocabulary)
        self.wxh = self.rng.normal(0, 0.12, (hidden_size, vocab_size))
        self.whh = self.rng.normal(0, 0.12, (hidden_size, hidden_size))
        self.why = self.rng.normal(0, 0.12, (vocab_size, hidden_size))
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

        self.training_samples = self._build_training_samples(tokenized)
        self.ngrams = self._build_ngram_counts(tokenized)
        self._train(epochs=360, learning_rate=0.045)

    def _build_training_samples(self, tokenized: List[List[str]]) -> List[Tuple[List[int], int]]:
        samples: List[Tuple[List[int], int]] = []
        for line in tokenized:
            indexes = [self.word_to_index[word] for word in line]
            for end in range(1, len(indexes)):
                samples.append((indexes[:end], indexes[end]))
        return samples

    def _build_ngram_counts(self, tokenized: List[List[str]]) -> Dict[Tuple[str, ...], Counter]:
        counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        for line in tokenized:
            for index in range(1, len(line)):
                for size in range(1, min(3, index) + 1):
                    prefix = tuple(line[index - size:index])
                    counts[prefix][line[index]] += 1
        return counts

    def _one_hot(self, index: int) -> np.ndarray:
        vector = np.zeros((len(self.vocabulary), 1))
        vector[index] = 1.0
        return vector

    def _forward(self, indexes: List[int]):
        hidden_states = {-1: np.zeros((self.hidden_size, 1))}
        inputs = {}
        for time_step, index in enumerate(indexes):
            inputs[time_step] = self._one_hot(index)
            hidden_states[time_step] = np.tanh(
                self.wxh @ inputs[time_step] + self.whh @ hidden_states[time_step - 1] + self.bh
            )
        logits = self.why @ hidden_states[len(indexes) - 1] + self.by
        logits -= np.max(logits)
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        return inputs, hidden_states, probabilities

    def _train(self, epochs: int, learning_rate: float) -> None:
        for _ in range(epochs):
            self.rng.shuffle(self.training_samples)
            for indexes, target in self.training_samples:
                inputs, hidden_states, probabilities = self._forward(indexes)
                dy = probabilities.copy()
                dy[target] -= 1

                dwhy = dy @ hidden_states[len(indexes) - 1].T
                dby = dy
                dwxh = np.zeros_like(self.wxh)
                dwhh = np.zeros_like(self.whh)
                dbh = np.zeros_like(self.bh)
                dh_next = self.why.T @ dy

                for time_step in reversed(range(len(indexes))):
                    dh_raw = (1 - hidden_states[time_step] * hidden_states[time_step]) * dh_next
                    dbh += dh_raw
                    dwxh += dh_raw @ inputs[time_step].T
                    dwhh += dh_raw @ hidden_states[time_step - 1].T
                    dh_next = self.whh.T @ dh_raw

                for gradient in (dwxh, dwhh, dwhy, dbh, dby):
                    np.clip(gradient, -4, 4, out=gradient)

                self.wxh -= learning_rate * dwxh
                self.whh -= learning_rate * dwhh
                self.why -= learning_rate * dwhy
                self.bh -= learning_rate * dbh
                self.by -= learning_rate * dby

    def _indexes_for(self, words: List[str]) -> List[int]:
        unknown = self.word_to_index["<unk>"]
        return [self.word_to_index.get(word, unknown) for word in words]

    def _ngram_boosts(self, words: List[str]) -> Dict[str, float]:
        boosts: Dict[str, float] = {}
        for size, weight in ((3, 0.72), (2, 0.46), (1, 0.24)):
            if len(words) < size:
                continue
            counter = self.ngrams.get(tuple(words[-size:]))
            if not counter:
                continue
            total = sum(counter.values())
            for word, count in counter.items():
                boosts[word] = max(boosts.get(word, 0.0), weight * count / total)
            break
        return boosts

    def predict_next(self, text: str, top_k: int = 3) -> Dict[str, object]:
        words = tokenize(text)
        if not words:
            words = ["hello"]

        indexes = self._indexes_for(words)
        _, hidden_states, probabilities = self._forward(indexes)
        scores = probabilities.flatten()

        for word, boost in self._ngram_boosts(words).items():
            scores[self.word_to_index[word]] += boost

        scores[self.word_to_index["<unk>"]] = 0.0
        scores = scores / np.sum(scores)
        ranked = np.argsort(scores)[::-1][:top_k]
        predictions = [
            {"word": self.vocabulary[index], "confidence": round(float(scores[index]), 4)}
            for index in ranked
        ]

        hidden = hidden_states[len(indexes) - 1].flatten()
        return {
            "input": text,
            "tokens": words,
            "prediction": predictions[0]["word"],
            "predictions": predictions,
            "hidden_state_preview": [round(float(value), 4) for value in hidden[:8]],
            "model": "Simple tanh RNN language model trained on a small educational corpus",
            "concept": "The RNN reads words in order, updates a hidden state after each token, and uses that final state to score the next word.",
        }


MODEL = SimpleRNNLanguageModel(CORPUS)
