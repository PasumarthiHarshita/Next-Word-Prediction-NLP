from collections import defaultdict, Counter

class NGramModel:
    def __init__(self, n):
        self.n = n
        self.model = defaultdict(Counter)

    def train(self, sentences):
        for sentence in sentences:
            words = sentence.lower().split()
            for i in range(len(words) - self.n + 1):
                context = tuple(words[i:i+self.n-1])
                target = words[i+self.n-1]
                self.model[context][target] += 1

    def predict_next(self, text):
        words = text.lower().split()
        if len(words) < self.n - 1:
            return "Not enough words"

        context = tuple(words[-(self.n-1):])

        if context not in self.model:
            return "No prediction available"

        return self.model[context].most_common(1)[0][0]

    def predict_with_probabilities(self, text):
        words = text.lower().split()
        context = tuple(words[-(self.n-1):])

        if context not in self.model:
            return {}

        total = sum(self.model[context].values())
        return {
            word: count / total
            for word, count in self.model[context].items()
        }
