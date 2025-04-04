class Vocabulary:
    def __init__(self):
        # Initialize the vocabulary
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab_size = 0

    def add_word(self, word):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.vocab_size
            self.index_to_word[self.vocab_size] = word
            self.vocab_size += 1

    def get_index(self, word):
        return self.word_to_index.get(word, None)

    def get_word(self, index):
        return self.index_to_word.get(index, None)

    def __len__(self):
        return self.vocab_size
