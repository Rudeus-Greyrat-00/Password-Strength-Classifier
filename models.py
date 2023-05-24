from torch import nn

class SimpleTextClassificationModel(nn.Module):
    """
    Copied from here
    https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
    """
    def __init__(self, vocab_size, embed_dim, num_class):
        super(SimpleTextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)