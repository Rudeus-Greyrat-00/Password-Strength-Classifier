from torch import nn


class SimpleTextClassificationModel(nn.Module):
    """
    Copied from here
    https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
    """

    def __init__(self, vocab_size, embed_dim, num_class):
        super(SimpleTextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc1 = self._layer_fully_connected(embed_dim, num_class)
        # self.fc2 = self._layer_fully_connected(embed_dim, num_class)

    def init_weights(self, layer, initrange=0.5):
        self.embedding.weight.data.uniform_(-initrange, initrange)
        layer.weight.data.uniform_(-initrange, initrange)
        layer.bias.data.zero_()

    def _layer_fully_connected(self, c_in, c_out):
        nnLinear = nn.Linear(c_in, c_out)
        self.init_weights(nnLinear)
        layer = nn.Sequential(
            nnLinear,
            # nn.ReLU(),
            # nn.Dropout(),
        )
        return layer

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded = self.fc1(embedded)
        return embedded
