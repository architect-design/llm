import torch
import torch.nn as nn


class TinyFinancialSLM(nn.Module):
    """
    A simple LSTM-based Language Model for generating structured financial JSON data.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TinyFinancialSLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        output = self.fc(lstm_out)
        return output, hidden