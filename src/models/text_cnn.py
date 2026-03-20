import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_filters, filter_sizes):
        super(TextCNN, self).__init__()
        # NLP Component: Using dense embeddings for word representation
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # CNN Component: Parallel filters for local n-gram extraction
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, 
                      out_channels=n_filters, 
                      kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, 128)

    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text).permute(0, 2, 1) # [batch_size, embed_dim, seq_len]
        
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        # Concatenate features to form the final content vector
        cat = torch.cat(pooled, dim=1)
        return self.fc(cat)
