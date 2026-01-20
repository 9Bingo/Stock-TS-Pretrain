import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttentionLayer, self).__init__()
        assert hidden_size % num_heads == 0, "Hidden size must be divisible by the number of heads."
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.fc_q = nn.Linear(hidden_size, hidden_size)
        self.fc_k = nn.Linear(hidden_size, hidden_size)
        self.fc_v = nn.Linear(hidden_size, hidden_size)
        self.fc_o = nn.Linear(hidden_size, hidden_size)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim).float().to(query.device))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = F.softmax(energy, dim=-1)
        output = torch.matmul(attention, V)
        output = output.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hidden_size)
        output = self.fc_o(output)

        return output


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hidden_size, ff_hidden_size):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(hidden_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, hidden_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FeatExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_feat_att_layers, num_heads, days, dropout=0.1):
        super(FeatExtractor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_feat_att_layers = num_feat_att_layers
        self.num_heads = num_heads
        self.days = days

        self.embedding = nn.Linear(input_size, hidden_size)
        # Learnable Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, days, hidden_size))
        
        self.attention_layers = nn.ModuleList(
            [SelfAttentionLayer(hidden_size, num_heads) for _ in range(num_feat_att_layers)])
        self.feedforward_layers = nn.ModuleList(
            [PositionwiseFeedforward(hidden_size, hidden_size * 4) for _ in range(num_feat_att_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        embedded = self.embedding(x)
        # Add Positional Encoding
        if x.size(1) <= self.days:
             embedded += self.positional_encoding[:, :x.size(1), :]
        else:
             # Fallback if input is longer than preset days (safety)
             embedded += self.positional_encoding[:, :self.days, :]

        for i in range(self.num_feat_att_layers):
            attention_output = self.attention_layers[i](embedded, embedded, embedded)
            embedded = embedded + self.dropout(attention_output)
            feedforward_output = self.feedforward_layers[i](embedded)
            embedded = embedded + self.dropout(feedforward_output)

        return embedded


class TransformerStockPrediction(nn.Module):
    def __init__(self, input_size, num_class, hidden_size, num_feat_att_layers, num_pre_att_layers, num_heads, days, dropout=0.1):
        super(TransformerStockPrediction, self).__init__()
        self.hidden_size = hidden_size
        self.pretrain_task = '' # Default: Empty string means Fine-tuning task
        self.pretrain_outlayers = nn.ModuleDict({})
        self.finetune = False # Controls freezing

        # 1. Feature Extractor (Bottom layers)
        self.feature_extractor = FeatExtractor(input_size, hidden_size, num_feat_att_layers, num_heads, days, dropout)
        
        # 2. Pre-training Attention Layers (Upper layers)
        self.attention_layers = nn.ModuleList(
            [SelfAttentionLayer(hidden_size, num_heads) for _ in range(num_pre_att_layers)])
        self.feedforward_layers = nn.ModuleList(
            [PositionwiseFeedforward(hidden_size, hidden_size * 4) for _ in range(num_pre_att_layers)])
        
        # 3. Default Output Layer (For Fine-tuning/Price Prediction)
        self.fc = nn.Linear(hidden_size, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        embedded = self.feature_extractor(x)

        for i in range(len(self.attention_layers)):
            attention_output = self.attention_layers[i](embedded, embedded, embedded)
            embedded = embedded + self.dropout(attention_output)
            feedforward_output = self.feedforward_layers[i](embedded)
            embedded = embedded + self.dropout(feedforward_output)

        # Mean Pooling
        pooled = torch.mean(embedded, dim=1)

        # Task Switching Logic
        if self.pretrain_task == '':
            # Default task (Fine-tuning: Price Prediction)
            output = self.fc(pooled)
        else:
            # Pre-training tasks (e.g., Stock Classification)
            output = self.pretrain_outlayers[self.pretrain_task](pooled)

        return output

    def add_outlayer(self, name, num_class, device):
        """Dynamically add a classification head for pre-training"""
        if name not in self.pretrain_outlayers:
            self.pretrain_outlayers[name] = nn.Linear(self.hidden_size, num_class).to(device)
            print(f"Added new output layer: {name}")

    def change_finetune_mode(self, mode, freezing='embedding'):
        """Control which layers are frozen"""
        self.finetune = mode
        if mode:
            if freezing == 'all': # Freeze everything except head
                for param in self.feature_extractor.parameters():
                    param.requires_grad = False
                for param in self.attention_layers.parameters():
                    param.requires_grad = False
                for param in self.feedforward_layers.parameters():
                    param.requires_grad = False
            elif freezing == 'embedding':
                for param in self.feature_extractor.embedding.parameters():
                    param.requires_grad = False
        else:
            # Unfreeze all
            for param in self.parameters():
                param.requires_grad = True
