import torch
import torch.nn as nn
import torch.nn.functional as F


class UserFiLM(nn.Module):

    def __init__(self, user_dim: int, feature_dim: int):
        super().__init__()
        self.gen = nn.Linear(user_dim, feature_dim * 2)

    def forward(self, x, user_emb):
        params = self.gen(user_emb).unsqueeze(1)
        gamma, beta = params.chunk(2, dim=-1)
        return (1 + gamma) * x + beta


class TransformerUserModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        area_vocab_size: int,
        embedding_dim: int = 128,
        user_embedding_dim: int = 64,
        area_embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        nhead: int = 2,
        max_seq_len: int = 100,
        num_users: int = 5619,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.user_emb = nn.Embedding(num_users, user_embedding_dim)
        self.area_emb = nn.Embedding(area_vocab_size, area_embedding_dim)
        self.hour_emb = nn.Embedding(24, 32)

        self.max_seq_len = max_seq_len

        self.feature_fusion = nn.Linear(embedding_dim + 32, embedding_dim)
        self.input_cell_dim = embedding_dim
        self.pos_emb_cell = nn.Embedding(max_seq_len, self.input_cell_dim)

        encoder_layer_cell = nn.TransformerEncoderLayer(
            d_model=self.input_cell_dim,
            nhead=nhead,
            dim_feedforward=self.input_cell_dim,
            activation="gelu",
            batch_first=True,
            dropout=dropout,
        )
        self.transformer_cell = nn.TransformerEncoder(
            encoder_layer_cell,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.input_cell_dim),
        )
        self.proj_cell = nn.Linear(self.input_cell_dim, hidden_dim)

        self.feature_fusion_area = nn.Linear(area_embedding_dim + 32, area_embedding_dim)
        self.input_area_dim = area_embedding_dim
        self.pos_emb_area = nn.Embedding(max_seq_len, self.input_area_dim)

        encoder_layer_area = nn.TransformerEncoderLayer(
            d_model=self.input_area_dim,
            nhead=nhead,
            dim_feedforward=self.input_area_dim,
            activation="gelu",
            batch_first=True,
            dropout=dropout,
        )
        self.transformer_area = nn.TransformerEncoder(
            encoder_layer_area,
            num_layers=num_layers,
            norm=nn.LayerNorm(self.input_area_dim),
        )
        self.proj_area = nn.Linear(self.input_area_dim, hidden_dim)

        self.film_cell = UserFiLM(user_embedding_dim, embedding_dim)
        self.film_area = UserFiLM(user_embedding_dim, area_embedding_dim)

        self.final_norm_cell = nn.LayerNorm(hidden_dim * 2)
        self.final_norm_area = nn.LayerNorm(hidden_dim)

        self.fc_cell = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

        self.fc_area = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, area_vocab_size),
        )

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(self, "film_cell"):
            nn.init.zeros_(self.film_cell.gen.weight)
            nn.init.zeros_(self.film_cell.gen.bias)
        if hasattr(self, "film_area"):
            nn.init.zeros_(self.film_area.gen.weight)
            nn.init.zeros_(self.film_area.gen.bias)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))


    def forward(
        self,
        input_ids,
        input_area,
        user_ids,
        input_hour,
    ):

        bsz, seq_len = input_ids.size()
        device = input_ids.device

        tok = self.token_emb(input_ids)   
        area = self.area_emb(input_area) 
        hour_emb = self.hour_emb(input_hour) 
        u_emb = self.user_emb(user_ids)       

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        causal_mask = self._causal_mask(seq_len, device)

        x = self.feature_fusion(torch.cat([tok, hour_emb], dim=-1))
        x = x + self.pos_emb_cell(positions)
        x_out = self.transformer_cell(x, mask=causal_mask)
        x_out = self.film_cell(x_out, u_emb)
        hidden_cell = self.proj_cell(x_out)

        y = self.feature_fusion_area(torch.cat([area, hour_emb], dim=-1))
        y = y + self.pos_emb_area(positions)
        y_out = self.transformer_area(y, mask=causal_mask)
        y_out = self.film_area(y_out, u_emb)
        hidden_area = self.proj_area(y_out)

        area_logits = self.fc_area(self.final_norm_area(hidden_area))
        cell_logits = self.fc_cell(
            self.final_norm_cell(torch.cat([hidden_cell, hidden_area], dim=-1))
        )

        return F.log_softmax(cell_logits, dim=-1), F.log_softmax(area_logits, dim=-1)
