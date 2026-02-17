from typing import Any

import torch
import torch.nn as nn

from build_llm.layer import LayerNorm
from build_llm.transformer import TransformerBlock


class GPTModel(nn.Module):
    """
    Implementação de um modelo GPT (decoder-only Transformer).

    Componentes:
    -----------
    - Embedding de tokens (tok_emb): converte IDs de tokens em vetores.
    - Embedding posicional (pos_emb): adiciona informação de posição na sequência.
    - Dropout (drop_emb): regularização nos embeddings somados.
    - Blocos Transformer (trf_blocks): pilha de TransformerBlock.
    - Normalização final (final_norm): LayerNorm antes da projeção final.
    - Cabeça de saída (out_head): projeção para logits no vocabulário.

    Parâmetros esperados em `cfg`:
    ------------------------------
    vocab_size : int
        Tamanho do vocabulário (número de tokens possíveis).
    emb_dim : int
        Dimensão dos embeddings / hidden size.
    context_length : int
        Comprimento máximo de contexto (seq_len máximo).
    drop_rate : float
        Taxa de dropout aplicada após a soma tok_emb + pos_emb.
    n_layers : int
        Número de blocos Transformer.

    Retorno do forward:
    -------------------
    torch.Tensor
        Logits com shape (batch_size, seq_len, vocab_size).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Executa o forward pass do GPT.

        Parâmetros:
        ----------
        in_idx : torch.Tensor
            Tensor de IDs de tokens com shape (batch_size, seq_len) e dtype inteiro.

        Retorno:
        -------
        torch.Tensor
            Logits com shape (batch_size, seq_len, vocab_size).

        Exceções:
        --------
        Levanta ValueError se `in_idx` não for um tensor 2D.
        """
        if not isinstance(in_idx, torch.Tensor):
            raise TypeError("`in_idx` deve ser um torch.Tensor.")
        if in_idx.ndim != 2:
            raise ValueError("`in_idx` deve ter shape (batch_size, seq_len).")

        batch_size, seq_len = in_idx.shape  # batch_size não é usado diretamente aqui

        tok_embeds = self.tok_emb(in_idx)  # (B, T, C)

        pos_ids = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_ids)  # (T, C) -> broadcast p/ (B, T, C)

        x = tok_embeds + pos_embeds  # (B, T, C)
        x = self.drop_emb(x)

        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)  # (B, T, vocab_size)
        return logits
