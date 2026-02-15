import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Implementa Multi-Head Causal Self-Attention (estilo Transformer) com máscara causal
    para impedir que cada token atenda tokens futuros.

    Parâmetros:
    ----------
    d_in : int
        Dimensão de entrada (features por token).
    d_out : int
        Dimensão total de saída (soma das dimensões de todas as heads).
        Deve ser divisível por num_heads.
    context_length : int
        Comprimento máximo de contexto (número máximo de tokens).
    dropout : float
        Probabilidade de dropout aplicada aos pesos de atenção.
    num_heads : int
        Número de cabeças de atenção.
    qkv_bias : bool, default = False
        Se True, adiciona bias nas camadas lineares de Q, K e V.

    Retorno:
    -------
    torch.Tensor
        Tensor de shape (batch_size, num_tokens, d_out) com os vetores de contexto.

    Exceções:
    --------
    Levanta ValueError se d_out não for divisível por num_heads.
    Levanta ValueError se x não tiver shape 3D (B, T, D).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()

        if d_out % num_heads != 0:
            raise ValueError("d_out must be divisible by num_heads")

        self.d_out = int(d_out)
        self.num_heads = int(num_heads)
        self.head_dim = self.d_out // self.num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Projeção final para combinar as heads
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        # Máscara causal: 1s acima da diagonal principal (tokens futuros).
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Esperado x com 3 dimensões (batch, tokens, d_in), mas veio shape={tuple(x.shape)}."
            )

        b, num_tokens, d_in = x.shape

        # Projeções: (b, T, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # (b, T, d_out) -> (b, T, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # (b, T, num_heads, head_dim) -> (b, num_heads, T, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scores por head: (b, h, T, head_dim) @ (b, h, head_dim, T) -> (b, h, T, T)
        attn_scores = queries @ keys.transpose(2, 3)

        # Aplicar máscara causal (broadcast nas heads e batch)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Softmax com escala por sqrt(d_k)
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Contexto por head: (b, h, T, T) @ (b, h, T, head_dim) -> (b, h, T, head_dim)
        context = attn_weights @ values

        # (b, h, T, head_dim) -> (b, T, h, head_dim)
        context = context.transpose(1, 2)

        # Concatenar heads: (b, T, d_out)
        context = context.contiguous().view(b, num_tokens, self.d_out)

        # Projeção final
        context = self.out_proj(context)

        return context
