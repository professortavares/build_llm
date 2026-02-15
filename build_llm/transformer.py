import torch
from torch import nn

from build_llm.attention import MultiHeadAttention
from build_llm.layer import LayerNorm
from build_llm.mlp import FeedForward


class TransformerBlock(nn.Module):
    """
    Bloco Transformer (estilo GPT) com:
    - Multi-Head Self-Attention
    - MLP / FeedForward
    - Pre-LayerNorm (LayerNorm antes de cada sub-bloco)
    - Conexões residuais (shortcut) e dropout nas saídas dos sub-blocos

    Fluxo (simplificado):
    ---------------------
    x -> LN -> MHA -> Dropout -> +residual
      -> LN -> FFN -> Dropout -> +residual

    Parâmetros:
    ----------
    cfg : dict
        Dicionário de configuração contendo:
        - "emb_dim" (int): dimensão do embedding (d_model).
        - "context_length" (int): tamanho máximo de contexto (n_tokens).
        - "n_heads" (int): número de cabeças de atenção.
        - "drop_rate" (float): taxa de dropout (0.0 a 1.0).
        - "qkv_bias" (bool): se deve usar bias em QKV no attention.

    Exceções:
    --------
    Levanta KeyError se alguma chave obrigatória não existir em cfg.
    Levanta TypeError/ValueError se algum valor não puder ser convertido
    para o tipo esperado ou se estiver fora de um intervalo válido.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()

        required_keys = [
            "emb_dim",
            "context_length",
            "n_heads",
            "drop_rate",
            "qkv_bias",
        ]
        missing = [k for k in required_keys if k not in cfg]
        if missing:
            raise KeyError(f"cfg está faltando as chaves obrigatórias: {missing}")

        # Validações / conversões defensivas
        try:
            emb_dim = int(cfg["emb_dim"])
            context_length = int(cfg["context_length"])
            n_heads = int(cfg["n_heads"])
            drop_rate = float(cfg["drop_rate"])
            qkv_bias = bool(cfg["qkv_bias"])
        except (TypeError, ValueError) as e:
            raise TypeError(
                "cfg contém valores em formato inválido para o TransformerBlock."
            ) from e

        if emb_dim <= 0:
            raise ValueError('"emb_dim" deve ser um inteiro positivo.')
        if context_length <= 0:
            raise ValueError('"context_length" deve ser um inteiro positivo.')
        if n_heads <= 0:
            raise ValueError('"n_heads" deve ser um inteiro positivo.')
        if emb_dim % n_heads != 0:
            raise ValueError('"emb_dim" deve ser divisível por "n_heads".')
        if not (0.0 <= drop_rate <= 1.0):
            raise ValueError('"drop_rate" deve estar entre 0.0 e 1.0.')

        self.att = MultiHeadAttention(
            d_in=emb_dim,
            d_out=emb_dim,
            context_length=context_length,
            num_heads=n_heads,
            dropout=drop_rate,
            qkv_bias=qkv_bias,
        )

        self.ff = FeedForward(cfg)

        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)

        # Dropout aplicado na saída de cada sub-bloco antes de somar o residual
        self.drop_shortcut = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executa o forward pass do bloco Transformer.

        Parâmetros:
        ----------
        x : torch.Tensor
            Tensor de entrada com shape [batch_size, num_tokens, emb_dim].

        Retorno:
        -------
        torch.Tensor
            Tensor de saída com o mesmo shape [batch_size, num_tokens, emb_dim].
        """
        # Residual + Attention
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # [batch, tokens, emb_dim]
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Residual + FeedForward
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)  # [batch, tokens, emb_dim]
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
