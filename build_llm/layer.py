import torch
from torch import nn


class LayerNorm(nn.Module):
    """
    Implementação de Layer Normalization (normalização por token, ao longo da última dimensão).

    Esta camada normaliza as ativações ao longo da dimensão de embedding (dim=-1),
    e então aplica uma transformação afim aprendível:

        y = scale * (x - mean) / sqrt(var + eps) + shift

    Parâmetros:
    ----------
    emb_dim : int
        Dimensão do embedding (tamanho do último eixo de `x`).
    eps : float, default = 1e-5
        Constante pequena para evitar divisão por zero.

    Entrada:
    -------
    x : torch.Tensor
        Tensor com shape (..., emb_dim). Exemplos comuns:
        - (batch, seq_len, emb_dim)
        - (seq_len, emb_dim)

    Saída:
    -----
    torch.Tensor
        Tensor normalizado com o mesmo shape de `x`.

    Observações:
    -----------
    - `unbiased=False` em `var` corresponde ao comportamento típico em LN.
    - `scale` e `shift` têm shape (emb_dim,) e são broadcastados para o shape de `x`.

    Referência:
    ----------
    Seção 4.2 — "Normalizing activations with layer normalization". :contentReference[oaicite:0]{index=0}
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()

        if not isinstance(emb_dim, int) or emb_dim <= 0:
            raise ValueError("emb_dim deve ser um int positivo.")

        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executa a normalização por camada (LayerNorm) ao longo do último eixo.

        Parâmetros:
        ----------
        x : torch.Tensor
            Tensor de entrada com shape (..., emb_dim).

        Retorno:
        -------
        torch.Tensor
            Tensor normalizado com o mesmo shape de `x`.
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
