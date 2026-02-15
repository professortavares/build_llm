import torch
from torch import nn

from build_llm.activation import GELU


class FeedForward(nn.Module):
    """
    Rede feed-forward (MLP) usada dentro de um bloco Transformer (estilo GPT).

    Estrutura:
    ----------
    Linear(emb_dim -> 4*emb_dim) -> GELU -> Linear(4*emb_dim -> emb_dim)

    Parâmetros:
    ----------
    cfg : dict
        Dicionário de configuração contendo:
        - "emb_dim" (int): dimensão do embedding / hidden size do modelo.

    Exceções:
    --------
    Levanta KeyError se "emb_dim" não existir em cfg.
    Levanta TypeError / ValueError se "emb_dim" não puder ser convertido para int
    ou não for um inteiro positivo.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()

        if "emb_dim" not in cfg:
            raise KeyError('cfg deve conter a chave "emb_dim".')

        try:
            emb_dim = int(cfg["emb_dim"])
        except (TypeError, ValueError) as e:
            raise TypeError(
                '"emb_dim" deve ser um inteiro (ou conversível para int).'
            ) from e

        if emb_dim <= 0:
            raise ValueError('"emb_dim" deve ser um inteiro positivo.')

        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executa o forward pass do feed-forward.

        Parâmetros:
        ----------
        x : torch.Tensor
            Tensor de entrada com shape (..., emb_dim).

        Retorno:
        -------
        torch.Tensor
            Tensor de saída com o mesmo shape de entrada (..., emb_dim).
        """
        return self.layers(x)
