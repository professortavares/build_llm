import torch
from torch import nn


class GELU(nn.Module):
    """
    Implementação do GELU (Gaussian Error Linear Unit) na forma aproximada
    com tanh, muito usada em arquiteturas tipo GPT.

    Fórmula (aproximação):
        GELU(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ))

    Observações:
    - Esta é a aproximação popularizada no contexto de Transformers.
    - Mantém o dtype e o device do tensor de entrada.

    Exceções:
    --------
    TypeError
        Se a entrada não for um torch.Tensor.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica a ativação GELU aproximada ao tensor de entrada.

        Parâmetros:
        ----------
        x : torch.Tensor
            Tensor de entrada.

        Retorno:
        -------
        torch.Tensor
            Tensor com GELU aplicado, mesmo shape do input.

        Exceções:
        --------
        TypeError
            Se x não for torch.Tensor.
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError("x deve ser um torch.Tensor.")

        # Constantes no mesmo dtype/device do input para evitar casts/CPU<->GPU.
        sqrt_2_over_pi = torch.sqrt(
            torch.tensor(2.0 / torch.pi, dtype=x.dtype, device=x.device)
        )
        return (
            0.5
            * x
            * (1.0 + torch.tanh(sqrt_2_over_pi * (x + 0.044715 * torch.pow(x, 3))))
        )
