from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .utils import salvar_json


class CBOWTorch(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(context_ids)      # (B, C, D)
        emb_mean = emb.mean(dim=1)             # (B, D)
        logits = self.proj(emb_mean)           # (B, V)
        return logits


def treinar_cbow(
    loader: DataLoader,
    *,
    vocab_size: int,
    embedding_dim: int = 256,
    epocas: int = 20,
    lr: float = 0.05,
    device: Optional[str] = None,
) -> CBOWTorch:
    """
    Treina CBOW e retorna o modelo.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CBOWTorch(vocab_size=vocab_size, embedding_dim=embedding_dim).to(device)
    opt = optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epocas + 1):
        model.train()
        loss_total = 0.0

        for ctx, y in loader:
            ctx = ctx.to(device)
            y = y.to(device)

            logits = model(ctx)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_total += float(loss.item())

        if ep == 1 or ep % 10 == 0:
            loss_med = loss_total / max(1, len(loader))
            print(f"Época {ep:03d}/{epocas} | loss médio: {loss_med:.4f}")

    return model


def salvar_embeddings_cbow(
    model: CBOWTorch,
    *,
    pasta_models_root: str | Path = "../../models",
    nome_pasta_models: Optional[str] = None,
    salvar_modelo_completo: bool = True,
    config_extra: Optional[dict] = None,
) -> Path:
    """
    Salva pesos do embedding e (opcionalmente) state_dict completo + config.
    """
    models_root = Path(pasta_models_root).expanduser().resolve()
    base_name = nome_pasta_models or f"cbow_embeddings_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    pasta_models = (models_root / base_name).resolve()
    pasta_models.mkdir(parents=True, exist_ok=True)

    torch.save(model.embedding.weight.detach().cpu(), pasta_models / "token_embedding_weight.pt")

    if salvar_modelo_completo:
        torch.save(model.state_dict(), pasta_models / "cbow_state_dict.pt")

    cfg = {
        "criado_em": datetime.utcnow().isoformat() + "Z",
    }
    if config_extra:
        cfg.update(config_extra)

    salvar_json(cfg, pasta_models / "config_cbow.json")
    return pasta_models


def criar_camadas_embedding_para_gpt(
    *,
    vocab_size_compacto: int,
    output_dim: int,
    context_length: int,
    caminho_pesos_token_embedding: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> Tuple[nn.Embedding, nn.Embedding]:
    """
    Cria token_embedding e pos_embedding; opcionalmente inicializa token_embedding
    com pesos pré-treinados.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    token_embedding_layer = nn.Embedding(vocab_size_compacto, output_dim).to(device)
    pos_embedding_layer = nn.Embedding(context_length, output_dim).to(device)

    if caminho_pesos_token_embedding is not None:
        w = torch.load(Path(caminho_pesos_token_embedding).expanduser().resolve(), map_location="cpu")
        if not isinstance(w, torch.Tensor):
            raise TypeError("Arquivo de pesos não contém um torch.Tensor.")
        if w.shape != (vocab_size_compacto, output_dim):
            raise ValueError(
                f"Shape incompatível: {tuple(w.shape)} (esperado {(vocab_size_compacto, output_dim)})"
            )
        token_embedding_layer.weight.data.copy_(w.to(device))

    return token_embedding_layer, pos_embedding_layer
