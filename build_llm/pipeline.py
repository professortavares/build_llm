from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Set

import torch

from .datasets import criar_dataloader_gpt, criar_dataloader_cbow
from .embedding import treinar_cbow, salvar_embeddings_cbow
from .tokenizador import (
    RemapeamentoVocab,
    tokenizar_texto,
    criar_remapeamento_vocab,
    remapear_token_ids,
)
from .utils import ler_texto_arquivo, salvar_json, slugify


def preparar_dataset_gpt_e_salvar(
    *,
    caminho_texto: str | Path,
    batch_size: int,
    max_length: int,
    stride: int,
    encoding_name: str = "gpt2",
    allowed_special: Optional[Set[str]] = None,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
    encoding_arquivo: str = "utf-8",
    pasta_data_root: str | Path = "../../data",
    nome_pasta_data: Optional[str] = None,
) -> tuple[Any, RemapeamentoVocab, Path]:
    """
    Lê arquivo, tokeniza, cria vocabulário compacto, monta dataloader GPT e salva artefatos.
    """
    allowed_special = allowed_special or {"<|endoftext|>"}

    caminho_texto = Path(caminho_texto).expanduser().resolve()
    texto = ler_texto_arquivo(caminho_texto, encoding=encoding_arquivo)

    token_ids_raw = tokenizar_texto(texto, encoding_name=encoding_name, allowed_special=allowed_special)
    remap = criar_remapeamento_vocab(token_ids_raw, encoding_name=encoding_name, allowed_special=allowed_special)
    token_ids_compactos = remapear_token_ids(token_ids_raw, remap)

    dl = criar_dataloader_gpt(
        token_ids_compactos,
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    data_root = Path(pasta_data_root).expanduser().resolve()
    base_name = nome_pasta_data or f"gpt_data_{slugify(caminho_texto.stem)}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    pasta_data = (data_root / base_name).resolve()
    pasta_data.mkdir(parents=True, exist_ok=True)

    # salva
    (pasta_data / "corpus.txt").write_text(texto, encoding="utf-8")
    torch.save(torch.tensor(token_ids_raw, dtype=torch.int32), pasta_data / "token_ids_raw.pt")
    torch.save(torch.tensor(token_ids_compactos, dtype=torch.int32), pasta_data / "token_ids_compactos.pt")
    salvar_json(remap.to_dict(), pasta_data / "remapeamento_vocab.json")

    cfg = {
        "origem_corpus": str(caminho_texto),
        "encoding_arquivo": encoding_arquivo,
        "encoding_name": encoding_name,
        "allowed_special": sorted(list(allowed_special)),
        "batch_size": batch_size,
        "max_length": max_length,
        "stride": stride,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "num_tokens_raw": len(token_ids_raw),
        "vocab_size_compacto": remap.vocab_size_compacto,
        "num_sequencias": len(dl.dataset),
        "criado_em": remap.criado_em,
    }
    salvar_json(cfg, pasta_data / "config_dataset.json")

    return dl, remap, pasta_data


def carregar_token_ids_compactos(pasta_data: str | Path) -> list[int]:
    """
    Carrega token_ids_compactos.pt e retorna como list[int].
    """
    p = Path(pasta_data).expanduser().resolve()
    t = torch.load(p / "token_ids_compactos.pt", map_location="cpu")
    return t.tolist()


def ler_config_dataset(pasta_data: str | Path) -> dict:
    """
    Lê config_dataset.json.
    """
    p = Path(pasta_data).expanduser().resolve()
    return json.loads((p / "config_dataset.json").read_text(encoding="utf-8"))


def pipeline_dataset_e_embeddings(
    *,
    caminho_texto: str | Path,
    # dataset GPT
    batch_size_gpt: int,
    max_length: int,
    stride: int,
    nome_base: Optional[str] = None,
    # saídas
    pasta_data_root: str | Path = "../../data",
    pasta_models_root: str | Path = "../../models",
    # tokenização
    encoding_name: str = "gpt2",
    allowed_special: Optional[Set[str]] = None,
    # treino embeddings (CBOW)
    treinar_embeddings: bool = True,
    embedding_dim: int = 256,
    cbow_janela: int = 2,
    cbow_batch_size: int = 256,
    cbow_epocas: int = 50,
    cbow_lr: float = 0.05,
    device: Optional[str] = None,
) -> dict[str, Any]:
    """
    Pipeline completo:
      1) cria dataset GPT com vocabulário compacto e salva em ../../data/...
      2) (opcional) treina embeddings CBOW com vocabulário compacto e salva em ../../models/...

    Retorno:
    -------
    dict com:
      - dataloader_gpt
      - remap
      - pasta_data
      - (se treinar_embeddings) pasta_models + caminho_pesos_token_embedding
    """
    caminho_texto = Path(caminho_texto).expanduser().resolve()
    base = nome_base or slugify(caminho_texto.stem)

    # 1) Dataset GPT + artefatos
    dl_gpt, remap, pasta_data = preparar_dataset_gpt_e_salvar(
        caminho_texto=caminho_texto,
        batch_size=batch_size_gpt,
        max_length=max_length,
        stride=stride,
        encoding_name=encoding_name,
        allowed_special=allowed_special,
        pasta_data_root=pasta_data_root,
        nome_pasta_data=f"gpt_data_{base}",
    )

    saida: dict[str, Any] = {
        "dataloader_gpt": dl_gpt,
        "remap": remap,
        "pasta_data": pasta_data,
        "vocab_size_compacto": remap.vocab_size_compacto,
    }

    # 2) Treino embeddings (CBOW)
    if treinar_embeddings:
        token_ids_compactos = carregar_token_ids_compactos(pasta_data)

        loader_cbow = criar_dataloader_cbow(
            token_ids_compactos,
            janela=cbow_janela,
            batch_size=cbow_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        model_cbow = treinar_cbow(
            loader_cbow,
            vocab_size=remap.vocab_size_compacto,
            embedding_dim=embedding_dim,
            epocas=cbow_epocas,
            lr=cbow_lr,
            device=device,
        )

        pasta_models = salvar_embeddings_cbow(
            model_cbow,
            pasta_models_root=pasta_models_root,
            nome_pasta_models=f"cbow_embeddings_{base}",
            salvar_modelo_completo=True,
            config_extra={
                "pipeline": "pipeline_dataset_e_embeddings",
                "pasta_data": str(pasta_data),
                "caminho_texto": str(caminho_texto),
                "vocab_size_compacto": remap.vocab_size_compacto,
                "embedding_dim": embedding_dim,
                "cbow_janela": cbow_janela,
                "cbow_batch_size": cbow_batch_size,
                "cbow_epocas": cbow_epocas,
                "cbow_lr": cbow_lr,
                "encoding_name": remap.encoding_name,
                "criado_em": datetime.utcnow().isoformat() + "Z",
            },
        )

        saida["pasta_models"] = pasta_models
        saida["caminho_pesos_token_embedding"] = pasta_models / "token_embedding_weight.pt"

    return saida
