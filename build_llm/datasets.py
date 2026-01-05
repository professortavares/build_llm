from __future__ import annotations

from typing import List

import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetVocabCompacto(Dataset):
    """
    Dataset GPT (input_ids, target_ids) com IDs compactos.
    """

    def __init__(self, token_ids_compactos: List[int], max_length: int, stride: int):
        if not isinstance(max_length, int) or max_length <= 1:
            raise ValueError("max_length deve ser int > 1.")
        if not isinstance(stride, int) or stride <= 0:
            raise ValueError("stride deve ser int > 0.")

        self.input_ids: List[torch.Tensor] = []
        self.target_ids: List[torch.Tensor] = []

        for i in range(0, len(token_ids_compactos) - max_length, stride):
            input_chunk = token_ids_compactos[i : i + max_length]
            target_chunk = token_ids_compactos[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


class CBOWDataset(Dataset):
    """
    Dataset CBOW simples a partir de token IDs compactos.
    """

    def __init__(self, token_ids: List[int], janela: int = 2):
        if not isinstance(janela, int) or janela <= 0:
            raise ValueError("janela deve ser int > 0.")
        if len(token_ids) < (2 * janela + 1):
            raise ValueError("Sequência muito curta para a janela informada.")

        self.janela = janela
        self.indices = list(range(janela, len(token_ids) - janela))

        self.token_ids = token_ids

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        left = self.token_ids[i - self.janela : i]
        right = self.token_ids[i + 1 : i + 1 + self.janela]
        context = left + right
        target = self.token_ids[i]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def criar_dataloader_gpt(
    token_ids_compactos: list[int],
    *,
    batch_size: int,
    max_length: int,
    stride: int,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Cria DataLoader GPT (estilo livro) a partir de IDs compactos.
    """
    ds = GPTDatasetVocabCompacto(token_ids_compactos, max_length=max_length, stride=stride)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def criar_dataloader_cbow(
    token_ids_compactos: list[int],
    *,
    janela: int = 2,
    batch_size: int = 256,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Cria DataLoader CBOW a partir de IDs compactos.
    """
    ds = CBOWDataset(token_ids_compactos, janela=janela)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
