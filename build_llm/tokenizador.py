from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional, Set, Tuple

import tiktoken


@dataclass(frozen=True)
class RemapeamentoVocab:
    """
    Metadados do remapeamento do vocabulário.
    - raw_vocab_ids: lista ordenada dos token IDs originais (GPT-2) observados no corpus.
      O novo ID (compacto) é o índice nessa lista.
    """
    encoding_name: str
    allowed_special: List[str]
    raw_vocab_ids: List[int]
    criado_em: str

    @property
    def vocab_size_compacto(self) -> int:
        return len(self.raw_vocab_ids)

    def construir_mapas(self) -> tuple[dict[int, int], dict[int, int]]:
        raw_id_to_new: dict[int, int] = {rid: j for j, rid in enumerate(self.raw_vocab_ids)}
        new_to_raw_id: dict[int, int] = {j: rid for j, rid in enumerate(self.raw_vocab_ids)}
        return raw_id_to_new, new_to_raw_id

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "RemapeamentoVocab":
        return RemapeamentoVocab(
            encoding_name=d["encoding_name"],
            allowed_special=list(d["allowed_special"]),
            raw_vocab_ids=list(d["raw_vocab_ids"]),
            criado_em=d["criado_em"],
        )


def tokenizar_texto(
    texto: str,
    *,
    encoding_name: str = "gpt2",
    allowed_special: Optional[Set[str]] = None,
) -> list[int]:
    """
    Tokeniza texto com tiktoken e retorna token IDs "raw" (GPT-2).
    """
    allowed_special = allowed_special or {"<|endoftext|>"}
    tok = tiktoken.get_encoding(encoding_name)
    return tok.encode(texto, allowed_special=set(allowed_special))


def criar_remapeamento_vocab(
    token_ids_raw: list[int],
    *,
    encoding_name: str = "gpt2",
    allowed_special: Optional[Set[str]] = None,
) -> RemapeamentoVocab:
    """
    Cria vocabulário compacto (apenas tokens presentes no corpus) e retorna metadados.
    """
    allowed_special = allowed_special or {"<|endoftext|>"}
    raw_vocab_ids = sorted(set(token_ids_raw))

    return RemapeamentoVocab(
        encoding_name=encoding_name,
        allowed_special=sorted(list(allowed_special)),
        raw_vocab_ids=raw_vocab_ids,
        criado_em=datetime.utcnow().isoformat() + "Z",
    )


def remapear_token_ids(
    token_ids_raw: list[int],
    remap: RemapeamentoVocab,
) -> list[int]:
    """
    Converte token IDs raw -> compactos.
    """
    raw_id_to_new, _ = remap.construir_mapas()
    return [raw_id_to_new[rid] for rid in token_ids_raw]


def obter_decoder_compacto(remap: RemapeamentoVocab) -> tuple[tiktoken.Encoding, dict[int, int]]:
    """
    Retorna (tokenizer, new_to_raw_id) para decodificar token compacto -> string.
    """
    tok = tiktoken.get_encoding(remap.encoding_name)
    _, new_to_raw_id = remap.construir_mapas()
    return tok, new_to_raw_id
