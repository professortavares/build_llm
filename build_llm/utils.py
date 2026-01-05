from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict


def slugify(texto: str) -> str:
    """
    Converte uma string em um "slug" seguro para nomes de pasta/arquivo.
    """
    s = str(texto).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = s.strip("._-") or "run"
    return s


def ler_texto_arquivo(caminho_arquivo: str | Path, encoding: str = "utf-8") -> str:
    """
    Lê um arquivo texto e retorna seu conteúdo.
    """
    p = Path(caminho_arquivo).expanduser().resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}")
    return p.read_text(encoding=encoding)


def salvar_json(obj: Dict[str, Any], caminho: str | Path) -> Path:
    """
    Salva um dict como JSON (UTF-8) com indentação.
    """
    p = Path(caminho).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p
