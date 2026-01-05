from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import wikipedia


def configurar_wikipedia(idioma: str = "pt") -> None:
    """
    Configura o idioma padrão do pacote `wikipedia`.

    Parâmetros:
    ----------
    idioma : str, default = "pt"
        Código do idioma (ex.: "pt", "en", "es").

    Retorno:
    -------
    None
    """
    wikipedia.set_lang(idioma)


def _nome_arquivo_seguro(nome_base: str) -> str:
    """
    Gera um nome de arquivo seguro para o sistema (remove caracteres problemáticos).

    Parâmetros:
    ----------
    nome_base : str
        Nome base do arquivo.

    Retorno:
    -------
    str
        Nome sanitizado (sem extensão).
    """
    nome = str(nome_base).strip()
    nome = re.sub(r"\s+", "_", nome)
    nome = re.sub(r"[^A-Za-z0-9._-]+", "_", nome)
    nome = nome.strip("._-") or "wikipedia"
    return nome


def salvar_texto_em_pasta(
    texto: str,
    pasta_destino: str | Path,
    nome_base: str,
    encoding: str = "utf-8",
) -> Path:
    """
    Salva um texto em arquivo .txt dentro da pasta especificada.

    Parâmetros:
    ----------
    texto : str
        Conteúdo a ser salvo.
    pasta_destino : str | Path
        Diretório onde o arquivo será salvo.
    nome_base : str
        Nome base do arquivo (sem extensão). Será sanitizado.
    encoding : str, default = "utf-8"
        Codificação do arquivo.

    Retorno:
    -------
    Path
        Caminho completo do arquivo salvo.

    Exceções:
    --------
    TypeError
        Se 'texto' não for str.
    OSError
        Se houver erro ao criar pasta ou escrever arquivo.
    """
    if not isinstance(texto, str):
        raise TypeError("O parâmetro 'texto' deve ser uma string.")

    pasta = Path(pasta_destino).expanduser().resolve()
    pasta.mkdir(parents=True, exist_ok=True)

    filename = _nome_arquivo_seguro(nome_base) + ".txt"
    out_path = pasta / filename
    out_path.write_text(texto, encoding=encoding)
    return out_path


def _extrair_titulo_de_url(url: str) -> Optional[str]:
    """
    Extrai o título da URL no formato https://pt.wikipedia.org/wiki/TITULO

    Parâmetros:
    ----------
    url : str
        URL da Wikipédia.

    Retorno:
    -------
    Optional[str]
        Título extraído ou None se não for possível.
    """
    try:
        s = str(url).strip()
    except (TypeError, ValueError):
        return None

    if not s.lower().startswith(("http://", "https://")):
        return None

    # Extrai o trecho após "/wiki/"
    m = re.search(r"/wiki/([^#?]+)", s)
    if not m:
        return None

    titulo = m.group(1).strip()
    return titulo or None


def buscar_conteudo_wikipedia_com_pacote(
    assunto: str,
    idioma: str = "pt",
    auto_suggest: bool = False,
    redirect: bool = True,
) -> str:
    """
    Busca o conteúdo textual de uma página da Wikipédia usando o pacote `wikipedia`.

    Aceita:
    - Título (ex.: "Machado de Assis")
    - URL (ex.: "https://pt.wikipedia.org/wiki/Machado_de_Assis")

    Parâmetros:
    ----------
    assunto : str
        Título ou URL.
    idioma : str, default = "pt"
        Idioma da Wikipédia a ser consultada.
    auto_suggest : bool, default = False
        Se True, permite sugestão automática de título (pode mudar o resultado).
    redirect : bool, default = True
        Se True, segue redirecionamentos.

    Retorno:
    -------
    str
        Conteúdo da página (texto).

    Exceções:
    --------
    TypeError
        Se 'assunto' não puder ser convertido para str.
    ValueError
        Se o assunto estiver vazio.
    wikipedia.exceptions.PageError
        Se a página não existir.
    wikipedia.exceptions.DisambiguationError
        Se houver ambiguidade (várias páginas possíveis).
    """
    try:
        assunto_str = str(assunto).strip()
    except (TypeError, ValueError) as e:
        raise TypeError("O parâmetro 'assunto' deve ser uma string.") from e

    if not assunto_str:
        raise ValueError("O parâmetro 'assunto' não pode estar vazio.")

    configurar_wikipedia(idioma)

    # Se vier URL, tenta extrair o título
    titulo_url = _extrair_titulo_de_url(assunto_str)
    titulo = titulo_url if titulo_url else assunto_str

    # O pacote aceita underscores no título; também funciona com espaços
    conteudo = wikipedia.page(title=titulo, auto_suggest=auto_suggest, redirect=redirect).content
    if not isinstance(conteudo, str) or not conteudo.strip():
        raise ValueError(f"Conteúdo vazio para o assunto: {assunto!r}")

    return conteudo


def baixar_wikipedia_e_salvar_com_pacote(
    assunto: str,
    pasta_destino: str | Path,
    idioma: str = "pt",
    auto_suggest: bool = False,
    redirect: bool = True,
) -> Path:
    """
    Busca conteúdo de um assunto na Wikipédia usando o pacote `wikipedia`
    e salva como .txt na pasta destino.

    Parâmetros:
    ----------
    assunto : str
        Título ou URL da página.
    pasta_destino : str | Path
        Pasta onde o arquivo será salvo.
    idioma : str, default = "pt"
        Idioma da Wikipédia.
    auto_suggest : bool, default = False
        Sugestão automática de título.
    redirect : bool, default = True
        Seguir redirecionamentos.

    Retorno:
    -------
    Path
        Caminho do arquivo salvo.
    """
    texto = buscar_conteudo_wikipedia_com_pacote(
        assunto=assunto,
        idioma=idioma,
        auto_suggest=auto_suggest,
        redirect=redirect,
    )

    # Nome do arquivo: usa o título extraído da URL se existir; senão usa o assunto
    nome_base = _extrair_titulo_de_url(str(assunto)) or str(assunto)
    return salvar_texto_em_pasta(texto=texto, pasta_destino=pasta_destino, nome_base=nome_base)