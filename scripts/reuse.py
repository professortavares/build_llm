import os
import re
import unicodedata
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import requests
import torch
from matplotlib.ticker import MaxNLocator
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from build_llm.util import generate_text_simple, text_to_token_ids, token_ids_to_text

_CHECKPOINT_RE = re.compile(r"^(?P<prefix>.+)_part_(?P<epoch>\d+)\.pth$")

def _find_latest_checkpoint(
    checkpoint_dir: str | os.PathLike,
    checkpoint_prefix: str,
) -> Path | None:
    """
    Encontra o checkpoint mais recente no diretório, baseado no sufixo _part_XX.pth.
    Retorna o Path do arquivo, ou None se não houver.
    """
    ckpt_dir = Path(checkpoint_dir).expanduser().resolve()
    if not ckpt_dir.exists():
        return None

    best_epoch = -1
    best_path: Path | None = None

    for p in ckpt_dir.glob(f"{checkpoint_prefix}_part_*.pth"):
        m = _CHECKPOINT_RE.match(p.name)
        if not m:
            continue
        if m.group("prefix") != checkpoint_prefix:
            continue

        epoch_num = int(m.group("epoch"))
        if epoch_num > best_epoch:
            best_epoch = epoch_num
            best_path = p

    return best_path


def _save_checkpoint(
    part_path: str | os.PathLike,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    completed_epochs: int,
    global_step: int,
    tokens_seen: int,
    train_losses: list[float],
    val_losses: list[float],
    track_tokens_seen: list[int],
    latest_name: str = "latest.pth",
) -> None:
    """
    Salva um checkpoint por época (part_XX) e também um atalho 'latest.pth'
    no mesmo diretório, sempre com o conteúdo mais recente.
    """
    part_path = Path(part_path).expanduser().resolve()
    part_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "completed_epochs": int(completed_epochs),
        "global_step": int(global_step),
        "tokens_seen": int(tokens_seen),
        "train_losses": list(train_losses),
        "val_losses": list(val_losses),
        "track_tokens_seen": list(track_tokens_seen),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # 1) salva o checkpoint por época
    torch.save(payload, str(part_path))

    # 2) salva/atualiza o atalho latest.pth (mesmo conteúdo)
    latest_path = part_path.parent / latest_name
    torch.save(payload, str(latest_path))

def _load_checkpoint(
    path: str | os.PathLike,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    """
    Carrega checkpoint e restaura model/optimizer.
    Retorna o dicionário com os metadados (completed_epochs, losses, etc).
    """
    ckpt_path = Path(path).expanduser().resolve()
    payload = torch.load(str(ckpt_path), map_location=device)

    if "model_state_dict" not in payload or "optimizer_state_dict" not in payload:
        raise ValueError(f"Checkpoint inválido (faltam chaves) em: {ckpt_path}")

    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    # Garante que tensores internos do optimizer fiquem no device correto
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

    return payload

def _slugify_filename(name: str, max_len: int = 120) -> str:
    """
    Converte um título em um nome de arquivo seguro (ASCII, sem espaços/símbolos problemáticos).

    Estratégia:
    - Normaliza Unicode (remove acentos)
    - Mantém letras, números, sublinhado e hífen
    - Espaços viram underscore
    - Limita tamanho

    Parâmetros:
    ----------
    name : str
        Texto base (ex.: título do livro).
    max_len : int, default = 120
        Tamanho máximo do nome final (sem extensão).

    Retorno:
    -------
    str
        Nome de arquivo "seguro" (sem extensão).

    Exceções:
    --------
    Levanta TypeError se 'name' não for string.
    """
    if not isinstance(name, str):
        raise TypeError("O parâmetro 'name' deve ser uma string.")

    # Remove acentos e caracteres especiais
    normalized = unicodedata.normalize("NFKD", name)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")

    # Substitui espaços por underscore e remove caracteres indesejados
    ascii_text = ascii_text.strip().lower().replace(" ", "_")
    ascii_text = re.sub(
        r"[^a-z0-9_-]+", "", ascii_text
    )  # remove tudo que não for seguro
    ascii_text = re.sub(r"_{2,}", "_", ascii_text).strip("_")

    if not ascii_text:
        ascii_text = "livro"

    return ascii_text[:max_len]


def baixar_livros_machado_assis_gutenberg(
    base_dir: str | os.PathLike = "../../data",
    livros: dict[str, str] | None = None,
    timeout: int = 30,
    overwrite: bool = False,
    session: requests.Session | None = None,
) -> list[tuple[str, str]]:
    """
    Baixa (ou reaproveita do disco) todos os livros do Machado de Assis listados, do Projeto Gutenberg.

    - Se o arquivo já existir e overwrite=False, apenas lê o conteúdo.
    - Se não existir (ou overwrite=True), faz download e salva em UTF-8.
    - Retorna uma lista (titulo, caminho_do_arquivo) para facilitar logging/uso posterior.

    Parâmetros:
    ----------
    base_dir : str | os.PathLike, default = "../../data"
        Pasta onde os arquivos serão salvos.
    livros : dict[str, str] | None, default = None
        Mapeamento {titulo: url_txt}. Se None, usa a lista fornecida no enunciado.
    timeout : int, default = 30
        Timeout do requests (em segundos).
    overwrite : bool, default = False
        Se True, força re-download mesmo que o arquivo já exista.
    session : requests.Session | None, default = None
        Sessão opcional para reutilizar conexões.

    Retorno:
    -------
    list[tuple[str, str]]
        Lista de tuplas (título, caminho absoluto do arquivo salvo).

    Exceções:
    --------
    Levanta:
    - TypeError se tipos de entrada forem inválidos.
    - requests.HTTPError para respostas HTTP inválidas (raise_for_status).
    - OSError em caso de falha ao criar diretório ou gravar arquivo.
    """
    if livros is None:
        livros = {
            "Dom Casmurro": "https://www.gutenberg.org/cache/epub/55752/pg55752.txt",
        }

    if not isinstance(timeout, int) or timeout <= 0:
        raise TypeError("O parâmetro 'timeout' deve ser um inteiro positivo.")
    if not isinstance(overwrite, bool):
        raise TypeError("O parâmetro 'overwrite' deve ser bool.")
    if not isinstance(livros, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in livros.items()
    ):
        raise TypeError(
            "O parâmetro 'livros' deve ser um dict[str, str] (titulo -> url)."
        )

    base_path = Path(base_dir).expanduser().resolve()
    base_path.mkdir(parents=True, exist_ok=True)

    sess = session or requests.Session()

    salvos: list[tuple[str, str]] = []

    for titulo, url in livros.items():
        filename = f"{_slugify_filename(titulo)}.txt"
        file_path = base_path / filename

        try:
            if file_path.exists() and not overwrite:
                # Reaproveita o que já tem em disco (sem recarregar rede)
                _ = file_path.read_text(encoding="utf-8")
            else:
                resp = sess.get(url, timeout=timeout)
                resp.raise_for_status()
                text_data = resp.text
                file_path.write_text(text_data, encoding="utf-8")

            salvos.append((titulo, str(file_path)))
            print(f"Salvo: {filename}")
        except requests.RequestException as e:
            # Falha de rede/HTTP
            raise requests.RequestException(
                f"Falha ao baixar '{titulo}' ({url})."
            ) from e
        except OSError as e:
            # Falha de I/O
            raise OSError(f"Falha ao salvar/ler '{titulo}' em '{file_path}'.") from e

    return salvos


def preprocessar_livros_gutenberg_por_linha(
    raw_dir: str | os.PathLike = "../../data/raw",
    out_dir: str | os.PathLike = "../../data/preprocessed",
    cortes: dict[str, tuple[int, int]] | None = None,
    encoding: str = "utf-8",
    overwrite: bool = False,
    debug: bool = False,
) -> list[tuple[str, str]]:
    """
    Remove cabeçalho e rodapé por números de linha (1-based), usando contagem "tradicional" de linhas
    (iteração do arquivo), evitando divergências do splitlines() com separadores como '\\f'.

    Interpretação (conforme seu enunciado):
    - header_end_line: linha onde TERMINA o cabeçalho
      => tudo ANTES dessa linha pode ser excluído
      => começamos a MANTER a partir da própria header_end_line (inclusive)
    - footer_start_line: linha onde COMEÇA o rodapé
      => tudo APÓS essa linha pode ser desconsiderado
      => mantemos ATÉ a própria footer_start_line (inclusive)

    Se você quiser EXCLUIR as linhas-limite também:
    - troque keep_header_end=True/keep_footer_start=True para False no código (ver abaixo).

    Parâmetros:
    ----------
    raw_dir : str | os.PathLike
        Pasta contendo os arquivos originais.
    out_dir : str | os.PathLike
        Pasta para salvar os arquivos pré-processados.
    cortes : dict[str, tuple[int, int]] | None
        {filename: (header_end_line, footer_start_line)}.
    encoding : str
        Encoding para leitura e escrita.
    overwrite : bool
        Se True, sobrescreve arquivos já existentes.
    debug : bool
        Se True, imprime as linhas de corte e amostras do que foi mantido.

    Retorno:
    -------
    list[tuple[str, str]]
        Lista de (filename, caminho_saida_absoluto).

    Exceções:
    --------
    - FileNotFoundError se algum arquivo não existir.
    - ValueError para cortes inválidos.
    - OSError para erros de I/O.
    """
    if cortes is None:
        cortes = {
            "dom_casmurro.txt": (56, 17460),
        }

    if not isinstance(cortes, dict) or not all(
        isinstance(k, str) and isinstance(v, tuple) and len(v) == 2
        for k, v in cortes.items()
    ):
        raise TypeError("O parâmetro 'cortes' deve ser um dict[str, tuple[int,int]].")

    raw_path = Path(raw_dir).expanduser().resolve()
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    # Mantém ou exclui as linhas-limite.
    # Pelo seu texto, faz sentido MANTER as linhas-limite (inclusive).
    keep_header_end = True
    keep_footer_start = True

    resultados: list[tuple[str, str]] = []

    for filename, (header_end_line, footer_start_line) in cortes.items():
        if not (
            isinstance(header_end_line, int) and isinstance(footer_start_line, int)
        ):
            raise TypeError(
                f"Cortes inválidos para '{filename}': precisam ser inteiros."
            )
        if header_end_line < 1 or footer_start_line < 1:
            raise ValueError(
                f"Cortes inválidos para '{filename}': linhas devem ser >= 1."
            )
        if header_end_line > footer_start_line:
            raise ValueError(
                f"Cortes inválidos para '{filename}': header_end_line ({header_end_line}) > footer_start_line ({footer_start_line})."
            )

        in_file = raw_path / filename
        if not in_file.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {in_file}")

        out_file = out_path / filename
        if out_file.exists() and not overwrite:
            resultados.append((filename, str(out_file)))
            continue

        # Define faixa a manter (1-based, inclusive)
        start_line = header_end_line if keep_header_end else header_end_line + 1
        end_line = footer_start_line if keep_footer_start else footer_start_line - 1

        if start_line > end_line:
            raise ValueError(
                f"Corte resultou vazio em '{filename}': start_line={start_line} > end_line={end_line}."
            )

        first_kept: str | None = None
        last_kept: str | None = None
        written = 0
        total = 0

        # Iterar o arquivo conta linhas no padrão "real", evitando problemas do splitlines() com '\f'
        with (
            open(in_file, encoding=encoding, newline=None) as fin,
            open(out_file, "w", encoding=encoding, newline="") as fout,
        ):
            for lineno, line in enumerate(fin, start=1):
                total = lineno
                if start_line <= lineno <= end_line:
                    if first_kept is None:
                        first_kept = line
                    last_kept = line
                    fout.write(line)
                    written += 1

        if total < header_end_line or total < footer_start_line:
            raise ValueError(
                f"Cortes fora do tamanho do arquivo '{filename}': total_linhas={total}, "
                f"header_end={header_end_line}, footer_start={footer_start_line}"
            )

        resultados.append((filename, str(out_file)))

        if debug:
            print(
                f"\n[{filename}] total_linhas={total} | mantido={start_line}..{end_line} | linhas_gravadas={written}"
            )
            if first_kept is not None:
                print("  --- primeira linha mantida (repr) ---")
                print(repr(first_kept[:200]))
            if last_kept is not None:
                print("  --- última linha mantida (repr) ---")
                print(repr(last_kept[:200]))

    return resultados


def concatenar_livros_em_uma_string(
    preprocessed_dir: str | os.PathLike = "../../data/preprocessed",
    arquivos: Iterable[str] | None = None,
    encoding: str = "utf-8",
    separador: str = "\n\n" + ("-" * 80) + "\n\n",
    ordenar: bool = True,
) -> str:
    """
    Lê todos (ou alguns) arquivos .txt pré-processados e concatena em uma única string.

    Parâmetros:
    ----------
    preprocessed_dir : str | os.PathLike, default = "../../data/preprocessed"
        Pasta contendo os textos pré-processados.
    arquivos : Iterable[str] | None, default = None
        Lista/iterável com nomes de arquivos para concatenar.
        - Se None, concatena todos os .txt da pasta.
    encoding : str, default = "utf-8"
        Encoding para leitura.
    separador : str, default = "\\n\\n" + ("-" * 80) + "\\n\\n"
        Texto inserido entre livros (ajuda a não "colar" finais/inícios).
    ordenar : bool, default = True
        Se True, ordena alfabeticamente os arquivos antes de concatenar.

    Retorno:
    -------
    str
        Uma única string com o conteúdo concatenado.

    Exceções:
    --------
    - FileNotFoundError se a pasta não existir ou se algum arquivo especificado não existir.
    - OSError para erros de leitura.
    """
    base = Path(preprocessed_dir).expanduser().resolve()
    if not base.exists() or not base.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {base}")

    if arquivos is None:
        files: list[Path] = list(base.glob("*.txt"))
        if ordenar:
            files.sort(key=lambda p: p.name.lower())
    else:
        files = []
        for name in arquivos:
            p = base / str(name)
            if not p.exists():
                raise FileNotFoundError(f"Arquivo não encontrado: {p}")
            files.append(p)
        if ordenar:
            files.sort(key=lambda p: p.name.lower())

    partes: list[str] = []
    for p in files:
        partes.append(p.read_text(encoding=encoding))

    return separador.join(partes)


def criar_train_val_dataloaders(
    text_data: str,
    create_dataloader_fn: Callable[..., Any],
    gpt_config: dict[str, Any],
    train_ratio: float = 0.90,
    batch_size: int = 2,
    stride: int | None = None,
    drop_last_train: bool = True,
    drop_last_val: bool = False,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    num_workers: int = 0,
) -> tuple[Any, Any]:
    """
    Divide um texto em treino/validação e cria DataLoaders (train_loader e val_loader).

    Parâmetros:
    ----------
    text_data : str
        Texto completo que será dividido em treino e validação.
    create_dataloader_fn : Callable[..., Any]
        Função responsável por criar o DataLoader (ex.: create_dataloader_v1).
        Deve aceitar os parâmetros usados abaixo (data, batch_size, max_length, stride, drop_last, shuffle, num_workers).
    gpt_config : dict[str, Any]
        Configuração do modelo contendo, no mínimo, a chave "context_length".
        Ex.: GPT_CONFIG_124M["context_length"].
    train_ratio : float, default = 0.90
        Proporção do texto destinada ao conjunto de treino (0 < train_ratio < 1).
    batch_size : int, default = 2
        Tamanho do batch para ambos os loaders.
    stride : int | None, default = None
        Stride usado na criação do DataLoader.
        Se None, usa gpt_config["context_length"] (igual ao seu exemplo).
    drop_last_train : bool, default = True
        Se True, descarta o último batch incompleto no treino.
    drop_last_val : bool, default = False
        Se True, descarta o último batch incompleto na validação.
    shuffle_train : bool, default = True
        Se True, embaralha batches no treino.
    shuffle_val : bool, default = False
        Se True, embaralha batches na validação.
    num_workers : int, default = 0
        Número de workers do DataLoader.

    Retorno:
    -------
    tuple[Any, Any]
        (train_loader, val_loader)

    Exceções:
    --------
    Levanta:
    - TypeError se tipos forem inválidos.
    - ValueError se parâmetros numéricos estiverem fora do esperado.
    - KeyError se "context_length" não existir em gpt_config.
    """
    if not isinstance(text_data, str):
        raise TypeError("O parâmetro 'text_data' deve ser uma string.")
    if not callable(create_dataloader_fn):
        raise TypeError(
            "O parâmetro 'create_dataloader_fn' deve ser uma função chamável."
        )
    if not isinstance(gpt_config, dict):
        raise TypeError("O parâmetro 'gpt_config' deve ser um dicionário.")
    if "context_length" not in gpt_config:
        raise KeyError("gpt_config deve conter a chave 'context_length'.")
    if not isinstance(train_ratio, (float, int)):
        raise TypeError("O parâmetro 'train_ratio' deve ser float.")
    train_ratio = float(train_ratio)
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio deve estar no intervalo (0, 1).")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size deve ser um inteiro positivo.")
    if not isinstance(num_workers, int) or num_workers < 0:
        raise ValueError("num_workers deve ser um inteiro >= 0.")

    context_length = int(gpt_config["context_length"])
    if context_length <= 0:
        raise ValueError("gpt_config['context_length'] deve ser um inteiro positivo.")

    stride = context_length if stride is None else int(stride)
    if stride <= 0:
        raise ValueError("stride deve ser um inteiro positivo.")

    # Divide o texto em treino e validação (por caracteres, como no seu exemplo)
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    # Cria os DataLoaders
    train_loader = create_dataloader_fn(
        train_data,
        batch_size=batch_size,
        max_length=context_length,
        stride=stride,
        drop_last=drop_last_train,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )

    val_loader = create_dataloader_fn(
        val_data,
        batch_size=batch_size,
        max_length=context_length,
        stride=stride,
        drop_last=drop_last_val,
        shuffle=shuffle_val,
        num_workers=num_workers,
    )

    return train_loader, val_loader


def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Calcula a loss (cross-entropy) para um único batch.

    Parâmetros:
    ----------
    input_batch : torch.Tensor
        Tensor de entradas (tokens) do batch.
    target_batch : torch.Tensor
        Tensor de targets do batch.
    model : nn.Module
        Modelo que retorna logits no forward.
    device : torch.device
        Dispositivo de execução (CPU/GPU).

    Retorno:
    -------
    torch.Tensor
        Loss do batch (tensor escalar).
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)  # esperado: (B, T, V)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),  # (B*T, V)
        target_batch.flatten(),  # (B*T,)
    )
    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: int | None = None,
) -> float:
    """
    Calcula a loss média ao longo de um DataLoader.

    Parâmetros:
    ----------
    data_loader : DataLoader
        DataLoader que retorna (input_batch, target_batch).
    model : nn.Module
        Modelo que retorna logits no forward.
    device : torch.device
        Dispositivo de execução (CPU/GPU).
    num_batches : int | None, default = None
        Se fornecido, limita o cálculo aos primeiros `num_batches` batches.

    Retorno:
    -------
    float
        Loss média no conjunto avaliado. Retorna NaN se o loader estiver vazio.
    """
    if len(data_loader) == 0:
        return float("nan")

    max_batches = (
        len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    )

    total_loss = 0.0
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= max_batches:
            break

        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += float(loss.item())

    return total_loss / max_batches


def get_device() -> torch.device:
    """
    Retorna o device apropriado (CUDA se disponível, caso contrário CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model_simple(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    start_context: str,
    tokenizer,
    *,
    checkpoint_dir: str | os.PathLike = "./models/partial",
    checkpoint_prefix: str = "full_model",
    resume_if_possible: bool = True,
) -> tuple[list[float], list[float], list[int]]:
    """
    Treina o modelo e avalia periodicamente, registrando perdas e tokens vistos.

    NOVO:
    - Salva checkpoint ao final de cada época em checkpoint_dir:
        {checkpoint_prefix}_part_XX.pth  (XX = número da época completa)
    - Se resume_if_possible=True, procura o último checkpoint e retoma automaticamente.

    Retorno:
    -------
    (train_losses, val_losses, track_tokens_seen)
    """

    # ---- Estado inicial (novo): tentar retomar do último checkpoint ----
    train_losses: list[float] = []
    val_losses: list[float] = []
    track_tokens_seen: list[int] = []

    tokens_seen = 0
    global_step = -1
    start_epoch_idx = 0  # índice 0-based

    if resume_if_possible:
        latest = _find_latest_checkpoint(checkpoint_dir, checkpoint_prefix)
        if latest is not None:
            payload = _load_checkpoint(
                latest,
                model=model,
                optimizer=optimizer,
                device=device,
            )

            completed_epochs = int(payload.get("completed_epochs", 0))
            # Se o arquivo é part_05 => completed_epochs=5 => próxima época é a 6 (índice 5)
            start_epoch_idx = completed_epochs

            global_step = int(payload.get("global_step", -1))
            tokens_seen = int(payload.get("tokens_seen", 0))

            train_losses = list(payload.get("train_losses", []))
            val_losses = list(payload.get("val_losses", []))
            track_tokens_seen = list(payload.get("track_tokens_seen", []))

            print(
                f"[RESUME] Checkpoint encontrado: {latest.name} | "
                f"épocas completas={completed_epochs} -> retomando na época {completed_epochs + 1}"
            )
        else:
            print("[RESUME] Nenhum checkpoint encontrado. Iniciando do zero.")

    # ---- Loop principal de treinamento ----
    for epoch_idx in range(start_epoch_idx, num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    eval_iter=eval_iter,
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Ep {epoch_idx + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Texto de exemplo no fim da época
        generate_and_print_sample(
            model=model,
            tokenizer=tokenizer,
            device=device,
            start_context=start_context,
        )

        # ---- NOVO: salva checkpoint ao final de cada época ----
        completed_epochs = epoch_idx + 1  # 1..N (épocas COMPLETAS)
        ckpt_name = f"{checkpoint_prefix}_part_{completed_epochs:02d}.pth"
        ckpt_path = Path(checkpoint_dir).expanduser().resolve() / ckpt_name

        _save_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            completed_epochs=completed_epochs,
            global_step=global_step,
            tokens_seen=tokens_seen,
            train_losses=train_losses,
            val_losses=val_losses,
            track_tokens_seen=track_tokens_seen,
        )
        print(f"[CHECKPOINT] Salvo: {ckpt_path}")
        print(f"[CHECKPOINT] Atualizado: {ckpt_path.parent / 'latest.pth'}")

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    eval_iter: int,
) -> tuple[float, float]:
    """
    Avalia o modelo em treino e validação usando um número limitado de batches.
    """
    model.eval()  # Coloca o modelo em modo de avaliação
    with torch.no_grad():
        train_loss = calc_loss_loader(
            data_loader=train_loader,
            model=model,
            device=device,
            num_batches=eval_iter,
        )
        val_loss = calc_loss_loader(
            data_loader=val_loader,
            model=model,
            device=device,
            num_batches=eval_iter,
        )
    model.train()  # Restaura o modo de treino
    return train_loss, val_loss


def generate_and_print_sample(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    start_context: str,
) -> None:
    """
    Gera um pequeno texto a partir de `start_context` e imprime na tela.
    """
    model.eval()  # Coloca o modelo em modo de avaliação

    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Impressão compacta (sem quebras de linha)

    model.train()  # Restaura o modo de treino


def plot_losses(
    epochs_seen: torch.Tensor,
    tokens_seen: list[int] | torch.Tensor,
    train_losses: list[float],
    val_losses: list[float],
    output_path: str = "loss-plot.pdf",
) -> None:
    """
    Plota as perdas (treino e validação) em função das épocas e adiciona um segundo eixo
    x para a quantidade de tokens vistos.

    Parâmetros:
    ----------
    epochs_seen : torch.Tensor
        Valores no eixo x representando as épocas correspondentes às medições.
    tokens_seen : list[int] | torch.Tensor
        Quantidade acumulada de tokens processados nas medições.
    train_losses : list[float]
        Lista de perdas de treino.
    val_losses : list[float]
        Lista de perdas de validação.
    output_path : str, default = "loss-plot.pdf"
        Caminho do arquivo de saída para salvar o gráfico.
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plota as losses de treino e validação em função das épocas
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )  # Mostra apenas inteiros no eixo x

    # Cria um segundo eixo x para tokens vistos
    ax2 = ax1.twiny()  # Cria um segundo eixo x compartilhando o mesmo eixo y
    ax2.plot(
        tokens_seen, train_losses, alpha=0
    )  # Plot invisível apenas para alinhar os ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Ajusta o layout para caber bem
    plt.savefig(output_path)
    plt.show()
