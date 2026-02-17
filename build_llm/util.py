import tiktoken
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def generate_text_simple(
    model: nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """
    Gera texto de forma simples (greedy decoding), adicionando tokens um a um.

    A cada iteração:
    - Recorta o contexto para no máximo `context_size` tokens (janela deslizante).
    - Calcula os logits do modelo.
    - Usa apenas o último passo de tempo (último token) para decidir o próximo token.
    - Aplica softmax e seleciona o token de maior probabilidade (argmax).
    - Concatena o token escolhido à sequência.

    Parâmetros:
    ----------
    model : nn.Module
        Modelo que recebe um tensor de IDs (batch_size, seq_len) e retorna logits
        (batch_size, seq_len, vocab_size).
    idx : torch.Tensor
        Tensor de IDs de tokens do contexto atual, com shape (batch_size, n_tokens).
        Deve ser inteiro (tipicamente torch.long).
    max_new_tokens : int
        Número máximo de novos tokens a serem gerados.
    context_size : int
        Tamanho máximo do contexto suportado (janela usada como entrada do modelo).

    Retorno:
    -------
    torch.Tensor
        Tensor com a sequência estendida, shape (batch_size, n_tokens + max_new_tokens).

    Exceções:
    --------
    Levanta TypeError/ValueError para entradas inválidas (tipos, shapes, valores).
    """
    if not isinstance(model, nn.Module):
        raise TypeError("`model` deve ser uma instância de torch.nn.Module.")
    if not isinstance(idx, torch.Tensor):
        raise TypeError("`idx` deve ser um torch.Tensor.")
    if idx.ndim != 2:
        raise ValueError("`idx` deve ter shape (batch_size, n_tokens).")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("`max_new_tokens` deve ser um inteiro >= 0.")
    if not isinstance(context_size, int) or context_size <= 0:
        raise ValueError("`context_size` deve ser um inteiro > 0.")

    # Garante que idx esteja em um dtype inteiro comum para embeddings
    if idx.dtype not in (
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
    ):
        raise TypeError("`idx` deve ser um tensor de inteiros (ex.: torch.long).")

    for _ in range(max_new_tokens):
        # Recorta o contexto para caber no tamanho máximo suportado
        idx_cond = idx[:, -context_size:]

        # Predição sem gradientes
        with torch.no_grad():
            logits = model(idx_cond)

        # Usa apenas o último passo de tempo: (B, T, V) -> (B, V)
        logits_last = logits[:, -1, :]

        # Probabilidades do próximo token: (B, V)
        probas = torch.softmax(logits_last, dim=-1)

        # Escolha greedy: token com maior probabilidade (B, 1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # Concatena o próximo token à sequência: (B, T) -> (B, T+1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


class GPTDatasetV1(Dataset):
    """
    Dataset para treino de modelos estilo GPT usando janelas deslizantes (sliding window).

    Este dataset:
    1) Tokeniza o texto completo com um tokenizer compatível (ex.: tiktoken).
    2) Divide a sequência de tokens em janelas sobrepostas de tamanho `max_length`.
    3) Para cada janela, cria:
       - input_ids  = tokens[i : i + max_length]
       - target_ids = tokens[i + 1 : i + max_length + 1]
    Isso prepara o formato clássico de predição do próximo token.

    Parâmetros:
    ----------
    txt : str
        Texto bruto a ser tokenizado.
    tokenizer : object
        Tokenizer com método `encode(str, allowed_special=...) -> List[int]`.
        Ex.: `tiktoken.get_encoding("gpt2")`.
    max_length : int
        Número de tokens por janela (comprimento do contexto).
    stride : int
        Passo do sliding window. `stride < max_length` gera sobreposição.

    Retorno (por item):
    -------------------
    Tuple[Tensor, Tensor]
        (input_ids, target_ids), ambos do tipo torch.LongTensor com shape (max_length,).

    Exceções:
    --------
    Levanta TypeError se `txt` não for string ou se `max_length/stride` não forem inteiros.
    Levanta ValueError se `max_length/stride` forem inválidos ou se o texto for curto demais.
    """

    def __init__(self, txt: str, tokenizer, max_length: int, stride: int) -> None:
        if not isinstance(txt, str):
            raise TypeError("`txt` deve ser uma string.")
        if not isinstance(max_length, int) or not isinstance(stride, int):
            raise TypeError("`max_length` e `stride` devem ser inteiros.")
        if max_length <= 0:
            raise ValueError("`max_length` deve ser maior que zero.")
        if stride <= 0:
            raise ValueError("`stride` deve ser maior que zero.")

        self.input_ids: list[Tensor] = []
        self.target_ids: list[Tensor] = []

        # Tokeniza o texto completo
        try:
            token_ids: list[int] = tokenizer.encode(
                txt,
                allowed_special={"<|endoftext|>"},
            )
        except Exception as e:
            raise TypeError(
                "Falha ao tokenizar o texto. Verifique se `tokenizer` possui o método "
                "`encode(txt, allowed_special=...)`."
            ) from e

        # Precisamos de pelo menos max_length + 1 tokens para montar (input, target)
        if len(token_ids) <= max_length:
            raise ValueError(
                "O texto tokenizado é curto demais: é necessário ter pelo menos "
                "`max_length + 1` tokens para criar uma amostra."
            )

        # Cria janelas deslizantes: input (t...t+max_length-1) e target (t+1...t+max_length)
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

        if len(self.input_ids) == 0:
            raise ValueError(
                "Nenhuma amostra foi gerada. Isso pode acontecer se o texto for curto "
                "ou se `stride` for grande demais para o tamanho disponível."
            )

    def __len__(self) -> int:
        """
        Retorna o número de amostras (janelas) geradas.
        """
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Retorna o par (input_ids, target_ids) para o índice `idx`.

        Parâmetros:
        ----------
        idx : int
            Índice da amostra.

        Retorno:
        -------
        Tuple[Tensor, Tensor]
            (input_ids, target_ids) com shape (max_length,) cada.
        """
        if not isinstance(idx, int):
            raise TypeError("`idx` deve ser um inteiro.")
        if idx < 0 or idx >= len(self.input_ids):
            raise IndexError("Índice fora do intervalo do dataset.")

        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Cria um DataLoader para treinar/avaliar um modelo estilo GPT a partir de um texto bruto.

    O texto é tokenizado com o tokenizer do GPT-2 (tiktoken) e então segmentado em janelas
    de tamanho `max_length` com deslocamento `stride` (sliding window), produzindo pares
    (input_ids, target_ids) via `GPTDatasetV1`.

    Parâmetros:
    ----------
    txt : str
        Texto bruto que será tokenizado e usado para construir o dataset.
    batch_size : int, default = 4
        Tamanho do batch do DataLoader.
    max_length : int, default = 256
        Comprimento máximo (número de tokens) por amostra/janela.
    stride : int, default = 128
        Passo do sliding window entre janelas consecutivas.
    shuffle : bool, default = True
        Se True, embaralha as amostras a cada época.
    drop_last : bool, default = True
        Se True, descarta o último batch caso ele seja menor que `batch_size`.
    num_workers : int, default = 0
        Número de subprocessos usados para carregar os dados.

    Retorno:
    -------
    DataLoader
        DataLoader pronto para iterar em batches do dataset `GPTDatasetV1`.

    Exceções:
    --------
    Levanta TypeError se `txt` não for uma string.
    Levanta ValueError se `batch_size`, `max_length`, `stride` ou `num_workers` forem inválidos.
    """
    if not isinstance(txt, str):
        raise TypeError("`txt` deve ser uma string.")

    for name, value in {
        "batch_size": batch_size,
        "max_length": max_length,
        "stride": stride,
        "num_workers": num_workers,
    }.items():
        if not isinstance(value, int):
            raise TypeError(f"`{name}` deve ser um inteiro.")
        if value < 0:
            raise ValueError(f"`{name}` não pode ser negativo.")
        if name in {"batch_size", "max_length", "stride"} and value == 0:
            raise ValueError(f"`{name}` deve ser maior que zero.")

    if not isinstance(shuffle, bool):
        raise TypeError("`shuffle` deve ser booleano (True/False).")
    if not isinstance(drop_last, bool):
        raise TypeError("`drop_last` deve ser booleano (True/False).")

    try:
        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )
        return dataloader
    except Exception as e:
        raise RuntimeError("Falha ao criar o DataLoader.") from e


def text_to_token_ids(text: str, tokenizer) -> torch.Tensor:
    """
    Converte um texto em IDs de tokens e retorna um tensor (com dimensão de batch).

    Parâmetros:
    ----------
    text : str
        Texto de entrada a ser tokenizado.
    tokenizer : Any
        Tokenizer compatível com a API do `tiktoken` (precisa ter `.encode`).

    Retorno:
    -------
    torch.Tensor
        Tensor com shape (1, seq_len), contendo os IDs de tokens.

    Exceções:
    --------
    Levanta TypeError se `text` não for string.
    Levanta RuntimeError se falhar ao codificar o texto.
    """
    if not isinstance(text, str):
        raise TypeError("O parâmetro `text` deve ser do tipo str.")

    try:
        encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        return torch.tensor(encoded, dtype=torch.long).unsqueeze(0)  # adiciona batch
    except Exception as e:
        raise RuntimeError("Falha ao converter texto em token IDs.") from e


def token_ids_to_text(token_ids: torch.Tensor, tokenizer) -> str:
    """
    Converte um tensor de IDs de tokens (com dimensão de batch) de volta para texto.

    Parâmetros:
    ----------
    token_ids : torch.Tensor
        Tensor com shape (1, seq_len) (ou compatível) contendo IDs de tokens.
    tokenizer : Any
        Tokenizer compatível com a API do `tiktoken` (precisa ter `.decode`).

    Retorno:
    -------
    str
        Texto decodificado.

    Exceções:
    --------
    Levanta TypeError se `token_ids` não for um torch.Tensor.
    Levanta ValueError se o tensor estiver vazio.
    """
    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("O parâmetro `token_ids` deve ser um torch.Tensor.")
    if token_ids.numel() == 0:
        raise ValueError("`token_ids` não pode estar vazio.")

    flat = token_ids.squeeze(0)  # remove batch
    return tokenizer.decode(flat.tolist())
