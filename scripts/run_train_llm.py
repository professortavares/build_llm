import time
from typing import Any

import tiktoken
import torch
from reuse import (
    baixar_livros_machado_assis_gutenberg,
    calc_loss_loader,
    concatenar_livros_em_uma_string,
    criar_train_val_dataloaders,
    get_device,
    plot_losses,
    preprocessar_livros_gutenberg_por_linha,
    train_model_simple,
)

from build_llm.gpt import GPTModel
from build_llm.util import (
    create_dataloader_v1,
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text,
)


def baixar_livros():
    # baixando os livros do Machado de Assis do Projeto Gutenberg
    livros = {
        "Dom Casmurro": "https://www.gutenberg.org/cache/epub/55752/pg55752.txt",
        "Memorias Posthumas de Braz Cubas": "https://www.gutenberg.org/cache/epub/54829/pg54829.txt",
        "Poesias Completas": "https://www.gutenberg.org/cache/epub/61653/pg61653.txt",
        "Quincas Borba": "https://www.gutenberg.org/cache/epub/55682/pg55682.txt",
        "Os Trabalhadores do Mar": "https://www.gutenberg.org/cache/epub/57895/pg57895.txt",
        "Papeis Avulsos": "https://www.gutenberg.org/cache/epub/57001/pg57001.txt",
        "Helena": "https://www.gutenberg.org/cache/epub/67162/pg67162.txt",
        "Historias Sem Data": "https://www.gutenberg.org/cache/epub/33056/pg33056.txt",
        "A Mao e A Luva": "https://www.gutenberg.org/cache/epub/53101/pg53101.txt",
        "Esau e Jacob": "https://www.gutenberg.org/cache/epub/56737/pg56737.txt",
        "Reliquias de Casa Velha": "https://www.gutenberg.org/cache/epub/67935/pg67935.txt",
        "Memorial de Ayres": "https://www.gutenberg.org/cache/epub/55797/pg55797.txt",
        "Quéda que as Mulheres Têm para os Tolos": "https://www.gutenberg.org/cache/epub/59620/pg59620.txt",
        "Yayá Garcia": "https://www.gutenberg.org/cache/epub/67780/pg67780.txt",
    }

    resultados = baixar_livros_machado_assis_gutenberg(
        base_dir="./data/raw/",
        livros=livros,
        timeout=30,
        overwrite=False,
    )
    print("\nArquivos:")
    for titulo, caminho in resultados:
        print(f"- {titulo}: {caminho}")


def preprocessar_livros():
    # realiza o pré-processamento dos livros (removendo cabeçalho e rodapé) usando os cortes definidos abaixo
    cortes = {
        "a_mao_e_a_luva.txt": (27, 4410),
        "dom_casmurro.txt": (28, 8732),
        "esau_e_jacob.txt": (31, 8305),
        "helena.txt": (35, 7210),
        "historias_sem_data.txt": (27, 5743),
        "memorial_de_ayres.txt": (28, 6270),
        "memorias_posthumas_de_braz_cubas.txt": (27, 8469),
        "os_trabalhadores_do_mar.txt": (38, 16365),
        "papeis_avulsos.txt": (37, 6480),
        "poesias_completas.txt": (33, 10582),
        "queda_que_as_mulheres_tem_para_os_tolos.txt": (41, 557),
        "quincas_borba.txt": (28, 10482),
        "reliquias_de_casa_velha.txt": (54, 8174),
        "yaya_garcia.txt": (35, 6951),
    }

    saidas = preprocessar_livros_gutenberg_por_linha(
        raw_dir="./data/raw",
        out_dir="./data/preprocessed",
        cortes=cortes,
        overwrite=True,
        debug=True,
    )
    print("\nPreprocessados:")
    for fname, outp in saidas:
        print(
            f"- {fname} -> {outp}"
        )  # pré-processando os livros (removendo cabeçalho e rodapé)


def concatenar_livros_preprocessados():
    corpus = concatenar_livros_em_uma_string("./data/preprocessed")
    print("Tamanho do corpus (chars):", len(corpus))
    return corpus


def verificar_presenca_gpu():
    print(torch.__version__)  # Versão do torch
    print(torch.cuda.is_available())  # Verificação de GPU
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando:", device)


def set_seed(seed: int = 42) -> None:
    """
    Fixa seeds para reprodutibilidade em Python, NumPy e PyTorch.
    """
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Garante determinismo (pode afetar performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # BAIXAR E PRE-PROCESSAR A BASE DE DADOS
    # executa o processo completo de download e pré-processamento dos livros do Machado de Assis
    baixar_livros()
    # pré-processando os livros (removendo cabeçalho e rodapé)
    preprocessar_livros()
    # concatenando os livros pré-processados em uma única string
    corpus = concatenar_livros_preprocessados()

    # TREINAR O MODELO
    # Verifica se a GPU está disponível para treinamento
    verificar_presenca_gpu()
    # Fixa seeds para reprodutibilidade
    set_seed(123)

    GPT_CONFIG_124M: dict[str, Any] = {
        "vocab_size": 50257,  # Tamanho do vocabulário
        "context_length": 256,  # Comprimento do contexto
        "emb_dim": 768,  # Dimensão do embedding
        "n_heads": 12,  # Número de cabeças de atenção
        "n_layers": 12,  # Número de camadas
        "drop_rate": 0.1,  # Taxa de dropout
        "qkv_bias": False,  # Viés em Query-Key-Value (QKV)
    }
    # Instancia o modelo com a configuração desejada
    model = GPTModel(GPT_CONFIG_124M)

    train_loader, val_loader = criar_train_val_dataloaders(
        text_data=corpus,
        create_dataloader_fn=create_dataloader_v1,
        gpt_config=GPT_CONFIG_124M,
        train_ratio=0.90,
        batch_size=2,
    )
    # Inspeciona os batches do DataLoader de treino
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    # Inspeciona os batches do DataLoader de validação
    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    # Conta a quantidade de tokens (elementos) nos batches de treino
    train_tokens = 0
    for input_batch, _target_batch in train_loader:
        train_tokens += input_batch.numel()

    # Conta a quantidade de tokens (elementos) nos batches de validação
    val_tokens = 0
    for input_batch, _target_batch in val_loader:
        val_tokens += input_batch.numel()

    # Exibe os totais
    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)
    print("All tokens:", train_tokens + val_tokens)

    # Seleciona o dispositivo (CPU/GPU) e move o modelo
    device = get_device()
    model.to(device)

    # Calcula as losses sem rastrear gradientes
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    ## Início do treinamento
    # Marca o tempo de início
    start_time = time.time()

    # Instancia o modelo e move para o dispositivo (CPU/GPU)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)

    # Configura o otimizador (AdamW)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=4e-4,
        weight_decay=0.1,
    )

    # Treina o modelo
    # Inicializa o tokenizer (encoding compatível com GPT-2)
    tokenizer = tiktoken.get_encoding("gpt2")

    num_epochs = 50
    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Uma noite destas, vindo da cidade para o",
        tokenizer=tokenizer,
        checkpoint_dir="./models/partial",
        checkpoint_prefix="full_model",
        resume_if_possible=True,
    )
    
    # Calcula e exibe o tempo total de execução
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Treino completo em {execution_time_minutes:.2f} minutos.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # Etapa de inferência: gera texto a partir do modelo treinado
    # Define o dispositivo para inferência (CPU)
    inference_device = torch.device("cpu")

    # Move o modelo para o dispositivo de inferência e coloca em modo de avaliação
    model.to(inference_device)
    model.eval()

    # Inicializa o tokenizer (encoding compatível com GPT-2)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Gera texto a partir de um prompt inicial
    prompt = "Uma noite destas, vindo da cidade para o"
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(inference_device),
        max_new_tokens=50,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    # Decodifica e imprime o resultado
    output_text = token_ids_to_text(token_ids, tokenizer)
    print(f"Texto de saída:\n{output_text}")

    # Salva o modelo treinado para uso futuro
    torch.save(model.state_dict(), "./models/full_model.pth")


if __name__ == "__main__":
    main()
