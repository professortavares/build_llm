# Build LLM

Este repositório contém um guia passo a passo para construir e treinar um modelo de linguagem grande (LLM) 
usando Python e bibliotecas populares de aprendizado de máquina.

## Instalação
**Observação:** testado em python 3.10.

Certifique-se de ter Python 3.10 ou superior instalado. Em seguida, instale as dependências necessárias:

Crie o ambiente virtual (py env):
```bash
python -m venv .venv
```
Ative o ambiente virtual (linux/macOS):
```bash
source .venv/bin/activate
```

Ative o ambiente virtual (Windows):
```bash
.venv\Scripts\activate
```

Instale as bibliotecas necessárias (apenas cpu / sem suporte gpu):
```bash
pip install -r requirements.txt
```

ou para instalações com gpu:
```bash
pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```
Em seguida as demais dependências:
```bash
pip install -r requirements_gpu.txt
```
  

Instale o projeto em modo de desenvolvimento:
```bash
pip install -e .
```

## Qualidade do código

Para garantir a qualidade do código, utilize as seguintes ferramentas:

```
ruff check . --fix --unsafe-fixes
ruff format . 
mypy 
pytest test
```

## Esturtura do Projeto

- `build_llm/`: Contém o código-fonte para construição e treinamento do LLM.
- `data/`: Contém os arquivos de dados que serão utilizados para treinar o LLM.
- `imagens/`: Contém imagens relacionadas ao projeto.
- `models/`: Contém os binários dos modelos pré-treinados.
- `notebooks/`: Contém notebooks Jupyter para experimentação e visualização.
- `tests/`: Contém testes unitários para o código-fonte.

### Tabela de conteúdo:

**Capítulo 02: Pré-processamento dos dados**

| # | Notebook                                                                                                        | Descrição                                                                                                                                                                                                     |
|---|-----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | [Tokenizador simples](./notebooks/cap02/01%20-%20Tokenizador%20simples.ipynb)                                   | Implementa um **tokenizador simples**, convertendo texto em tokens e IDs numéricos. Serve como base para entender tokenização antes de abordagens mais avançadas.                                             |
| 2 | [Encoder simples](./notebooks/cap02/02%20-%20Encoder%20simples.ipynb)                                           | Apresenta um **encoder simples**, transformando tokens em representações vetoriais numéricas. Introduz a ideia de embeddings como base para modelos de linguagem.                                             | 
| 3 | [Encoder melhorado](./notebooks/cap02/03%20-%20Encoder%20melhorado.ipynb)                                       | Apresenta um **encoder melhorado**, adicionando refinamentos ao encoder simples para gerar representações mais robustas dos tokens. Explora melhorias práticas comuns em pipelines de processamento de texto. |
| 4 | [BPE Tokenizer](./notebooks/cap02/04%20-%20BPE%20Tokenizer.ipynb)                                               | Implementa um **tokenizador BPE (Byte Pair Encoding)**, uma técnica avançada de tokenização que combina subunidades de palavras para melhorar a eficiência e cobertura do vocabulário.                        |
| 5 | [Perceptron](./notebooks/cap02/05%20-%20Perceptron.ipynb)                                                       | Introduz o **perceptron**, um modelo de rede neural simples usado para tarefas de classificação. Serve como base para entender conceitos fundamentais de redes neurais.                                       |
| 6 | [Redes neurais - MLP](./notebooks/cap02/06%20-%20Redes%20neurais%20-%20MLP.ipynb)                               | Apresenta **redes neurais feedforward (MLP)**, explorando arquiteturas mais complexas que o perceptron. Introduz conceitos de camadas ocultas e funções de ativação.                                          |
| 7 | [PyTorch MLP](./notebooks/cap02/07%20-%20PyTorch%20MLP.ipynb)                                                   | Implementa um **MLP usando PyTorch**, demonstrando como construir e treinar redes neurais com uma biblioteca popular de aprendizado de máquina.                                                               |
| 8 | [CBoW](./notebooks/cap02/08%20-%20CBoW.ipynb)                                                                   | Apresenta o modelo **Continuous Bag of Words (CBoW)**, uma técnica de modelagem de linguagem que prevê palavras com base no contexto circundante.                                                             |
| 9 | [Pipeline BPE + CBoW](./notebooks/cap02/09%20-%20Pipeline%20BPE%20+%20CBoW.ipynb)                               | Combina o **tokenizador BPE com o modelo CBoW**, demonstrando um pipeline completo de pré-processamento e modelagem de linguagem.                                                                             |
|10 | [Pipeline CBoW + Embedding PyTorch](./notebooks/cap02/10%20-%20Pipeline%20CBoW%20+%20Embedding%20PyTorch.ipynb) | Integra o **modelo CBoW com embeddings do PyTorch**, mostrando como utilizar embeddings pré-treinados em um pipeline de modelagem de linguagem.                                                               |
|11 | [Pipeline com tokens positions](./notebooks/cap02/11%20-%20Pipeline%20com%20tokens%20positions.ipynb)           | Adiciona **informações de posição dos tokens** ao pipeline, permitindo que o modelo capture a ordem das palavras no texto.                                                                                    |

**Capítulo 03: Modelos de atenção**

| # | Notebook                                                                                                     | Descrição                                                                                                                                                                   |
|---|--------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | [Camada de atenção simples](./notebooks/cap03/01%20-%20S01%20-%20Simplified%20Self%20Attention.ipynb)        | Implementa uma **camada de atenção simples**, introduzindo o mecanismo de atenção que permite ao modelo focar em diferentes partes da entrada ao fazer previsões.           |
| 2 | [Camada de atenção treinavel](./notebooks/cap03/02%20-%20Self%20Attention%20Trainable.ipynb)                 | Apresenta uma **camada de atenção treinável**, permitindo que o modelo aprenda pesos de atenção durante o treinamento para melhorar o desempenho.                           |
| 3 | [Camada de atenção causal](./notebooks/cap03/03%20-%20Self%20Attention%20-%20Hidding%20future%20words.ipynb) | Implementa uma **camada de atenção causal**, garantindo que o modelo não tenha acesso a informações futuras ao fazer previsões, essencial para tarefas de geração de texto. |
| 4 | [Multi-Head Attention](./notebooks/cap03/04%20-%20Multi-head%20attention.ipynb)                              | Apresenta o **mecanismo de Multi-Head Attention**, permitindo que o modelo capture múltiplas relações contextuais simultaneamente para melhorar a compreensão do texto.     |


## Referências gerais

O código aqui presente é baseado nas seguintes referências:

- Livro "Building Large Language from Scratch" de Sebastian Raschka.
[Building Large Language Models from Scratch](https://www.amazon.com.br/Build-Large-Language-Model-Scratch/dp/B0DNR5SPQR)

- Repositório oficial do livro no GitHub: [repo no github](https://github.com/rasbt/LLMs-from-scratch)

- Stanford CS336 Language Modeling from Scratch Course: [youtube](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)

- Material da aula: [material](https://stanford-cs336.github.io/spring2025/)

- Playlist Understanding Large Language Model [youtube](https://www.youtube.com/playlist?list=PLUfbC589u-FSwnqsvTHXVcgmLg8UnbIy3)

### Referências específicas

BPE: 
* Philip Gage: **A New Algorithm for Data Compression** (1994) [artigo](https://www.derczynski.com/papers/archive/BPE_Gage.pdf)

Word2Vec:
* Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean: **Efficient Estimation of Word Representations in Vector Space** (2013) [artigo](https://arxiv.org/abs/1301.3781)

Attention Is All You Need:
* Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin **Attention Is All You Need** (2023) [artigo](https://arxiv.org/abs/1706.03762)

Arquitetura LLM / GPT-2:
* Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever: **Language Models are Unsupervised Multitask Learners** (2019) [artigo](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Layer Normalization:
* Jimmy Lei Ba and Jamie Ryan Kiros and Geoffrey E. Hinton **Layer Normalization** (2016) [artigo](https://arxiv.org/abs/1607.06450)

