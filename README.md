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
- imagens/: Contém imagens relacionadas ao projeto.
- `models/`: Contém os binários dos modelos pré-treinados.
- `notebooks/`: Contém notebooks Jupyter para experimentação e visualização.
- `tests/`: Contém testes unitários para o código-fonte.

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