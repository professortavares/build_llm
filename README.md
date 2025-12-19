# Build LLM

Este repositório contém um guia passo a passo para construir e treinar um modelo de linguagem grande (LLM) 
usando Python e bibliotecas populares de aprendizado de máquina.

## Instalação

Certifique-se de ter Python 3.11 ou superior instalado. Em seguida, instale as dependências necessárias:

Crie o ambiente virtual:
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

Instale as bibliotecas necessárias:
```bash
pip install -r requirements.txt
```
Instale o projeto em modo de desenvolvimento:
```bash
pip install -e .
```

## Qualidade do código

Para garantir a qualidade do código, utilize as seguintes ferramentas:

```
ruff check . --fix 
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

## Referências

O código aqui presente é totalmente baseado nas seguintes referências:

- Livro "Building Large Language from Scratch" de Sebastian Raschka.
[Building Large Language Models from Scratch](https://www.amazon.com.br/Build-Large-Language-Model-Scratch/dp/B0DNR5SPQR)

- Repositório oficial do livro no GitHub: [repo no github](https://github.com/rasbt/LLMs-from-scratch)

- Stanford CS336 Language Modeling from Scratch Course: [youtube](https://www.youtube.com/playlist?list=PLoROMvodv4rOY23Y0BoGoBGgQ1zmU_MT_)

- Material da aula: [material](https://stanford-cs336.github.io/spring2025/)