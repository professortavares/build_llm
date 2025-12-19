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

