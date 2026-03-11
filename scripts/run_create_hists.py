from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
from typing import List

import requests


# =========================
# CONFIGURAÇÕES
# =========================
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"
OUTPUT_DIR = Path("./data/raw2")
NUM_HISTORIAS_POR_TEMA = 5
TIMEOUT = 300


# =========================
# TEMAS (30)
# =========================
TEMAS: List[str] = [
    "amizade verdadeira",
    "aventura no espaço",
    "mistério em uma cidade pequena",
    "superação após uma perda",
    "descoberta de um talento escondido",
    "viagem no tempo",
    "um robô aprendendo emoções",
    "vida no fundo do mar",
    "fantasia medieval",
    "sobrevivência em uma ilha deserta",
    "um segredo de família",
    "o último livro de uma biblioteca mágica",
    "um inventor excêntrico",
    "a primeira missão em Marte",
    "conflito entre humanos e inteligência artificial",
    "infância em uma vila distante",
    "um herói improvável",
    "um detetive iniciante",
    "reencontro depois de muitos anos",
    "uma floresta encantada",
    "um diário encontrado no sótão",
    "corrida contra o tempo",
    "vida após uma grande catástrofe",
    "um gato que muda destinos",
    "um reino submerso",
    "a última carta nunca enviada",
    "amizade entre espécies diferentes",
    "um músico em busca da canção perfeita",
    "um portal para outro mundo",
    "o preço de um desejo realizado",
]


# =========================
# PROMPTS
# =========================
SYSTEM_PROMPT = """
Você é um escritor especializado em criar histórias originais para composição de dataset de treinamento de modelos de linguagem.

Regras obrigatórias:
1. Escreva em português do Brasil.
2. Crie uma história completa com início, meio e fim.
3. Use linguagem clara, natural e coesa.
4. Entregue a resposta em Markdown.
5. A estrutura da resposta deve ser:

# Título

## Tema
<tema>

## História
<texto da história>

## Estrutura narrativa
- Início: ...
- Meio: ...
- Fim: ...

## Metadados
- Gênero:
- Tom:
- Público sugerido:
- Palavras-chave:

6. Não explique o processo.
7. Não use disclaimers.
8. Não repita frases desnecessariamente.
9. Gere conteúdo original.
10. A história deve ter entre 3000 e 3500 palavras.
""".strip()


def montar_prompt(tema: str) -> str:
    return f"""
Crie uma história original baseada no tema: "{tema}".

Objetivo:
Gerar texto de boa qualidade para treinamento de outra LLM.

Requisitos:
- A história deve ter começo, desenvolvimento e desfecho bem definidos.
- O texto deve ser totalmente em Markdown.
- Evite clichês excessivos e frases muito genéricas.
- Crie personagens, contexto, conflito e resolução.
- A narrativa deve ser autossuficiente.
- Não cite que foi criada por IA.
- Não use listas dentro da seção "História"; escreva em prosa.
- Mantenha consistência de nomes, tempo e ambiente.

Capriche na qualidade literária, mas preserve clareza e objetividade.
""".strip()


# =========================
# UTILITÁRIOS
# =========================
def slugify(texto: str) -> str:
    texto = texto.lower().strip()
    texto = re.sub(r"[^\w\s-]", "", texto, flags=re.UNICODE)
    texto = re.sub(r"[\s_-]+", "_", texto)
    return texto.strip("_")


def chamar_ollama(model: str, system: str, prompt: str) -> str:
    payload = {
        "model": model,
        "system": system,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    response.raise_for_status()

    data = response.json()

    # No /api/generate, a resposta principal costuma vir em "response"
    texto = data.get("response", "").strip()
    if not texto:
        raise RuntimeError("O Ollama retornou uma resposta vazia.")

    return texto


def salvar_markdown(tema: str, indice: int, conteudo: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{slugify(tema)}_{indice:02d}_{timestamp}.md"
    caminho = OUTPUT_DIR / nome_arquivo

    caminho.write_text(conteudo, encoding="utf-8")
    return caminho


# =========================
# EXECUÇÃO
# =========================
def main() -> None:
    print(f"Modelo: {MODEL_NAME}")
    print(f"Saída: {OUTPUT_DIR.resolve()}")
    print("-" * 60)

    total = 0
    falhas = 0

    for tema in TEMAS:
        for i in range(1, NUM_HISTORIAS_POR_TEMA + 1):
            try:
                prompt = montar_prompt(tema)
                historia_md = chamar_ollama(
                    model=MODEL_NAME,
                    system=SYSTEM_PROMPT,
                    prompt=prompt,
                )
                arquivo = salvar_markdown(tema, i, historia_md)
                total += 1
                print(f"[OK] {tema} -> {arquivo.name}")
            except Exception as e:
                falhas += 1
                print(f"[ERRO] {tema} ({i}): {e}")

    print("-" * 60)
    print(f"Histórias geradas: {total}")
    print(f"Falhas: {falhas}")


if __name__ == "__main__":
    main()