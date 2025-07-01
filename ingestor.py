import json
from sentence_transformers import SentenceTransformer
import pickle

def quebrar_texto(texto, tamanho=300):
    partes = []
    atual = ""
    for linha in texto.split("\n"):
        if len(atual) + len(linha) < tamanho:
            atual += " " + linha.strip()
        else:
            partes.append(atual.strip())
            atual = linha.strip()
    if atual:
        partes.append(atual.strip())
    return partes

def ingestar_texto(texto):
    dados = quebrar_texto(texto)
    with open("banco.json", "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=4, ensure_ascii=False)

    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    vetores = modelo.encode(dados)

    with open("vetor_index.pkl", "wb") as f:
        pickle.dump(vetores, f)

    print(f"Ingestão completa: {len(dados)} pedaços armazenados.")