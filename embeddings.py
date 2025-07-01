import json
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

CAMINHO_JSON = "banco.json"
CAMINHO_PKL = "vetor_index.pkl"

def carregar_dados():
    if not os.path.exists(CAMINHO_JSON):
        with open(CAMINHO_JSON, "w", encoding="utf-8") as f:
            json.dump([], f)

    with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
        dados = json.load(f)

    if os.path.exists(CAMINHO_PKL):
        with open(CAMINHO_PKL, "rb") as f:
            vetores = pickle.load(f)
    else:
        modelo = SentenceTransformer("all-MiniLM-L6-v2")
        vetores = modelo.encode(dados)
        with open(CAMINHO_PKL, "wb") as f:
            pickle.dump(vetores, f)

    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    return dados, vetores, modelo

def responder_pergunta(pergunta, dados, vetores, modelo):
    if not dados:
        return "Ainda não aprendi nada. Me ensine algo primeiro!"

    pergunta_vetor = model.encode([pergunta])
    similaridades = cosine_similarity(pergunta_vetor, vetores)[0]
    indice = np.argmax(similaridades)
    score = similaridades[indice]

    if score > 0.5:
        return dados[indice]
    else:
        return "Não encontrei uma resposta clara com base no que aprendi."

def adicionar_conhecimento(pergunta, resposta):
    if os.path.exists(CAMINHO_JSON):
        with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
            dados = json.load(f)
    else:
        dados = []

    novo_dado = f"{pergunta.strip()} | {resposta.strip()}"
    dados.append(novo_dado)

    with open(CAMINHO_JSON, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=4, ensure_ascii=False)

    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    vetores = modelo.encode(dados)

    with open(CAMINHO_PKL, "wb") as f:
        pickle.dump(vetores, f)