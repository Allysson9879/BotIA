import json
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import heapq

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

def responder_pergunta(pergunta, dados, vetores, modelo, top_n=3):
    if not dados:
        return "🤖 Ainda não aprendi nada. Me ensine algo, por favor! 🙏"
    pergunta_vetor = modelo.encode([pergunta])
    similaridades = cosine_similarity(pergunta_vetor, vetores)[0]
    top_indices = heapq.nlargest(top_n, range(len(similaridades)), key=similaridades.__getitem__)
    if similaridades[top_indices[0]] < 0.4:
        return ("🤔 Desculpe, não consegui encontrar uma resposta clara. "
                "Você pode tentar reformular sua pergunta ou me ensinar! 😊")
    respostas = [dados[i] for i in top_indices]
    resposta_completa = " ".join(respostas)
    resposta_final = (f"✨ Baseado no que aprendi, aqui está o que posso te dizer:

"
                      f"{resposta_completa}

"
                      "Se quiser, pergunte outra coisa ou me ensine algo novo! 🚀")
    return resposta_final

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
