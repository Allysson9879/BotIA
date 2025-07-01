from flask import Flask, render_template, request, jsonify
from embeddings import carregar_dados, responder_pergunta, adicionar_conhecimento
import os

app = Flask(__name__)

dados, vetores, modelo = carregar_dados()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/perguntar", methods=["POST"])
def perguntar():
    global dados, vetores, modelo
    pergunta = request.json.get("pergunta", "")
    resposta = responder_pergunta(pergunta, dados, vetores, modelo)
    return jsonify({"resposta": resposta})

@app.route("/feedback", methods=["POST"])
def feedback():
    global dados, vetores, modelo
    data = request.json
    pergunta = data.get("pergunta", "")
    resposta_correta = data.get("resposta_correta", "")
    adicionar_conhecimento(pergunta, resposta_correta)
    dados, vetores, modelo = carregar_dados()
    return jsonify({"status": "aprendido"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
