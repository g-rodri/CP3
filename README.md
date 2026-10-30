# Dashboard de Queimadas na Amazônia (CP3)

Dashboard interativa desenvolvida em **Streamlit** para análise histórica (1998–2017) e previsão de focos de incêndios florestais no Brasil, utilizando modelos de Machine Learning (Regressão Linear, XGBoost e Rede Neural com TensorFlow/Keras).

## Pré-requisitos

- [Python 3.9+](https://www.python.org/downloads/)
- [Git](https://git-scm.com/)

## Como rodar localmente

### 1. Clone o repositório
```bash
git clone https://github.com/g-rodri/CP3.git
cd CP3
```

### 2. Faça um ambiente virtual
```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

# 3. Instale as dependências
```bash
pip install -r requirements.txt
```
# 4. Rode a aplicação
```bash
streamlit run "app.py"
```
