
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Queimadas na Amazônia", layout="wide")

#Carrega dados
@st.cache_data
def load_data():
    df = pd.read_csv("amazon.csv", encoding="latin1")
    return df

df = load_data()

# Estilização customizada da Sidebar e Filtros Fixos
st.markdown("""
<style>
    /* Ajusta o espaçamento da sidebar para caber todos os filtros sem rolagem desnecessária */
    [data-testid="stSidebarContent"] {
        padding-top: 1.5rem !important;
    }
    
    /* Cabeçalho de seção na sidebar */
    .sidebar-section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.5rem;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .state-counter-badge {
        background: rgba(220, 20, 60, 0.18);
        color: #ff6b81;
        border: 1px solid rgba(220, 20, 60, 0.4);
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Estilização dos chips/pills de Estados (fixos e sem barra de rolagem) */
    [data-testid="stPills"] div[role="listbox"] {
        display: grid !important;
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 6px !important;
        max-height: none !important;
        overflow: visible !important;
    }
    [data-testid="stPills"] button {
        width: 100% !important;
        justify-content: center !important;
        border-radius: 8px !important;
        padding: 4px 8px !important;
        min-height: 30px !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        transition: all 0.18s ease !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        background-color: rgba(255, 255, 255, 0.03) !important;
    }
    [data-testid="stPills"] button:hover {
        border-color: rgba(220, 20, 60, 0.6) !important;
        background-color: rgba(220, 20, 60, 0.1) !important;
        transform: translateY(-1px);
    }
    [data-testid="stPills"] button[aria-selected="true"] {
        background: linear-gradient(135deg, rgba(220, 20, 60, 0.28), rgba(220, 20, 60, 0.14)) !important;
        border-color: #dc143c !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(220, 20, 60, 0.2) !important;
    }

    /* Botões rápidos (Todos / Limpar) mais compactos */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] button {
        padding: 0.2rem 0.5rem !important;
        min-height: 28px !important;
        font-size: 0.78rem !important;
        border-radius: 6px !important;
    }
</style>
""", unsafe_allow_html=True)

#Sidebar
st.sidebar.title("Filtros")
st.sidebar.markdown("**Fonte:** [Kaggle – Forest Fires in Brazil](https://www.kaggle.com/datasets/gustavomodelli/forest-fires-in-brazil)")

anos = sorted(df["year"].unique())
estados = sorted(df["state"].unique())

ano_range = st.sidebar.slider("Período (ano)", int(min(anos)), int(max(anos)), (int(min(anos)), int(max(anos))))

# Inicializa estado da seleção de estados
if "estados_pills" not in st.session_state:
    st.session_state["estados_pills"] = estados

# Botões de ação rápida (Selecionar todos / Limpar)
btn_col1, btn_col2 = st.sidebar.columns(2)
if btn_col1.button("✓ Todos", use_container_width=True):
    st.session_state["estados_pills"] = estados
    st.rerun()
if btn_col2.button("✕ Limpar", use_container_width=True):
    st.session_state["estados_pills"] = []
    st.rerun()

qtd_sel = len(st.session_state.get("estados_pills", estados))
st.sidebar.markdown(
    f'<div class="sidebar-section-header">'
    f'<span>Estados</span>'
    f'<span class="state-counter-badge">{qtd_sel}/{len(estados)}</span>'
    f'</div>',
    unsafe_allow_html=True
)

estados_sel = st.sidebar.pills(
    "Estados",
    options=estados,
    selection_mode="multi",
    key="estados_pills",
    label_visibility="collapsed"
)

if not estados_sel:
    st.warning("⚠️ Selecione pelo menos um estado no filtro lateral para visualizar os dados.")
    st.stop()

df_f = df[(df["year"] >= ano_range[0]) & (df["year"] <= ano_range[1]) & (df["state"].isin(estados_sel))]

#KPIs
total_casos = df_f["number"].sum()
ano_pico = df_f.groupby("year")["number"].sum().idxmax()
estado_pico = df_f.groupby("state")["number"].sum().idxmax()
variacao = df.groupby("year")["number"].sum().pct_change().dropna()
variacao_media = variacao.mean() * 100

st.title("Queimadas na Amazônia (1998–2017)")
st.markdown("Dashboard interativa para análise de incêndios florestais no Brasil.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de Casos (filtro)", f"{total_casos:,.0f}")
c2.metric("Ano com Mais Casos", str(ano_pico))
c3.metric("Estado Mais Afetado", estado_pico)
c4.metric("Variação Média Anual", f"{variacao_media:+.1f}%")

st.divider()

#Gráficos
tab1, tab2, tab3, tab4 = st.tabs(["Por Ano", "Por Estado", "Variação %", "Previsão"])

with tab1:
    st.markdown("**Análise:** Identificação dos anos com maior e menor número de queimadas registradas.")
    ano_df = df_f.groupby("year")["number"].sum().reset_index()
    cores = ["crimson" if y == ano_df["number"].max() else
             "green"   if y == ano_df["number"].min() else
             "steelblue" for y in ano_df["number"]]
    fig = px.bar(ano_df, x="year", y="number",
                 title="Total de casos por ano",
                 labels={"year": "Ano", "number": "Nº de casos"})
    fig.update_traces(marker_color=cores)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("🔴 **Vermelho** = ano com mais casos &nbsp;|&nbsp; 🟢 **Verde** = ano com menos casos")

with tab2:
    st.markdown("**Análise:** Estados com maior e menor acumulado de queimadas no período selecionado.")
    est_df = df_f.groupby("state")["number"].sum().reset_index().sort_values("number", ascending=False)
    cores2 = ["crimson" if v == est_df["number"].max() else
              "green"   if v == est_df["number"].min() else
              "steelblue" for v in est_df["number"]]
    fig2 = px.bar(est_df, x="state", y="number",
                  title="Total de casos por estado",
                  labels={"state": "Estado", "number": "Nº de casos"})
    fig2.update_traces(marker_color=cores2)
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("🔴 **Vermelho** = estado mais afetado &nbsp;|&nbsp; 🟢 **Verde** = estado menos afetado")

with tab3:
    st.markdown("**Análise:** Variação percentual ano a ano — vermelho indica aumento, verde indica redução.")
    anual = df_f.groupby("year")["number"].sum()
    var_df = anual.pct_change().dropna().reset_index()
    var_df.columns = ["year", "variacao"]
    var_df["variacao"] *= 100
    cores3 = ["crimson" if v > 0 else "green" for v in var_df["variacao"]]
    fig3 = px.bar(var_df, x="year", y="variacao",
                  title="Variação % de queimadas (ano a ano)",
                  labels={"year": "Ano", "variacao": "Variação (%)"})
    fig3.update_traces(marker_color=cores3)
    fig3.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
    st.plotly_chart(fig3, use_container_width=True)
    maior_alta = var_df.loc[var_df["variacao"].idxmax()]
    maior_queda = var_df.loc[var_df["variacao"].idxmin()]
    st.markdown(f"Maior alta: **{int(maior_alta['year'])}** ({maior_alta['variacao']:+.1f}%) &nbsp;|&nbsp; 📉 Maior queda: **{int(maior_queda['year'])}** ({maior_queda['variacao']:+.1f}%)")

with tab4:
    st.markdown("**Análise preditiva:** Três modelos de Machine Learning treinados para prever o número de queimadas por estado/mês.")

    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb
    import tensorflow as tf
    from tensorflow import keras
    import plotly.graph_objects as go
    import numpy as np

    #Preparação dos dados
    df_ml = df.copy()
    le = LabelEncoder()
    df_ml["state_enc"] = le.fit_transform(df_ml["state"])
    le_month = LabelEncoder()
    df_ml["month_enc"] = le_month.fit_transform(df_ml["month"])

    features = ["year", "state_enc", "month_enc"]
    target   = "number"

    X = df_ml[features]
    y = df_ml[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.expander("Sobre a modelagem (clique para expandir)"):
        st.markdown("""
        **Problema:** Regressão — prever o número de queimadas por estado/mês/ano.

        **Variável alvo (target):** `number` — quantidade de focos de queimada registrados.

        **Variáveis preditoras (features):**
        - `year` — ano do registro
        - `state_enc` — estado (codificado com LabelEncoder)
        - `month_enc` — mês (codificado com LabelEncoder)

        **Pré-processamento:**
        - Codificação de variáveis categóricas (estado e mês) com LabelEncoder
        - Split 80/20 treino/teste com `random_state=42`

        **Modelos treinados:**
        1. Regressão Linear (scikit-learn)
        2. XGBoost Regressor (árvore melhorada)
        3. Rede Neural (Keras/TensorFlow)
        """)

    #Treina os 3 modelos
    with st.spinner("Treinando modelos... aguarde"):

        #1 Regressão Linear
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        pred_lr = lr.predict(X_test)

        #2 XGBoost
        xgb_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                       max_depth=5, random_state=42, verbosity=0)
        xgb_model.fit(X_train, y_train)
        pred_xgb = xgb_model.predict(X_test)

        #3 Rede Neural (Keras)
        X_train_np = X_train.values.astype("float32")
        X_test_np  = X_test.values.astype("float32")
        y_train_np = y_train.values.astype("float32")

        nn = keras.Sequential([
            keras.layers.Dense(64, activation="relu", input_shape=(3,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1)
        ])
        nn.compile(optimizer="adam", loss="mse")
        nn.fit(X_train_np, y_train_np, epochs=50, batch_size=32, verbose=0)
        pred_nn = nn.predict(X_test_np, verbose=0).flatten()

    #Métricas
    resultados = {
        "Modelo": ["Regressão Linear", "XGBoost", "Rede Neural"],
        "MAE":    [mean_absolute_error(y_test, pred_lr),
                   mean_absolute_error(y_test, pred_xgb),
                   mean_absolute_error(y_test, pred_nn)],
        "R²":     [r2_score(y_test, pred_lr),
                   r2_score(y_test, pred_xgb),
                   r2_score(y_test, pred_nn)],
    }
    df_res = pd.DataFrame(resultados)
    df_res["MAE"] = df_res["MAE"].map("{:.2f}".format)
    df_res["R²"]  = df_res["R²"].map("{:.3f}".format)

    st.subheader("Comparação dos modelos")
    st.table(df_res)  #st.table renderiza melhor no PDF

    melhor = resultados["Modelo"][resultados["R²"].index(max(resultados["R²"]))]
    st.success(f"✅ Melhor modelo: **{melhor}** (maior R²)")

    #Gráfico: Real vs Previsto
    pred_xgb_clip = np.clip(pred_xgb, 0, None)
    pred_nn_clip  = np.clip(pred_nn,  0, None)

    st.subheader("📈 Real vs Previsto — XGBoost")
    idx = np.argsort(y_test.values)[:100]
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Scatter(y=y_test.values[idx], mode="lines",
                                  name="Real", line=dict(color="steelblue")))
    fig_cmp.add_trace(go.Scatter(y=pred_xgb_clip[idx], mode="lines",
                                  name="XGBoost previsto",
                                  line=dict(color="crimson", dash="dash")))
    fig_cmp.update_layout(xaxis_title="Amostra (ordenada)", yaxis_title="Nº de casos",
                           yaxis=dict(rangemode="nonnegative"))
    st.plotly_chart(fig_cmp, use_container_width=True)

    #Importância das features (XGBoost)
    st.subheader("🔍 Importância das variáveis — XGBoost")
    imp = pd.DataFrame({
        "Feature": features,
        "Importância": xgb_model.feature_importances_
    }).sort_values("Importância", ascending=True)
    fig_imp = px.bar(imp, x="Importância", y="Feature", orientation="h",
                     color="Importância", color_continuous_scale="Reds")
    st.plotly_chart(fig_imp, use_container_width=True)

    #Interpretação
    st.subheader("💡 Interpretação dos resultados")
    st.markdown("""
    Os modelos preditivos complementam os indicadores da Parte 1 adicionando uma **visão prospectiva**:

    - A **Regressão Linear** captura a tendência geral de crescimento dos focos, mas não consegue
      modelar oscilações não-lineares entre anos e estados.
    - O **XGBoost** obtém melhor desempenho ao capturar interações entre ano, estado e mês —
      mostrando que sazonalidade e localização são determinantes para prever queimadas.
    - A **Rede Neural** aprende padrões complexos, mas com um dataset pequeno tende a não superar
      o XGBoost sem tunagem adicional.

    A variável mais importante segundo o XGBoost é o **estado**, alinhado com o que os KPIs
    da Parte 1 mostraram: Mato Grosso concentra proporcionalmente muito mais focos que os demais.
    """)

st.divider()
st.caption("Checkpoint 3 — Dashboard de Queimadas na Amazônia | Dados: Kaggle (1998–2017)")
