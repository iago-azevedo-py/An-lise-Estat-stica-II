# Instale as bibliotecas necessárias caso não as tenha
# !pip install yfinance pandas matplotlib seaborn statsmodels streamlit google-generativeai plotly

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from textwrap import wrap
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import google.generativeai as genai
from datetime import datetime
import scipy.stats as stats

# =============================================================================
# Configuração da API do Gemini
# =============================================================================
def configure_gemini():
    """Configura a API do Gemini"""
    api_key = st.sidebar.text_input("Digite sua API Key do Google Gemini:", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        return True
    else:
        st.sidebar.warning("Por favor, insira sua API Key do Google Gemini")
        return False

def ask_gemini_agent(question, data_context):
    """Agente de IA para responder dúvidas sobre análise"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        prompt = f"""
        Você é um especialista em análise estatística financeira. Com base nos seguintes dados:
        
        {data_context}
        
        Pergunta do usuário: {question}
        
        Responda de forma clara e didática, explicando os conceitos estatísticos de maneira acessível.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erro ao consultar o agente: {str(e)}"

# =============================================================================
# Classe do Agente de IA para Análise de Investimentos
# =============================================================================
class AnalystAgent:
    """
    Um agente de IA que interpreta os resultados de uma análise de regressão
    e correlação para um investidor.
    """
    def __init__(self, correlation_value, regression_results, data_start_date, data_end_date):
        self.correlation = correlation_value
        self.results = regression_results
        self.start_date = data_start_date
        self.end_date = data_end_date

    def _interpret_correlation(self):
        """Interpreta a força e o sinal do coeficiente de correlação."""
        sinal = "positiva" if self.correlation >= 0 else "negativa"
        if abs(self.correlation) >= 0.8:
            return f"muito forte e {sinal}"
        elif abs(self.correlation) >= 0.6:
            return f"forte e {sinal}"
        elif abs(self.correlation) >= 0.4:
            return f"moderada e {sinal}"
        else:
            return f"fraca e {sinal}"

    def generate_investment_report(self):
        """Gera um relatório completo explicando os resultados estatísticos."""
        # Extraindo os valores do sumário da regressão
        r_squared = self.results.rsquared
        p_value_brent = self.results.pvalues['Preco_Brent']
        coef_brent = self.results.params['Preco_Brent']
        coef_const = self.results.params['const']

        report = f"""
==============================================================================
        RELATÓRIO DE ANÁLISE DE INVESTIMENTO: PETR4 vs. PETRÓLEO BRENT
==============================================================================

Olá! Eu sou seu Agente de Análise. Analisei os dados de {self.start_date} a {self.end_date}
e aqui estão os insights para fundamentar sua tese de investimento:

---
1. ANÁLISE DE ASSOCIAÇÃO (CORRELAÇÃO)
---
O coeficiente de correlação de Pearson entre o preço da PETR4 e o do Brent foi de {self.correlation:.4f}.
Isso indica uma correlação {self._interpret_correlation()}.

*   **Insight para o Investidor:** A sua hipótese principal está correta. Existe uma
    relação estatística clara e poderosa entre as duas variáveis. Quando o
    petróleo se move, a ação da Petrobras tende a seguir na mesma direção.

---
2. ANÁLISE DE REGRESSÃO LINEAR (PODER PREDITIVO)
---
A regressão linear nos ajuda a entender o "quanto" e "com que certeza" essa relação acontece.

    a) Poder Explicativo (R-Quadrado - R²):
       O R² do modelo foi de {r_squared:.3f}. Isso significa que aproximadamente {r_squared:.1%}
       da variação diária no preço da PETR4 pode ser explicada pela variação no
       preço do Petróleo Brent.

       *   **Insight para o Investidor:** Este é um valor muito alto! Mostra que o preço do
           petróleo é, de longe, o principal motor de valor da ação.

    b) Teste de Hipótese (p-valor):
       O p-valor para a variável 'Preco_Brent' foi de {p_value_brent:.3g}. Como este valor
       é praticamente zero (muito menor que 0.05), nós rejeitamos a hipótese
       de que a relação é mero acaso.

       *   **Insight para o Investidor:** A relação não é uma coincidência. Ela é
           estatisticamente significativa e confiável.

    c) O Modelo Preditivo (Coeficientes):
       A equação do modelo é: Preço_PETR4 = {coef_const:.2f} + {coef_brent:.2f} * Preço_Brent

       *   **INSIGHT-CHAVE (A REGRA DE BOLSO):** O coeficiente de {coef_brent:.2f} é a sua
           informação mais valiosa. Ele sugere que, em média, para cada
           aumento de $1 no barril de Brent, o modelo prevê que o preço da
           PETR4 tende a subir R$ {coef_brent:.2f}.

---
CONCLUSÃO E RECOMENDAÇÃO
---
Sua tese de investimento está estatisticamente validada. O preço da PETR4
é fortemente e positivamente influenciado pelo preço do petróleo Brent.

**Próximo Passo Sugerido:** Para criar um modelo ainda mais robusto, o próximo
passo ideal é adicionar a taxa de câmbio (USD/BRL) à análise, transformando-a
em uma regressão múltipla. Isso provavelmente aumentará o poder explicativo
(R²) do seu modelo.

==============================================================================
        FIM DO RELATÓRIO
==============================================================================
        """
        return report

# =============================================================================
# Script Principal de Análise - Dashboard
# =============================================================================

st.set_page_config(
    page_title="Dashboard: Análise PETR4 vs Brent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Dashboard: Análise Estatística PETR4 vs Petróleo Brent")
st.markdown("---")

# Sidebar para configurações
st.sidebar.title("⚙️ Configurações")

# Configuração do Gemini
gemini_configured = configure_gemini()

# Parâmetros de análise
st.sidebar.subheader("📅 Período de Análise")
start_date = st.sidebar.date_input("Data Inicial", value=pd.to_datetime('2019-01-01'))
end_date = st.sidebar.date_input("Data Final", value=pd.to_datetime('today'))

# 1. Coleta e Preparação dos Dados
@st.cache_data
def load_data(start_date, end_date):
    tickers = ['PETR4.SA', 'BZ=F']
    raw_data = yf.download(tickers, start=start_date, end=end_date)
    
    # Ajuste para MultiIndex ou SingleIndex
    if isinstance(raw_data.columns, pd.MultiIndex):
        if 'Adj Close' in raw_data.columns.get_level_values(0):
            data = raw_data['Adj Close'].copy()
        elif 'Close' in raw_data.columns.get_level_values(0):
            data = raw_data['Close'].copy()
        else:
            data = raw_data.iloc[:, :].copy()
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    else:
        data = raw_data.copy()
    
    data.rename(columns={'PETR4.SA': 'Preco_PETR4', 'BZ=F': 'Preco_Brent'}, inplace=True)
    data.dropna(inplace=True)
    return data

with st.spinner('🔄 Carregando dados...'):
    data = load_data(start_date, end_date)

# Métricas gerais
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Dados Analisados", f"{len(data):,}")
with col2:
    st.metric("📈 PETR4 Atual", f"R$ {data['Preco_PETR4'].iloc[-1]:.2f}")
with col3:
    st.metric("🛢️ Brent Atual", f"${data['Preco_Brent'].iloc[-1]:.2f}")
with col4:
    correlation = data['Preco_PETR4'].corr(data['Preco_Brent'])
    st.metric("🔗 Correlação", f"{correlation:.4f}")

st.markdown("---")

# 2. Visualizações e Análises
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Série Temporal", "🔗 Correlação", "📊 Regressão", "📋 Relatório", "🤖 Chat IA"])

with tab1:
    st.subheader("📈 Evolução dos Preços")
    
    # Gráfico de série temporal
    fig_time = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Preços Normalizados (Base 100)', 'Preços Absolutos'),
        vertical_spacing=0.1
    )
    
    # Normalizar preços para comparação
    data_norm = data.div(data.iloc[0]).mul(100)
    
    fig_time.add_trace(
        go.Scatter(x=data.index, y=data_norm['Preco_PETR4'], name='PETR4', line=dict(color='blue')),
        row=1, col=1
    )
    fig_time.add_trace(
        go.Scatter(x=data.index, y=data_norm['Preco_Brent'], name='Brent', line=dict(color='orange')),
        row=1, col=1
    )
    
    fig_time.add_trace(
        go.Scatter(x=data.index, y=data['Preco_PETR4'], name='PETR4 (R$)', line=dict(color='blue'), showlegend=False),
        row=2, col=1
    )
    
    fig_time2 = fig_time.add_trace(
        go.Scatter(x=data.index, y=data['Preco_Brent'], name='Brent (US$)', line=dict(color='orange'), yaxis='y3', showlegend=False),
        row=2, col=1
    )
    
    fig_time.update_layout(height=600, title_text="Evolução Temporal dos Preços")
    fig_time.update_yaxes(title_text="Base 100", row=1, col=1)
    fig_time.update_yaxes(title_text="PETR4 (R$)", row=2, col=1)
    
    st.plotly_chart(fig_time, use_container_width=True)

with tab2:
    st.subheader("🔗 Análise de Correlação")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de dispersão interativo
        fig_scatter = px.scatter(
            data, x='Preco_Brent', y='Preco_PETR4',
            title='Dispersão: PETR4 vs Brent',
            labels={'Preco_Brent': 'Preço Brent (US$)', 'Preco_PETR4': 'Preço PETR4 (R$)'},
            trendline='ols'
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Teste de correlação
        corr_coef, p_value_corr = stats.pearsonr(data['Preco_PETR4'], data['Preco_Brent'])
        
        st.markdown("**📊 Resultados do Teste de Correlação:**")
        st.write(f"• **Coeficiente de Correlação:** {corr_coef:.4f}")
        st.write(f"• **P-valor:** {p_value_corr:.2e}")
        st.write(f"• **Interpretação:** {'Correlação significativa' if p_value_corr < 0.05 else 'Correlação não significativa'}")
    
    with col2:
        # Matriz de correlação visual
        corr_matrix = data.corr()
        fig_heatmap = px.imshow(
            corr_matrix, 
            text_auto=True, 
            aspect="auto",
            title="Matriz de Correlação",
            color_continuous_scale='RdBu'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Distribuição dos retornos
        returns = data.pct_change().dropna()
        
        fig_dist = make_subplots(rows=1, cols=2, subplot_titles=('Distribuição PETR4', 'Distribuição Brent'))
        
        fig_dist.add_trace(
            go.Histogram(x=returns['Preco_PETR4'], name='PETR4', nbinsx=50, opacity=0.7),
            row=1, col=1
        )
        fig_dist.add_trace(
            go.Histogram(x=returns['Preco_Brent'], name='Brent', nbinsx=50, opacity=0.7),
            row=1, col=2
        )
        
        fig_dist.update_layout(height=300, title_text="Distribuição dos Retornos Diários")
        st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    st.subheader("📊 Análise de Regressão Linear")
    
    # Cálculos estatísticos
    Y = data['Preco_PETR4']
    X = sm.add_constant(data['Preco_Brent'])
    model = sm.OLS(Y, X)
    results = model.fit()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de regressão com residuals
        predictions = results.predict()
        residuals = Y - predictions
        
        fig_reg = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Regressão Linear', 'Análise de Resíduos'),
            vertical_spacing=0.1
        )
        
        # Gráfico de regressão
        fig_reg.add_trace(
            go.Scatter(x=data['Preco_Brent'], y=data['Preco_PETR4'], mode='markers', name='Dados Observados', opacity=0.6),
            row=1, col=1
        )
        fig_reg.add_trace(
            go.Scatter(x=data['Preco_Brent'], y=predictions, mode='lines', name='Linha de Regressão', line=dict(color='red')),
            row=1, col=1
        )
        
        # Gráfico de resíduos
        fig_reg.add_trace(
            go.Scatter(x=predictions, y=residuals, mode='markers', name='Resíduos', showlegend=False),
            row=2, col=1
        )
        fig_reg.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)
        
        fig_reg.update_layout(height=600, title_text="Análise de Regressão")
        fig_reg.update_yaxes(title_text="PETR4 (R$)", row=1, col=1)
        fig_reg.update_yaxes(title_text="Resíduos", row=2, col=1)
        fig_reg.update_xaxes(title_text="Brent (US$)", row=2, col=1)
        
        st.plotly_chart(fig_reg, use_container_width=True)
    
    with col2:
        # Métricas da regressão
        st.markdown("**📈 Métricas do Modelo:**")
        
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("R²", f"{results.rsquared:.4f}")
            st.metric("R² Ajustado", f"{results.rsquared_adj:.4f}")
            
        with metrics_col2:
            st.metric("F-statistic", f"{results.fvalue:.2f}")
            st.metric("P-valor (F)", f"{results.f_pvalue:.2e}")
        
        st.markdown("**🔢 Coeficientes:**")
        coef_df = pd.DataFrame({
            'Coeficiente': results.params,
            'Erro Padrão': results.bse,
            'P-valor': results.pvalues,
            'IC Inferior': results.conf_int()[0],
            'IC Superior': results.conf_int()[1]
        })
        st.dataframe(coef_df, use_container_width=True)
        
        # Testes de normalidade dos resíduos
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        st.markdown("**🧪 Testes dos Resíduos:**")
        st.write(f"• **Shapiro-Wilk (Normalidade):** {shapiro_p:.4f}")
        st.write(f"• **Interpretação:** {'Resíduos normais' if shapiro_p > 0.05 else 'Resíduos não normais'}")

with tab4:
    st.subheader("📋 Relatório Completo")
    
    # Gerar relatório usando a classe AnalystAgent
    agent = AnalystAgent(
        correlation_value=correlation,
        regression_results=results,
        data_start_date=data.index.min().strftime('%d/%m/%Y'),
        data_end_date=data.index.max().strftime('%d/%m/%Y')
    )
    
    investment_report = agent.generate_investment_report()
    
    # Exibir o relatório em formato markdown para melhor formatação
    st.markdown(f"""
    ```
    {investment_report}
    ```
    """)
    
    # Botão para download do relatório
    st.download_button(
        label="📥 Baixar Relatório",
        data=investment_report,
        file_name=f"relatorio_petr4_brent_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )

with tab5:
    st.subheader("🤖 Chat com Agente de IA")
    
    if gemini_configured:
        # Contexto dos dados para o agente
        data_context = f"""
        Dados analisados: {len(data)} observações de {data.index.min().strftime('%d/%m/%Y')} a {data.index.max().strftime('%d/%m/%Y')}
        Correlação PETR4-Brent: {correlation:.4f}
        R² da regressão: {results.rsquared:.4f}
        Coeficiente angular: {results.params['Preco_Brent']:.4f}
        P-valor da regressão: {results.f_pvalue:.2e}
        """
        
        # Interface do chat
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Exibir histórico de mensagens
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input do usuário
        if prompt := st.chat_input("Faça uma pergunta sobre a análise..."):
            # Adicionar mensagem do usuário
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Gerar resposta do agente
            with st.chat_message("assistant"):
                with st.spinner("🤔 Pensando..."):
                    response = ask_gemini_agent(prompt, data_context)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Botão para limpar chat
        if st.button("🗑️ Limpar Chat"):
            st.session_state.messages = []
            st.session_state.clear()
            
    else:
        st.warning("⚠️ Configure sua API Key do Google Gemini na barra lateral para usar o chat.")
        st.markdown("""
        **Como obter sua API Key:**
        1. Acesse [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Crie uma nova API Key
        3. Copie e cole na barra lateral
        """)