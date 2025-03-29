import streamlit as st
import pandas as pd
import fundamentus  # Para obter os dados da B3
import google.generativeai as genai  # Para usar a API do Gemini
from urllib.error import HTTPError
import threading  # Para executar a análise em segundo plano
import time  # Para simular o tempo de análise
import yfinance as yf # Para obter dados históricos do ativo
import ta # Para indicadores técnicos
import plotly.graph_objects as go # Para criar os gráficos
from plotly.subplots import make_subplots # Para criar subplots
import subprocess # Importa o módulo subprocess para instalar o ta
import os # Importa o módulo os para acessar variáveis de ambiente


api_key = "GEMINI_API_KEY"

genai.configure(api_key=api_key)  # Substitua pela sua chave de API do Gemini
model = genai.GenerativeModel('gemini-2.0-flash')

# Defina a lista de ações da B3
acoes = [
    "ITUB4", "BBDC4", "BBAS3", "B3SA3", "SANB11", "CIEL3", "IRBR3",
    "BBSE3", "BPAC11", "BIDI11", "PETR4", "PETR3", "PRIO3", "UGPA3",
    "VBBR3", "RRRP3", "VALE3", "CSNA3", "USIM5", "GGBR4", "GOAU4",
    "BRAP4", "CBAV3", "CMIG4", "ELET3", "ELET6", "ENGI11", "EQTL3",
    "CPFE3", "EGIE3", "ENEV3", "AURE3", "TAEE11", "NEOE3", "ABEV3",
    "MGLU3", "LREN3", "AMER3", "NTCO3", "PCAR3", "CRFB3", "ASAI3",
    "VVAR3", "SOMA3", "AMAR3", "LJQQ3", "VIVA3", "CYRE3", "EZTC3",
    "MRVE3", "ALSO3", "BRML3", "IGTI11", "MULT3", "TEND3", "DIRR3",
    "EVEN3", "RENT3", "RAIL3", "CCRO3", "STBP3", "PSSA3", "AZUL4",
    "GOLL4", "ECOR3", "MOVIDA3", "LOGN3", "RDOR3", "HAPV3", "FLRY3",
    "HYPE3", "GNDI3", "RADL3", "PAGS3", "QUAL3", "ONCO3", "LWSA3",
    "TOTS3", "CASH3", "MODL3", "MELI34", "POSI3", "NINJ3", "SUZB3",
    "KLBN11", "JBSS3", "BEEF3", "MRFG3", "SLCE3", "AGRO3", "SMTO3",
    "MDIA3", "WEGE3", "EMBR3", "BRKM5", "SBSP3", "CSAN3", "COGN3",
    "YDUQ3", "SEQL3", "TKNO4", "MYPK3", "AZEV4"
]

# Função para obter os dados da ação usando fundamentus
def obter_dados_acao(acao):
    try:
        pipeline = fundamentus.Pipeline(acao)
        response = pipeline.get_all_information()
        # Verifica se a resposta contém os dados esperados
        if not response or not hasattr(response, 'transformed_information'):
            print(f"Erro ao obter dados para {acao}: Resposta da API incompleta.")
            return None
        return response.transformed_information
    except HTTPError as e:
        if e.code == 404:
            print(f"Erro ao obter dados para {acao}: Ação não encontrada (404).")
            return None
        else:
            print(f"Erro ao obter dados para {acao}: Erro HTTP {e.code}")
            return None
    except Exception as e:
        print(f"Erro ao obter dados para {acao}: {e}")
        return None

# Função para preparar os dados para o modelo de IA
def preparar_dados_para_ia(dados, acao):
    if dados is None:
        return None

    # Verifica se as chaves existem antes de acessá-las
    price_info = dados.get('price_information', {})
    detailed_info = dados.get('detailed_information', {})
    valuation_indicators = dados.get('valuation_indicators', {})
    profitability_indicators = dados.get('profitability_indicators', {})
    oscillations_data = dados.get('oscillations', {})
    balance_sheet_data = dados.get('balance_sheet', {})
    income_statement_data = dados.get('income_statement_data', {})

    # Extrai os dados, com tratamento de KeyError
    try:
        data = {
            "acao": acao,
            "cotacao": price_info.get('price', None).value if price_info.get('price') else None,
            "data_cotacao": price_info.get('date', None).value if price_info.get('date') else None,
            "tipo_acao": detailed_info.get('stock_type', None).value if detailed_info.get('stock_type') else None,
            "volume_negociado": detailed_info.get('traded_volume_per_day', None).value if detailed_info.get('traded_volume_per_day') else None,
            "vpa": detailed_info.get('equity_value_per_share', None).value if detailed_info.get('equity_value_per_share') else None,
            "lpa": detailed_info.get('earnings_per_share', None).value if detailed_info.get('earnings_per_share') else None,
            "min_52_sem": detailed_info.get('variation_52_weeks', {}).get('lowest_value', None).value if detailed_info.get('variation_52_weeks', {}).get('lowest_value') else None,
            "max_52_sem": detailed_info.get('variation_52_weeks', {}).get('highest_value', None).value if detailed_info.get('variation_52_weeks', {}).get('highest_value') else None,
            "variacao_dia": oscillations_data.get('variation_day', None).value if oscillations_data.get('variation_day') else None,
            "variacao_mes": oscillations_data.get('variation_month', None).value if oscillations_data.get('variation_month') else None,
            "variacao_30_dias": oscillations_data.get('variation_30_days', None).value if oscillations_data.get('variation_30_days') else None,
            "variacao_12_meses": oscillations_data.get('variation_12_months', None).value if oscillations_data.get('variation_12_months') else None,
            "variacao_2022": oscillations_data.get('variation_2022', None).value if oscillations_data.get('variation_2022') else None,
            "variacao_2021": oscillations_data.get('variation_2021', None).value if oscillations_data.get('variation_2021') else None,
            "variacao_2020": oscillations_data.get('variation_2020', None).value if oscillations_data.get('variation_2020') else None,
            "variacao_2019": oscillations_data.get('variation_2019', None).value if oscillations_data.get('variation_2019') else None,
            "variacao_2018": oscillations_data.get('variation_2018', None).value if oscillations_data.get('variation_2018') else None,
            "variacao_2017": oscillations_data.get('variation_2017', None).value if oscillations_data.get('variation_2017') else None,
            "pl": valuation_indicators.get('price_divided_by_profit_title', None).value if valuation_indicators.get('price_divided_by_profit_title') else None,
            "pvp": valuation_indicators.get('price_divided_by_asset_value', None).value if valuation_indicators.get('price_divided_by_asset_value') else None,
            "pebit": valuation_indicators.get('price_divided_by_ebit', None).value if valuation_indicators.get('price_divided_by_ebit') else None,
            "psr": valuation_indicators.get('price_divided_by_net_revenue', None).value if valuation_indicators.get('price_divided_by_net_revenue') else None,
            "preco_ativos": valuation_indicators.get('price_divided_by_total_assets', None).value if valuation_indicators.get('price_divided_by_total_assets') else None,
            "preco_ativ_circ_liq": valuation_indicators.get('price_divided_by_net_current_assets', None).value if valuation_indicators.get('price_divided_by_net_current_assets') else None,
            "dividend_yield": valuation_indicators.get('dividend_yield', None).value if valuation_indicators.get('dividend_yield') else None,
            "ev_ebitda": valuation_indicators.get('enterprise_value_by_ebitda', None).value if valuation_indicators.get('enterprise_value_by_ebitda') else None,
            "ev_ebit": valuation_indicators.get('enterprise_value_by_ebit', None), # Corrigido esta linha
            "preco_capital_giro": valuation_indicators.get('price_by_working_capital', None).value if valuation_indicators.get('price_by_working_capital') else None,
            "roe": profitability_indicators.get('return_on_equity', None).value if profitability_indicators.get('return_on_equity') else None,
            "roic": profitability_indicators.get('return_on_invested_capital', None).value if profitability_indicators.get('return_on_invested_capital') else None,
            "ebit_ativo": profitability_indicators.get('ebit_divided_by_total_assets', None).value if profitability_indicators.get('ebit_divided_by_total_assets') else None,
            "crescimento_receita_5_anos": profitability_indicators.get('net_revenue_growth_last_5_years', None).value if profitability_indicators.get('net_revenue_growth_last_5_years') else None,
            "giro_ativos": profitability_indicators.get('net_revenue_divided_by_total_assets', None).value if profitability_indicators.get('net_revenue_divided_by_total_assets') else None,
            "margem_bruta": profitability_indicators.get('gross_profit_divided_by_net_revenue', None).value if profitability_indicators.get('gross_profit_divided_by_net_revenue') else None,
            "margem_ebit": profitability_indicators.get('ebit_divided_by_net_revenue', None).value if profitability_indicators.get('ebit_divided_by_net_revenue') else None,
            "margem_liquida": profitability_indicators.get('net_income_divided_by_net_revenue', None).value if profitability_indicators.get('net_income_divided_by_net_revenue') else None,
            "liquidez_corrente": dados.get('indebtedness_indicators', {}).get('current_liquidity', None).value if dados.get('indebtedness_indicators', {}).get('current_liquidity') else None,
            "divida_bruta_patrimonio": dados.get('indebtedness_indicators', {}).get('gross_debt_by_equity', None).value if dados.get('indebtedness_indicators', {}).get('gross_debt_by_equity') else None,
            "divida_liquida_patrimonio": dados.get('indebtedness_indicators', {}).get('net_debt_by_equity', None).value if dados.get('indebtedness_indicators', {}).get('net_debt_by_equity') else None,
            "divida_liquida_ebitda": dados.get('indebtedness_indicators', {}).get('net_debt_by_ebitda', None).value if dados.get('indebtedness_indicators', {}).get('net_debt_by_ebitda') else None,
            "patrimonio_ativos": dados.get('indebtedness_indicators', {}).get('equity_by_total_assets', None).value if dados.get('indebtedness_indicators', {}).get('equity_by_total_assets') else None,
            "total_ativos": balance_sheet_data.get('total_assets', None).value if balance_sheet_data.get('total_assets') else None,
            "ativo_circulante": balance_sheet_data.get('current_assets', None).value if balance_sheet_data.get('current_assets') else None,
            "disponibilidades": balance_sheet_data.get('cash', None).value if balance_sheet_data.get('cash') else None,
            "divida_bruta": balance_sheet_data.get('gross_debt', None).value if balance_sheet_data.get('gross_debt') else None,
            "divida_liquida": balance_sheet_data.get('net_debt', None).value if balance_sheet_data.get('net_debt') else None,
            "patrimonio_liquido": balance_sheet_data.get('equity', None).value if balance_sheet_data.get('equity') else None,
            "receita_liquida_3meses": income_statement_data.get('three_months', {}).get('revenue', None).value if income_statement_data.get('three_months', {}).get('revenue') else None,
            "ebit_3meses": income_statement_data.get('three_months', {}).get('ebit', None).value if income_statement_data.get('three_months', {}).get('ebit') else None,
            "lucro_liquido_3meses": income_statement_data.get('three_months', {}).get('net_income', None).value if income_statement_data.get('three_months', {}).get('net_income') else None,
            "receita_liquida_12meses": income_statement_data.get('twelve_months', {}).get('revenue', None).value if income_statement_data.get('twelve_months', {}).get('revenue') else None,
            "ebit_12meses": income_statement_data.get('twelve_months', {}).get('ebit', None),
            "lucro_liquido_12meses": income_statement_data.get('twelve_months', {}).get('net_income', None).value if income_statement_data.get('twelve_months', {}).get('net_income') else None,
        }
        return data
    except KeyError as e:
        print(f"Erro ao extrair dados para IA da ação {acao}: Chave não encontrada: {e}")
        return None
    except Exception as e:
        print(f"Erro ao preparar dados para IA da ação {acao}: {e}")
        return None

# Função para enviar a análise para o Gemini
def enviar_analise_para_ia(data):
    prompt_text = f"""
        Analise os dados da ação {data['acao']} e forneça uma análise qualitativa completa,
        baseada no histórico, indicando se o ativo representa uma oportunidade de compra
        por preço baixo e venda em no mínimo 1 dia útil e no máximo 5 dias úteis.
        Forneça indicativos para o usuário com previsão de análise do que pode ocorrer
        com base em todos os dados extraídos e fornecidos para cada ação.
        Nunca sugira operações de day trade. Dê peso maior para ROIC e P/L.
        Classifique a recomendação como "Positivo", "Negativo" ou "Neutro". "Com base nos siguientes dados da ação, analise se há uma oportunidade de investimento de curto prazo. Considere a tendência recente, liquidez, múltiplos de valuation, endividamento e rentabilidade. Identifique sinais de valorização ou risco elevado e forneça uma conclusão objetiva sobre o potencial de curto prazo.
        Com base nesses indicadores, avalie:

        Se há tendência de valorização ou queda no curto prazo.

        Se o volume negociado sugere alta liquidez ou risco de baixa demanda.

        Se os múltiplos indicam uma ação subvalorizada ou sobrevalorizada.

        Se o histórico recente e os fundamentos financeiros justificam um investimento de curto prazo.

        Forneça uma recomendação clara, destacando os fatores positivos e os riscos envolvidos."
        
        Dados da Ação:
        Cotação: {data['cotacao']}
        Data da Cotação: {data['data_cotacao']}
        Tipo de Ação: {data['tipo_acao']}
        Volume Negociado por Dia: {data['volume_negociado']}
        VPA: {data['vpa']}
        LPA: {data['lpa']}
        Mínimo 52 Semanas: {data['min_52_sem']}
        Máximo 52 Semanas: {data['max_52_sem']}
        Variação Dia: {data['variacao_dia']}
        Variação Mês: {data['variacao_mes']}
        Variação 30 Dias: {data['variacao_30_dias']}
        Variação 12 Meses: {data['variacao_12_meses']}
        Variação 2022: {data['variacao_2022']}
        Variação 2021: {data['variacao_2021']}
        Variação 2020: {data['variacao_2020']}
        Variação 2019: {data['variacao_2019']}
        Variação 2018: {data['variacao_2018']}
        Variação 2017: {data['variacao_2017']}
        P/L: {data['pl']}
        P/VP: {data['pvp']}
        P/EBIT: {data['pebit']}
        PSR: {data['psr']}
        Preço/Ativos: {data['preco_ativos']}
        Preço/Ativ Circ Liq: {data['preco_ativ_circ_liq']}
        Dividend Yield: {data['dividend_yield']}
        EV/EBITDA: {data['ev_ebitda']}
        EV/EBIT: {data['ev_ebit']}
        Preço/Capital de Giro: {data['preco_capital_giro']}
        ROE: {data['roe']}
        ROIC: {data['roic']}
        EBIT/Ativo: {data['ebit_ativo']}
        Crescimento Receita 5 Anos: {data['crescimento_receita_5_anos']}
        Giro Ativos: {data['giro_ativos']}
        Margem Bruta: {data['margem_bruta']}
        Margem EBIT: {data['margem_ebit']}
        Margem Líquida: {data['margem_liquida']}
        Liquidez Corrente: {data['liquidez_corrente']}
        Dívida Bruta/Patrimônio: {data['divida_bruta_patrimonio']}
        Dívida Líquida/Patrimônio: {data['divida_liquida_patrimonio']}
        Dívida Líquida/EBITDA: {data['divida_liquida_ebitda']}
        Patrimônio/Ativos: {data['patrimonio_ativos']}
        Total de Ativos: {data['total_ativos']}
        Ativo Circulante: {data['ativo_circulante']}
        Disponibilidades: {data['disponibilidades']}
        Dívida Bruta: {data['divida_bruta']}
        Dívida Líquida: {data['divida_liquida']}
        Patrimônio Líquido: {data['patrimonio_liquido']}
        Receita Líquida 3 Meses: {data['receita_liquida_3meses']}
        EBIT 3 Meses: {data['ebit_3meses']}
        Lucro Líquido 3 Meses: {data['lucro_liquido_3meses']}
        Receita Líquida 12 Meses: {data['receita_liquida_12meses']}
        EBIT 12 Meses: {data['ebit_12meses']}
        Lucro Líquido 12 Meses: {data['lucro_liquido_12meses']}
    """

    if model is None:
        return None, "Erro: Modelo de IA não inicializado."  # Retorna None e mensagem de erro

    try:
        response = model.generate_content(prompt_text)
        analise = response.text
        # Classifica a recomendação (simplificado para demonstração)
        if "compra" in analise.lower() or "positivo" in analise.lower():
            classificacao = "Positivo"
        elif "venda" in analise.lower() or "negativo" in analise.lower():
            classificacao = "Negativo"
        else:
            classificacao = "Neutro"
        return analise, classificacao
    except Exception as e:
        print(f"Erro ao enviar prompt para o Gemini: {e}")
        return None, f"Erro ao obter análise da IA: {e}"

def analisar_acao(acao, resultados, acao_status):
    dados_acao = obter_dados_acao(acao)
    if dados_acao is None:
        resultados[acao] = {"erro": f"Não foi possível obter dados para a ação {acao}."}
        acao_status[acao] = "Erro"
        return

    dados_para_ia = preparar_dados_para_ia(dados_acao, acao)
    if dados_para_ia is None:
        resultados[acao] = {"erro": f"Não foi possível preparar os dados para a ação {acao} para análise da IA."}
        acao_status[acao] = "Erro"
        return

    analise_ia, classificacao = enviar_analise_para_ia(dados_para_ia) # Recebe a análise e a classificação
    if analise_ia is None:
        resultados[acao] = {"erro": f"Erro ao obter análise da IA para a ação {acao}."}
        acao_status[acao] = "Erro"
        return

    # Converter o dicionário em um DataFrame para exibição
    df_dados = pd.DataFrame([dados_para_ia])

    # Obter dados para o gráfico
    try:
        ticket = yf.Ticker(f"{acao}.SA")
        hist = ticket.history(period="1y") # Pega o histórico de 1 ano
        price_series = hist['Close'].to_dict()
        volume_series = hist['Volume'].to_dict()

        # Tenta instalar a biblioteca ta e a importa
        try:
            subprocess.run(['pip', 'install', 'ta'], check=True)
            import ta # Importa ta
        except Exception as e:
            print(f"Erro ao instalar a biblioteca 'ta': {e}")
            resultados[acao] = {"erro": f"Erro ao instalar a biblioteca 'ta': {e}.  Verifique se o pip está instalado e o ambiente virtual está configurado corretamente. A análise da ação {acao} será feita sem os indicadores técnicos."}
            acao_status[acao] = "Erro"
            resultados[acao]["analise"] = f"Erro ao instalar a biblioteca 'ta'. A análise da ação {acao} será feita sem os indicadores técnicos. Erro: {e}"
            resultados[acao]["dados"] = df_dados
            return

        # Calcular Médias Móveis
        try:
            hist['MA9'] = ta.trend.sma(hist['Close'], window=9)
            hist['MA20'] = ta.trend.sma(hist['Close'], window=20)
            hist['MA50'] = ta.trend.sma(hist['Close'], window=50)

            ma_values = {
                'MA9': hist['MA9'].to_dict(),
                'MA20': hist['MA20'].to_dict(),
                'MA50': hist['MA50'].to_dict(),
            }
        except Exception as e:
            print(f"Erro ao calcular médias móveis para a ação {acao}: {e}")
            ma_values = {}

        # Calcular RSI
        try:
            hist['RSI'] = ta.momentum.rsi(hist['Close'], window=14)
            rsi_value = hist['RSI'].iloc[-1]  # Obtém o último valor de RSI
        except Exception as e:
            print(f"Erro ao calcular RSI para a ação {acao}: {e}")
            rsi_value = None

        chart_data = {
            'price_series': price_series,
            'volume_series': volume_series,
            'ma_values': ma_values,
            'rsi': rsi_value
        }
    except Exception as e:
        print(f"Erro ao obter dados para gráfico de {acao}: {e}")
        chart_data = None

    # Converter o dicionário em um DataFrame para exibição
    df_dados = pd.DataFrame([dados_para_ia])
    resultados[acao] = {"analise": analise_ia, "classificacao": classificacao, "dados": df_dados, "chart_data": chart_data} # Armazena a classificação e os dados e os dados do gráfico
    acao_status[acao] = "Concluído" #Atualiza o status da ação

def iniciar_analise(acoes_selecionadas, resultados, acao_status, analise_concluida):
    for acao in acoes_selecionadas:
        acao_status[acao] = "Analisando" # Define o status da ação como "Analisando"
        thread = threading.Thread(target=analisar_acao, args=(acao, resultados, acao_status))
        thread.start()
    # Aguarda a conclusão de todas as threads
    while any(status != "Concluído" and status != "Erro" for status in acao_status.values()):
        time.sleep(1)  # Aguarda 1 segundo

    analise_concluida.set() # Sinaliza que a análise foi concluída

# Função para plotar o gráfico do ativo
def plot_asset_chart(ticker, data):
    """Cria um gráfico para um ativo específico com indicadores técnicos."""
    try:
        # Verificar se temos dados suficientes
        if data is None or 'price_series' not in data or not data['price_series']:
            return None

        # Converter dados de série para listas ordenadas
        dates = sorted(data['price_series'].keys())
        prices = [data['price_series'][date] for date in dates]

        # Criar figura com dois subplots: preço e volume
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=(f"{ticker} - Preço", "Volume"),
                            row_heights=[0.7, 0.3])

        # Adicionar série de preços
        fig.add_trace(
            go.Scatter(x=dates, y=prices, mode='lines', name='Preço',
                       line=dict(color='#2962FF', width=2)),
            row=1, col=1
        )

        # Adicionar Médias Móveis
        ma_values = data.get('ma_values', {})
        for ma_name, ma_value in ma_values.items():
            if ma_name == 'MA9':
                color = '#FF6D00'  # Laranja
            elif ma_name == 'MA20':
                color = '#00C853'  # Verde
            elif ma_name == 'MA50':
                color = '#D500F9'  # Roxo
            else:
                color = '#757575'  # Cinza

            # Converter ma_value para lista
            ma_list = list(ma_value.values())
            if len(prices) >= int(ma_name[2:]):
                fig.add_trace(
                    go.Scatter(x=dates, y=ma_list, mode='lines', name=ma_name,
                                 line=dict(color=color, width=1.5, dash='dot')),
                    row=1, col=1
                )

        # Adicionar volume, se disponível
        if 'volume_series' in data and data['volume_series']:
            volumes = [data['volume_series'][date] for date in dates]
            fig.add_trace(
                go.Bar(x=dates, y=volumes, name='Volume', marker_color='#90A4AE'),
                row=2, col=1
            )

            # Média de volume de 20 dias
            if len(volumes) >= 20:
                vol_ma = []
                for i in range(len(volumes)):
                    if i < 19:
                        vol_ma.append(None)
                    else:
                        vol_ma.append(sum(volumes[i-19:i+1])/20)

                fig.add_trace(
                    go.Scatter(x=dates, y=vol_ma, mode='lines', name='Volume MA20',
                                 line=dict(color='#FF6D00', width=1.5)),
                    row=2, col=1
                )

        # Adicionar marcador para RSI, se disponível
        if 'rsi' in data and data['rsi'] is not None:
            # Mostrar o último valor do RSI como anotação
            fig.add_annotation(
                x=dates[-1], y=prices[-1] * 1.05,
                text=f"RSI: {data['rsi']:.1f}",
                showarrow=True,
                arrowhead=1,
                bgcolor="#FFECB3" if data['rsi'] < 30 else "#E1F5FE" # Muda a cor de fundo dependendo se está sobrecomprado ou sobrevendido
            )

        # Ajustar layout
        fig.update_layout(
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=40, b=40),
            hovermode="x unified",
            template="plotly_white"
        )

        return fig
    except Exception as e:
        print(f"Erro ao plotar o gráfico: {e}")
        return None

# Streamlit App
def main():
    st.title("Análise de Ações da B3 com IA")
    st.write("Clique no botão para analisar as ações selecionadas.")

    acoes_selecionadas = st.multiselect("Selecione as Ações", acoes, default=["ITUB4", "PETR4"])
    resultados = {}
    acao_status = {acao: "Pendente" for acao in acoes_selecionadas}  # Rastreia o status de cada ação
    analise_concluida = threading.Event()
    if st.button("Analisar Ações Selecionadas"):
        st.session_state.analise_iniciada = True # Define a variável de estado
        st.session_state.resultados = {} # Armazena os resultados no estado da sessão
        st.session_state.acao_status = {} # Armazena o status da ação
        st.session_state.analise_concluida = threading.Event()

        with st.spinner("Analisando ações..."):
            thread_analise = threading.Thread(target=iniciar_analise, args=(acoes_selecionadas, st.session_state.resultados, st.session_state.acao_status, st.session_state.analise_concluida))
            thread_analise.start()
            # Exibe o status de cada ação
            while not st.session_state.analise_concluida.is_set():
                status_texto = "Status da Análise:\n"
                for acao, status in st.session_state.acao_status.items():
                    status_texto += f"- {acao}: {status}\n"
                #st.text(status_texto)
                time.sleep(1) # Atualiza o status a cada segundo
            
            status_texto = "Status da Análise:\n" #Exibe o status final
            for acao, status in st.session_state.acao_status.items():
                status_texto += f"- {acao}: {status}\n"
            st.text(status_texto)
        
    if st.session_state.get("analise_iniciada"): # Verifica se a análise foi iniciada
        if not st.session_state.analise_concluida.is_set():
            st.warning("A análise ainda estáem andamento. Aguarde a conclusão.")
        else:
            resultados = st.session_state.resultados
            st.header("Resultados da Análise")
            resultados_positivos = []
            resultados_negativos = []
            resultados_neutros = []
            for acao, resultado in resultados.items():
                if "erro" in resultado:
                    st.error(f"Erro na análise da ação {acao}: {resultado['erro']}")
                else:
                    st.subheader(f"Análise para {acao}")
                    #st.write(f"Classificação: {resultado['classificacao']}") # Exibe a classificação
                    st.write(resultado['analise'])
                    st.dataframe(resultado['dados']) # Exibe os dados da ação

                    # Exibir gráfico
                    chart = plot_asset_chart(acao, resultado['chart_data'])
                    if chart:
                        st.plotly_chart(chart)
                    else:
                        st.error(f"Não foi possível gerar o gráfico para a ação {acao}.")

                    if resultado['classificacao'] == "Positivo":
                        resultados_positivos.append(acao)
                    elif resultado['classificacao'] == "Negativo":
                        resultados_negativos.append(acao)
                    else:
                        resultados_neutros.append(acao)

            if resultados_positivos:
                st.success(f"Ações com recomendação positiva: {', '.join(resultados_positivos)}")
            if resultados_negativos:
                st.error(f"Ações com recomendação negativa: {', '.join(resultados_negativos)}")
            if resultados_neutros:
                st.info(f"Ações com recomendação neutra: {', '.join(resultados_neutros)}")
            
if __name__ == "__main__":
    st.session_state.setdefault('analise_iniciada', False) # Inicializa a variável de estado
    main()
