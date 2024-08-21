# Import das Libs
import json
import os
from datetime import datetime, timedelta
import yfinance as yf
from crewai import Agent, Task, Process, Crew
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults
import streamlit as st


# Função para obter o preço das ações usando a API do Yahoo Finance
def obter_preco_acoes(ticket):
    """
    Busca os dados históricos de preços das ações para o ticker especificado, desde 1 ano antes da data atual.

    Args:
        ticket (str): O símbolo do ticker da ação (por exemplo, 'AAPL' para Apple).

    Returns:
        pandas.DataFrame: Um DataFrame contendo os dados históricos de preços da ação.
    """
    # Data atual
    data_atual = datetime.now().strftime('%Y-%m-%d')

    # Data de 1 ano atrás
    data_inicial = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Faz o download dos dados históricos de preços das ações para o período especificado
    stock = yf.download(ticket, start=data_inicial, end=data_atual)
    return stock


# Criando a ferramenta Yahoo Finance Tool
yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",  # Nome da ferramenta
    description="Busca os preços das ações do {ticket} desde 1 ano atrás até a data atual, usando a API do Yahoo Finance.",
    # Descrição da ferramenta
    func=lambda ticket: obter_preco_acoes(ticket),  # Função que a ferramenta vai executar
)
# Carregando as variáveis de ambiente do arquivo .env
# Isso é importante para garantir que as chaves de API e outras configurações sensíveis estejam disponíveis no ambiente
load_dotenv()

# Obtendo a chave da API da OpenAI a partir das variáveis de ambiente
# A chave é armazenada no arquivo .env com o nome OPENAI_API_KEY
os.environ['OPENAI_API_KEY']= st.secrets["OPENAI_API_KEY"]

# Inicializando o modelo de linguagem GPT-4 mini da OpenAI
# Esse modelo é uma versão otimizada e menor do GPT-4, adequada para tarefas que exigem menos recursos
from langchain_openai import ChatOpenAI

gpt_4o_mini = ChatOpenAI(model="gpt-4o-mini")

# Criando o agente de análise de preços de ações
analista_precos_acoes = Agent(
    role="Analista Financeiro Senior de Preços de Ações",  # Define o papel do agente
    backstory=(
        "Você é um experiente analista financeiro com anos de experiência no mercado de ações. "
        "Sua especialidade é analisar tendências e prever movimentos de preços de ações com base em dados históricos e atuais. "
        "Você trabalha principalmente com análise técnica e fundamentalista para fornecer insights detalhados sobre o comportamento de ações."
    ),  # Fornece um contexto sobre a experiência e especialidade do agente
    goal="Buscar os dados do {ticket} e analisar detalhadamente as tendências, fornecendo insights precisos sobre o comportamento futuro da ação.",
    # Objetivo do agente
    verbose=True,  # Habilita saídas detalhadas durante a execução do agente
    llm=gpt_4o_mini,  # Especifica o modelo de linguagem que o agente irá utilizar
    max_iter=5,  # Define o número máximo de iterações que o agente pode realizar
    memory=True,  # Habilita a memória para que o agente possa reter informações ao longo das iterações
    tools=[yahoo_finance_tool]
    # Ferramentas que o agente irá utilizar, neste caso, o YahooFinanceTool para buscar dados financeiros
)

# Criando a tarefa para obter o preço das ações
tarefa_obter_preco = Task(
    description=(
        "Você deve buscar os dados históricos de preços da ação especificada pelo ticket {ticket} "
        "utilizando a ferramenta Yahoo Finance Tool. A análise deve cobrir o período de um ano, "
        "começando 1 ano antes da data atual até o dia presente. O resultado deve incluir informações "
        "como preço de abertura, preço de fechamento, volume de negociações, entre outros dados relevantes."
    ),  # Descrição da tarefa que o agente deve executar
    expected_output=(
        "Um DataFrame contendo os dados históricos de preços das ações do ticket {ticket}, "
        "com colunas para datas, preços de abertura, fechamento, volume, etc. O DataFrame deve estar "
        "pronto para ser usado em análises financeiras detalhadas."
    ),  # Saída esperada da tarefa, ou seja, o que o agente deve retornar ao final
    tools=[yahoo_finance_tool],  # Ferramentas que o agente utilizará para realizar a tarefa
    agent=analista_precos_acoes  # O agente responsável por executar a tarefa
)

# Inicializando a Ferramenta de Pesquisa
ferramenta_pesquisa = DuckDuckGoSearchResults(
    backend='news',  # Define o backend para pesquisar especificamente em notícias
    num_results=10  # Especifica o número de resultados que a pesquisa deve retornar
)

# Criando o agente de análise de notícias do mercado de ações
analista_noticias = Agent(
    role="Analista de Notícias relacionadas ao mercado de ações",  # Define o papel do agente
    backstory=(
        "Você é um analista veterano, em seus 15 anos de experiência, com profundo conhecimento do mercado de ações, especializado em interpretar "
        "e analisar notícias que podem impactar os preços das ações. Seu instinto apurado e experiência permitem "
        "identificar rapidamente quais notícias são mais relevantes para os investidores. Você é mestre em analisar mercados tradicionais e tem um entendimento profundo da psicologia humana. Você entende, correlaciona e compreende as notícias e seus títulos, mas sempre com uma dose saudável de bom senso e ceticismo, além de sempre considerar a fonte da notícia, buscando sempre as confiáveis."
    ),  # Contexto sobre a experiência e especialidade do agente
    goal=(
        "Buscar e analisar as notícias mais recentes e relevantes relacionadas ao mercado de ações. "
        "Com base nessas notícias, você deve identificar a tendência atual de cada ação: se está subindo, "
        "descendo ou se mantendo estável. Além disso, você deve classificar a situação de cada ação numericamente "
        "em uma escala de 0 a 100, onde 0 indica medo extremo (desfavorável à compra) e 100 indica uma excelente "
        "oportunidade de aquisição. A análise deve incluir a notícia específica que fundamenta sua avaliação."
    ),  # Objetivo claro do agente
    verbose=True,  # Habilita saídas detalhadas durante a execução do agente
    llm=gpt_4o_mini,  # Especifica o modelo de linguagem que o agente irá utilizar
    max_iter=5,  # Define o número máximo de iterações que o agente pode realizar
    memory=True,  # Habilita a memória para que o agente possa reter informações ao longo das iterações
    tools=[ferramenta_pesquisa]
    # Ferramentas que o agente irá utilizar, neste caso, uma ferramenta de busca de notícias
)

# Criando a tarefa para obter notícias relevantes do mercado de ações
tarefa_obter_noticias = Task(
    description=(
        "Você deve buscar as notícias mais recentes e relevantes relacionadas ao mercado de ações, principalmente sobre o ticket: {ticket} usando a ferramenta de pesquisa. "
        "Concentre-se em identificar artigos de notícias que possam impactar os preços das ações, como anúncios de ganhos, mudanças na liderança de empresas, "
        f"tensões geopolíticas, relatórios econômicos, e outras informações que possam influenciar o comportamento do mercado. Sempre inclua BTC (se não foi requisitado). A data atual é {datetime.now()}."
    ),  # Descrição da tarefa que o agente deve executar
    expected_output=(
        "Uma relatório extremamente útil das notícias relevantes sobre o mercado de ações, incluindo o título da notícia, a fonte, e um resumo de como cada notícia pode impactar o mercado. "
        "O agente deve também classificar o impacto potencial de cada notícia em uma escala de 0 a 100, onde 0 indica impacto negativo extremo e 100 indica impacto positivo extremo para cada asset analisado. use a seguinte estrutura:"
        "# Ação avaliada"
        "## Relatório da ação baseado nas notícias"
        "## Previsão da Tendência"
        "### Pontuação do Impacto Negativo/Positivo "
    ),  # Saída esperada da tarefa, ou seja, o que o agente deve retornar ao final
    agent=analista_noticias  # O agente responsável por executar a tarefa
)

# Criando o agente de análise e escrita de relatórios financeiros
escritor_relatorio_financeiro = Agent(
    role="Senior Stock Analyst Writer",  # Define o papel do agente
    backstory=(
        "Você é amplamente reconhecido como o melhor analista de ações do mercado. "
        "Com uma profunda compreensão dos conceitos financeiros mais complexos, você tem a capacidade única de criar histórias "
        "e narrativas que ressoam com uma audiência ampla, desde investidores iniciantes até profissionais experientes. "
        "Você é um especialista em interpretar fatores macroeconômicos e em combinar múltiplas teorias, como teoria dos ciclos "
        "e análises fundamentalistas, para fornecer uma visão abrangente e informativa do mercado. Sua habilidade de comunicar "
        "esses insights de forma clara e convincente faz de você uma figura central no mundo das finanças."
    ),  # Contexto sobre a experiência e especialidade do agente
    goal=(
        "Seu objetivo é criar um boletim informativo de 3 parágrafos que seja perspicaz, envolvente e informativo, "
        "baseado em relatórios de ações e nas tendências de preços. Esse boletim deve sintetizar as análises de mercado, "
        "apresentando uma narrativa que destaque oportunidades de investimento promissoras e identifique os riscos potenciais. "
        "Através desse boletim, você deve ajudar os investidores a entender as dinâmicas atuais do mercado e tomar decisões "
        "informadas, utilizando uma combinação de fatores macroeconômicos e análises detalhadas das tendências de preço."
    ),  # Objetivo claro do agente
    verbose=True,  # Habilita saídas detalhadas durante a execução do agente
    llm=gpt_4o_mini,  # Especifica o modelo de linguagem que o agente irá utilizar
    max_iter=5,  # Define o número máximo de iterações que o agente pode realizar
    memory=True,  # Habilita a memória para que o agente possa reter informações ao longo das iterações
    allow_delegation=True
)
# Obtendo a data e hora atuais para nomear o arquivo de saída
data_hora_atual = datetime.now().strftime("%Y%m%d_%H%M%S")

# Criando a tarefa para o agente escrever o relatório financeiro final
tarefa_escrever_relatorio_final = Task(
    description=(
        """
        Use a tendência do preço das ações e o relatório de notícias do mercado para criar uma análise e escrever um boletim informativo sobre a empresa {ticket}. 
        O relatório deve ser breve e destacar os pontos mais importantes, com foco na tendência do preço das ações, nas notícias e no índice de medo/ganância. 
        Quais são as considerações para o futuro próximo? Inclua as análises anteriores das tendências de ações e o resumo das notícias.
        O relatório deve ser escrito em português do Brasil.
        """
    ),  # Descrição da tarefa que o agente deve executar
    expected_output=(
        "Um boletim informativo breve e informativo sobre a empresa {ticket}, destacando as tendências de preço das ações, as notícias e o índice de medo/ganância. "
        "O relatório deve também incluir considerações para o futuro próximo e um resumo das análises anteriores das tendências de ações e das notícias. "
        "A saída final deve ser salva em um arquivo markdown."
    ),  # Saída esperada da tarefa, ou seja, o que o agente deve retornar ao final
    context=[tarefa_obter_preco, tarefa_obter_noticias],
    # Tarefas anteriores cujo contexto será utilizado para a escrita do relatório
    agent=escritor_relatorio_financeiro,  # O agente responsável por executar a tarefa
    output_file=f"relatório_{data_hora_atual}.md"
    # Nome do arquivo de saída, usando o ticket da empresa e a data e hora atuais
)

# Configuração do Crew para a execução das tarefas
crew = Crew(
    agents=[
        analista_precos_acoes,  # Agente responsável por analisar os preços das ações
        analista_noticias,  # Agente responsável por analisar as notícias relacionadas ao mercado de ações
        escritor_relatorio_financeiro  # Agente responsável por escrever o relatório financeiro final
    ],
    tasks=[
        tarefa_obter_preco,  # Tarefa para obter os preços das ações
        tarefa_obter_noticias,  # Tarefa para obter as notícias relevantes
        tarefa_escrever_relatorio_final  # Tarefa para escrever o relatório financeiro final
    ],
    verbose=True,  # Habilita saídas detalhadas durante a execução do Crew
    process=Process.hierarchical,  # Define o processo como hierárquico, onde as tarefas podem depender umas das outras
    full_output=True,  # Garante que o output completo seja retornado após a execução
    share_crew=False,  # Especifica que este Crew não será compartilhado com outros processos
    manager_llm=gpt_4o_mini,  # Define o modelo de linguagem que gerenciará o processo de execução do Crew
    max_iter=15  # Define o número máximo de iterações que o Crew pode executar
)


with st.sidebar:
    st.header("Digite o Ticket: ")
    with st.form(key='research_form'):
        ticket = st.text_input("Selecione o ticket")
        submit_button = st.form_submit_button(label='Pesquisar')
if submit_button:
    if not ticket:
        st.error("Digite o ticket")
    else:
        results = crew.kickoff(inputs={'ticket': ticket})

        st.subheader("Resultados da Pesquisa")
        st.write(results)
