# Aplicação Web Para Previsão de Preço de Casas 

# importanto as bibliotecas necessárias 
import pandas as pd
import streamlit as st
import plotly.express as px 
from sklearn.ensemble import RandomForestRegressor

# função para carregar o dataset
@st.cache
def get_data():
    return pd.read_csv('model/data.csv')

# função para treinar o modelo
def train_model():
    data = get_data()
    x = data.drop('MEDV', axis=1)
    y = data['MEDV']
    rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=7,max_features=3)
    rf_regressor.fit(x,y)
    return rf_regressor

# criando um dataframe
data = get_data()

# treinando o modelo 
model = train_model()

# Título 
st.title('Data App -  Prevendo Valores de Imóveis')

# Subtítulo
st.markdown('Esse é um Data App utilizado para exibir a solução de Machine Learning para o problema de predição')

# verificando o dataset
st.subheader('Selecionado apenas um pequeno conjunto de atributos')

# atributos para serem exibidos por padrão
defaultcols = ['RM','PTRATIO','LSTAT','MEDV']

# defiinindo atributos a partir de multiselect
cols = st.multiselect('Atributos', data.columns.tolist(), default = defaultcols)

# exibindo os top 10 registros do dataframe 
st.dataframe(data[cols].head(10))

st.subheader('Distribuição de Imóveis por Preço')

# definindo a faixa de valores 
faixa_valores = st.slider('Faixa de Preço', float(data.MEDV.min()),150.,(10.0, 100.0))

# filtrando os dados 
dados = data[data['MEDV'].between(left = faixa_valores[0], right = faixa_valores[1])]

# plota a distribuição dos dados 
f = px.histogram(dados, x="MEDV", nbins=100, title="Distribuição de Preços")
f.update_xaxes(title="MEDV")
f.update_yaxes(title="Total Imóveis")
st.plotly_chart(f)

st.sidebar.subheader('Defina os atributos do Imóvel para predição')

# mapeando dados do usuário para cada atributo 
crim = st.sidebar.number_input("Taxa de Criminalidade", value=data.CRIM.mean())
indus = st.sidebar.number_input("Proporção de Hectares de Negócio", value=data.CRIM.mean())
chas = st.sidebar.selectbox("Faz limite com o rio?",("Sim","Não"))

# transformando dado de entrada em valor binário 
chas = 1 if chas == 'Sim' else 0

nox = st.sidebar.number_input("Concentração de óxido nítrico", value=data.NOX.mean())

rm = st.sidebar.number_input("Número de Quartos", value=1)

ptratio = st.sidebar.number_input("Índice de alunos para professores",value=data.PTRATIO.mean())

b = st.sidebar.number_input("Proporção de pessoas com descendencia afro-americana",value=data.B.mean())

lstat = st.sidebar.number_input("Porcentagem de status baixo",value=data.LSTAT.mean())

# inserindo um botão na tela 
bnt_predict = st.sidebar.button('Realizar Predição')

# verificação se o botão foi ativado
if bnt_predict:
    result = model.predict([[crim,indus,chas,nox,rm,ptratio,b,lstat]])
    st.subheader('O valor previsto para o imóvel é:')
    result = 'US $ ' +str(round(result[0]*10,2))
    st.write(result)