import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from pandas import DataFrame
import statsmodels.api as sm

dados=pd.read_csv('pintos.csv') #o meu arquivo com os dados está na mesma pasta que o arquivo do código

print(dados.head())
print(dados.shape)

print(dados.describe())


plt.figure(figsize = (16,8))

X = dados['agua'].values
Y = dados['pintos'].values

r = pearsonr(X,Y)
print(f'Coeficiente de correlação dos dados: {r}')

# Criar um modelo para prever numero de pintos com base nos dados existentes
x = dados['agua'].values.reshape(-1,1)
y = dados['pintos'].values.reshape(-1,1)

reg = LinearRegression()

reg.fit(x, y)

print("Pintos = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

previsao = reg.predict(y)


plt.scatter(X, Y,  c='blue')

plt.xlabel("Quantidade de Agua")
plt.ylabel("Numero de Pintos")

# previsao
plt.plot( X,    previsao,    c='red',    linewidth=3,    linestyle=':')
plt.show()

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)

est2 = est.fit()
print(est2.summary())


