import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import statsmodels.api as sm

dados=pd.read_csv('pintos.csv') 

print(dados.head())
print(dados.shape)

print(dados.describe())


plt.figure(figsize = (16,8))

X = dados['agua'].values
Y = dados['pintos'].values

r = pearsonr(X,Y)
print(f'Coeficiente de correlação dos dados: {r}')

# Criar um modelo para prever numero de pintos com base nos dados existentes
Xs = dados.drop(['pintos', 'ano'], axis=1)


X = np.column_stack((dados['agua'], dados['racao']))
y = dados['pintos'].values.reshape(-1,1)

reg = LinearRegression()
reg.fit(X, y)

print("Modelo Pintos = {:.5} + {:.5}*Agua + {:.5}*Racao".format(reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]))


previsao = reg.predict(Xs)


plt.scatter(Xs, X,  c='blue')

plt.xlabel("Quantidade de Agua")
plt.ylabel("Numero de Pintos")

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)

est2 = est.fit()
print(est2.summary())

# # previsao
plt.plot( X,    previsao,    c='red',    linewidth=3,    linestyle=':')
plt.show()


