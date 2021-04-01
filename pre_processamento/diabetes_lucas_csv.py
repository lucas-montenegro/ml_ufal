#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
import requests

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')

# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

if('SkinThickness' in data.columns):
    data.drop('SkinThickness', axis='columns', inplace=True) 
    
#data = data.dropna(subset=['BMI', 'Glucose'])    
data.drop_duplicates(inplace=True)
data.reset_index(drop=True, inplace=True)

X = data[feature_cols]
y = data.Outcome

# substitui por todos os valores, independente se tem diabetes ou não
def replace_nan_with_mean(columns):
  for column in columns:
    X.loc[(X[column].isna()), column] = X[column].mean().round(3)


def normalize_data(data_frame, columns):
  for column in columns:
    normalized_column = (data_frame[column] - data_frame[column].min()) / (data_frame[column].max() - data_frame[column].min())
    data_frame[column] = normalized_column.round(4)

def create_interval(data_frame, columns):
    for column in columns:
        data_frame[column] = data_frame[column] / 10
        data_frame[column] = data_frame[column].astype(int)

def knn_imputer(columns_and_indexes):
    imputer = KNNImputer(n_neighbors=10, weights="uniform")
    result = pd.DataFrame(imputer.fit_transform(data))
    
    for index, column in columns_and_indexes:
        X[column] = result[index]

columns_and_indexes = enumerate(['Glucose', 'BloodPressure', 'Insulin', 'BMI'], start=1)

knn_imputer(columns_and_indexes)
#replace_nan_with_mean(X.columns)
create_interval(X, ['Age'])
normalize_data(X, ['Glucose', 'BloodPressure'])

# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')

create_interval(data_app, ['Age'])
normalize_data(data_app, ['Glucose', 'BloodPressure'])

if('SkinThickness' in data_app.columns):
    data_app.drop('SkinThickness', axis='columns', inplace=True) 

y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "LW"

# json para ser enviado para o servidor
data = {'dev_key':DEV_KEY,
        'predictions':pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url = URL, data = data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")
