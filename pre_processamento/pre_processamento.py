import requests
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def _column_in_ranges(table, column, size=10):
    col = table[column]
    h = size
    lmt_min, lmt_max = -h, 0
    while lmt_max <= col.max():
        lmt_min += h
        lmt_max += h
        col[(col>=lmt_min) & (col<lmt_max)] = lmt_min

def columns_in_ranges(table, columns, sizes=[]):
    for i in range(len(columns)):
        if len(sizes) == 0:
            _column_in_ranges(table, columns[i])
        else:
            _column_in_ranges(table, columns[i], sizes[i])
        
def replace_nan_with_mean(columns_and_indices, table):
  for i, name in list(columns_and_indices):
    table.iloc[(table[name].isna()), i] = float(f'{table[name].mean():.1f}')

def min_max_scaler(table, columns):
    for col in columns:
        column = table[col]
        _min, _max = column.min(), column.max()
        column /= (_max - _min)
        column -= _max / (_max - _min)
        table[col] = 1 + column

def scale(table):
    scaler = MinMaxScaler()
    cols = feature_cols
    #df = pd.DataFrame(scaler.fit_transform(table[cols]), columns=cols)
    min_max_scaler(table, cols)
    #table[cols] = df[cols]

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('data/diabetes_dataset.csv')

# Criando X and y par ao algorítmo de aprendizagem de máquina.\
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
# Caso queira modificar as colunas consideradas basta algera o array a seguir.
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']


#remove rows
interest = data.iloc[:, 1:7]
nan_count = interest.isna().sum(axis=1)  # soma na linha
X = data.loc[~(nan_count >= 3)]
interest = feature_cols[:]
interest.append('Outcome')
X = X[interest]

# na tabela de treino
names = enumerate(feature_cols[1:5], 1)
range_columns = ['Age']
range_h = [10]
replace_nan_with_mean(names, X)
columns_in_ranges(X, range_columns, range_h)
#scale(X)


y = X.Outcome
X = X[feature_cols]

# Ciando o modelo preditivo para a base trabalhada
print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

#realizando previsões com o arquivo de
print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('data/diabetes_app.csv')
data_app = data_app[feature_cols]
columns_in_ranges(data_app, range_columns, range_h)
#scale(data_app)
y_pred = neigh.predict(data_app)
y_pred


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
