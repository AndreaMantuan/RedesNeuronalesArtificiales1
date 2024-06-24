
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# URL del archivo CSV en tu repositorio de GitHub
url = 'https://github.com/AndreaMantuan/RedesNeuronalesArtificiales1/raw/main/regresion.csv'

# Leer el dataset
df = pd.read_csv(url)
print(df.head())

# Preprocesar los datos
X = df.drop('Precio (en miles)', axis=1)
y = df['Precio (en miles)']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar las características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear la red neuronal artificial
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))  # Para regresión

# Compilar el modelo
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

# Resumen del modelo
model.summary()

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluar el modelo
loss, mse = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')
print(f'Mean Squared Error: {mse}')
