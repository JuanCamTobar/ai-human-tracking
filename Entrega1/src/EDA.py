# -*- coding: utf-8 -*-
"""Untitled33.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cJV5EvYphAr8cJfe5Me6u0pFRGKtxJd2

#EDA
"""

pip install pandas matplotlib seaborn

"""###Datos caminar girar"""

import pandas as pd

df = pd.read_csv('/content/datosCaminarGirar.csv', delimiter=';')

print(df.head())

print(df.info())

print(df.describe())

print(df.isnull().sum())

for col in ['landmark_23_z', 'landmark_24_z', 'landmark_25_z', 'landmark_26_z',
            'pie_izquierdo_prom_y', 'pie_izquierdo_prom_z', 'pie_derecho_prom_y']:
    df[col] = pd.to_numeric(df[col], errors='coerce')


df['Etiqueta'] = df['Etiqueta'].astype('category')

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Estadísticas descriptivas solo para columnas numéricas
df.describe()

df['Etiqueta'].value_counts()

sns.countplot(x='Etiqueta', data=df)
plt.title('Distribución de Etiquetas')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

df.hist(bins=30, figsize=(10, 8))
plt.show()

import seaborn as sns

# Calcular correlación entre columnas numéricas
corr = df.select_dtypes(include=['float64', 'int64']).corr()

# Hacer el heatmap
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='coolwarm')
plt.title('Mapa de calor de correlaciones')
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['landmark_10_x'], df['landmark_10_y'], df['landmark_10_z'], c='blue', s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Distribución 3D de landmark_10')
plt.show()

# prompt: Quiero un grafico de dispersion normal

import matplotlib.pyplot as plt
# Crear un gráfico de dispersión para dos columnas numéricas
plt.figure(figsize=(8, 6))
sns.scatterplot(x='landmark_10_x', y='landmark_10_y', data=df, hue='Etiqueta') # Agrega la columna 'Etiqueta' para el color
plt.title('Gráfico de Dispersión de landmark_10_x vs landmark_10_y')
plt.xlabel('landmark_10_x')
plt.ylabel('landmark_10_y')
plt.show()

"""#Datos inclinación"""

df = pd.read_csv('/content/datosInclinacion.csv', delimiter=';')

print(df.head())

print(df.info())

print(df.describe())



for col in ['landmark_11_z', 'landmark_12_y', 'landmark_12_z',
    'landmark_13_z', 'landmark_14_z', 'landmark_23_z',
    'landmark_24_z', 'cara_prom_y', 'mano_izquierda_prom_z',
    'pie_izquierdo_prom_y', 'pie_derecho_prom_y']:
    df[col] = pd.to_numeric(df[col], errors='coerce')


df['Etiqueta'] = df['Etiqueta'].astype('category')

df.info()

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

df.describe()

df['Etiqueta'].value_counts()

sns.countplot(x='Etiqueta', data=df)
plt.title('Distribución de Etiquetas')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

df.hist(bins=30, figsize=(10, 8))
plt.show()

import seaborn as sns

# Calcular correlación entre columnas numéricas
corr = df.select_dtypes(include=['float64', 'int64']).corr()

# Hacer el heatmap
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='coolwarm')
plt.title('Mapa de calor de correlaciones')
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['landmark_10_x'], df['landmark_10_y'], df['landmark_10_z'], c='blue', s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Distribución 3D de landmark_10')
plt.show()

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.scatterplot(x='landmark_10_x', y='landmark_10_y', data=df, hue='Etiqueta') # Agrega la columna 'Etiqueta' para el color
plt.title('Gráfico de Dispersión de landmark_10_x vs landmark_10_y')
plt.xlabel('landmark_10_x')
plt.ylabel('landmark_10_y')
plt.show()

"""#Datos sentarse"""

import pandas as pd

df = pd.read_csv('/content/datosSentarse.csv', delimiter=';')

print(df.head())

print(df.info())

for col in ['landmark_10_y', 'landmark_10_z', 'landmark_11_z',
    'landmark_12_z', 'landmark_13_y', 'landmark_13_z',
    'landmark_14_y', 'landmark_14_z', 'landmark_23_y',
    'landmark_23_z', 'landmark_24_y', 'landmark_24_z',
    'landmark_25_y', 'landmark_25_z', 'landmark_26_y',
    'landmark_26_z', 'cara_prom_y', 'cara_prom_z',
    'mano_izquierda_prom_y', 'mano_izquierda_prom_z',
    'mano_derecha_prom_y', 'pie_izquierdo_prom_y',
    'pie_izquierdo_prom_z', 'pie_derecho_prom_y',
    'pie_derecho_prom_z']:
    df[col] = pd.to_numeric(df[col], errors='coerce')


df['etiqueta'] = df['etiqueta'].astype('category')

df.info()

df.info()

df['Etiqueta'].value_counts()

sns.countplot(x='Etiqueta', data=df)
plt.title('Distribución de Etiquetas')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

df.hist(bins=30, figsize=(10, 8))
plt.show()

import seaborn as sns

# Calcular correlación entre columnas numéricas
corr = df.select_dtypes(include=['float64', 'int64']).corr()

# Hacer el heatmap
plt.figure(figsize=(12,10))
sns.heatmap(corr, cmap='coolwarm')
plt.title('Mapa de calor de correlaciones')
plt.show()