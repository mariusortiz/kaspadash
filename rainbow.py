import pandas as pd
import numpy as np
from sklearn.linear_model import RANSACRegressor
from datetime import datetime, timedelta

# Charger les données de prix
df = pd.read_csv('kas_d.csv')
df['date'] = pd.to_datetime(df['date'])

# Définir les dates de référence
genesis_date = datetime(2021, 11, 7)
start_date = datetime(2022, 6, 15)  # Premier jour de kas_d.csv
end_date = df['date'].max()
future_days = np.arange(1, 365 * 2)

# Calculer les jours depuis la genèse
df['days_from_genesis'] = (df['date'] - genesis_date).dt.days
df = df[df['days_from_genesis'] >= 0]

# Appliquer la transformation logarithmique
df['log_close'] = np.log(df['close'])
df['log_days_from_genesis'] = np.log(df['days_from_genesis'])

# Ajuster le modèle RANSAC
X = df[['log_days_from_genesis']]
y = df['log_close']
model = RANSACRegressor().fit(X, y)
df['predicted_log_close'] = model.predict(X)

# Trouver les résidus pour les bandes
df['residuals'] = df['log_close'] - df['predicted_log_close']
highest_residual_index = df['residuals'].idxmax()
lowest_residual_index = df['residuals'].idxmin()

# Calculer les pentes et les interceptions ajustées
slope = model.estimator_.coef_[0]
intercept_high = df.loc[highest_residual_index, 'log_close'] - (slope * df.loc[highest_residual_index, 'log_days_from_genesis'])
intercept_low = df.loc[lowest_residual_index, 'log_close'] - (slope * df.loc[lowest_residual_index, 'log_days_from_genesis'])

# Générer des données futures
last_day_from_genesis = np.log(df['days_from_genesis'].max() + 1)
future_log_days_from_genesis = np.log(df['days_from_genesis'].max() + 1 + future_days)
future_days_from_genesis = np.exp(future_log_days_from_genesis)
future_dates = [genesis_date + timedelta(days=int(day)) for day in future_days_from_genesis]

# Créer les bandes de couleurs
colors = ['blue', 'green', 'yellow', 'orange', 'red']
num_bands = 3
intercepts_original = []

for i in range(num_bands + 2):
    intercept_band = intercept_low + i * (intercept_high - intercept_low) / (num_bands + 1)
    intercepts_original.append(intercept_band)

# Créer un DataFrame pour les données du Rainbow Chart
rainbow_data = []

# Ajouter les données historiques
for i, intercept in enumerate(intercepts_original):
    y_values = slope * df['log_days_from_genesis'] + intercept
    color = colors[i % len(colors)]
    for date, price in zip(df['date'], np.exp(y_values)):
        rainbow_data.append({'date': date, 'price': price, 'color': color})

# Ajouter les données futures
for i, intercept in enumerate(intercepts_original):
    y_values = slope * future_log_days_from_genesis + intercept
    color = colors[i % len(colors)]
    for date, price in zip(future_dates, np.exp(y_values)):
        rainbow_data.append({'date': date, 'price': price, 'color': color})

rainbow_df = pd.DataFrame(rainbow_data)

# Enregistrer dans un fichier CSV
rainbow_df.to_csv('rainbow_chart_data_kas.csv', index=False)