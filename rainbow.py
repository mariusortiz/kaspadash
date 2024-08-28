import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Définir la date de genèse
genesis_date = datetime(2021, 11, 7)
current_date = datetime.now()

# Charger les données de prix
df = pd.read_csv('kas_d.csv')
df['date'] = pd.to_datetime(df['date'])

# Calculer les jours écoulés depuis la genèse
df['days_from_genesis'] = (df['date'] - genesis_date).dt.days

# Paramètres de la Power Law
exp = 4.218461
support_coefficient = 10**-13.41344198
resistance_coefficient = 10**-13.10611888
fair_coefficient = 10**-13.25978043

# Augmenter les multiplicateurs pour écarter les bandes
support_multiplier = 0.5  # Moins que 1 pour écarter davantage la bande bleue
resistance_multiplier = 2.0  # Plus que 1 pour écarter davantage la bande rouge

# Calculer les prix pour chaque bande
df['support_price'] = support_coefficient * (df['days_from_genesis']**exp)
df['resistance_price'] = resistance_coefficient * (df['days_from_genesis']**exp)
df['fair_price'] = fair_coefficient * (df['days_from_genesis']**exp)

# Créer un DataFrame pour le Rainbow Chart
rainbow_data = []

colors = ['blue', 'green', 'yellow', 'orange', 'red']
bands = {
    'blue': df['support_price'] * support_multiplier,
    'green': df['support_price'] * (resistance_coefficient/support_coefficient)**(1/4),
    'yellow': df['fair_price'],
    'orange': df['resistance_price'] * (support_coefficient/resistance_coefficient)**(1/4),
    'red': df['resistance_price'] * resistance_multiplier
}

# Ajouter les bandes historiques
for color, price_series in bands.items():
    for date, price in zip(df['date'], price_series):
        rainbow_data.append({'date': date, 'price': price, 'color': color})

# Ajouter les données futures pour les 24 prochains mois
future_days = np.arange(1, 365 * 2 + 1)  # 24 mois = 365 * 2 jours
last_day_from_genesis = df['days_from_genesis'].max()
future_days_from_genesis = last_day_from_genesis + future_days
future_dates = [df['date'].max() + timedelta(days=int(day)) for day in future_days]

for color, intercept in bands.items():
    future_prices = intercept.iloc[-1] * (future_days_from_genesis / last_day_from_genesis)**exp
    for date, price in zip(future_dates, future_prices):
        rainbow_data.append({'date': date, 'price': price, 'color': color})

# Créer un DataFrame final pour le Rainbow Chart
rainbow_df = pd.DataFrame(rainbow_data)

# Enregistrer dans un fichier CSV (immuable)
rainbow_df.to_csv('rainbow_chart_data_kas.csv', index=False)
print("Le fichier 'rainbow_chart_data_kas.csv' a été mis à jour avec les bandes élargies et les prévisions sur 24 mois.")
