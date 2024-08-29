import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Définir la date de genèse pour Bitcoin
genesis_date = datetime(2009, 1, 3)
current_date = datetime.now()

# Charger les données de prix pour BTC
df = pd.read_csv('btc_d.csv')
df['date'] = pd.to_datetime(df['date'])

# Calculer les jours écoulés depuis la genèse
df['days_from_genesis'] = (df['date'] - genesis_date).dt.days

# Paramètres de la Power Law pour Bitcoin
exp = 5.82
fair_coefficient = 1.0117e-17  # Coefficient pour la bande verte (juste valeur)

# Multiplier pour écarter les bandes (logique x1,5)
multipliers = {
    'purple': 0.42,   # "Bad news"
    'dark_blue': 0.563, # "Buy"
    'light_blue': 0.755, # "Cheap"
    'green': 1,    # "Fair Price"
    'yellow': 1.34,  # "Pricey"
    'orange': 1.79, # "Expensive"
    'red': 2.38    # "Sell"
}

# Calculer les prix pour chaque bande
df['fair_price'] = fair_coefficient * (df['days_from_genesis']**exp)

# Créer un DataFrame pour le Rainbow Chart
rainbow_data = []

colors = list(multipliers.keys())

for color, multiplier in multipliers.items():
    price_series = df['fair_price'] * multiplier
    for date, price in zip(df['date'], price_series):
        rainbow_data.append({'date': date, 'price': price, 'color': color})

# Ajouter les données futures pour les 24 prochains mois
future_days = np.arange(1, 365 * 2 + 1)  # 24 mois = 365 * 2 jours
last_day_from_genesis = df['days_from_genesis'].max()
future_days_from_genesis = last_day_from_genesis + future_days
future_dates = [df['date'].max() + timedelta(days=int(day)) for day in future_days]

for color, multiplier in multipliers.items():
    future_prices = df['fair_price'].iloc[-1] * multiplier * (future_days_from_genesis / last_day_from_genesis)**exp
    for date, price in zip(future_dates, future_prices):
        rainbow_data.append({'date': date, 'price': price, 'color': color})

# Créer un DataFrame final pour le Rainbow Chart
rainbow_df = pd.DataFrame(rainbow_data)

# Enregistrer dans un fichier CSV (immuable)
rainbow_df.to_csv('rainbow_chart_data_btc.csv', index=False)
