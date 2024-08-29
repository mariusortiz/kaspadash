import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Définir la date de genèse pour Bitcoin
genesis_date = pd.Timestamp('2009-01-03')

# Charger les données de prix
df = pd.read_csv('btc_d.csv')
df['date'] = pd.to_datetime(df['date'])

# Calculer les jours écoulés depuis la genèse
df['days_from_genesis'] = (df['date'] - genesis_date).dt.days

# Paramètres de la Power Law pour Bitcoin
exp = 5.82
fair_coefficient = 1.0117e-17  # Coefficient pour la bande verte (Fair Price)

# Calculer le Fair Price et le Bottom Price
df['fair_price'] = fair_coefficient * (df['days_from_genesis']**exp)
df['bottom_price'] = df['fair_price'] * 0.42

# Définir les multiplicateurs pour les autres bandes
multipliers_above = np.geomspace(1, 2.38, 3)  # Pour les courbes au-dessus du fair price (Sell, Expensive, Pricey)
multipliers_below = np.geomspace(0.42, 1, 3)  # Pour les courbes en dessous du fair price (Cheap, Buy, Bad news)

# Appliquer les multiplicateurs pour calculer les autres bandes
df['sell_price'] = df['fair_price'] * multipliers_above[-1]
df['expensive_price'] = df['fair_price'] * multipliers_above[1]
df['pricey_price'] = df['fair_price'] * multipliers_above[0]
df['cheap_price'] = df['fair_price'] * multipliers_below[1]
df['buy_price'] = df['fair_price'] * multipliers_below[0]

# Créer un DataFrame pour le Rainbow Chart
rainbow_data = []
colors = {
    'purple': 'bottom_price',
    'blue': 'buy_price',
    'light_blue': 'cheap_price',
    'green': 'fair_price',
    'yellow': 'pricey_price',
    'orange': 'expensive_price',
    'red': 'sell_price'
}

for color, column in colors.items():
    for date, price in zip(df['date'], df[column]):
        rainbow_data.append({'date': date, 'price': price, 'color': color})

# Ajouter les données futures pour les 24 prochains mois
future_days = np.arange(1, 365 * 2 + 1)  # 24 mois = 365 * 2 jours
last_day_from_genesis = df['days_from_genesis'].max()
future_days_from_genesis = last_day_from_genesis + future_days
future_dates = [df['date'].max() + timedelta(days=int(day)) for day in future_days]

for color, column in colors.items():
    future_prices = df[column].iloc[-1] * (future_days_from_genesis / last_day_from_genesis)**exp
    for date, price in zip(future_dates, future_prices):
        rainbow_data.append({'date': date, 'price': price, 'color': color})

# Créer un DataFrame final pour le Rainbow Chart
rainbow_df = pd.DataFrame(rainbow_data)

# Enregistrer dans un fichier CSV (immuable)
rainbow_df.to_csv('rainbow_chart_data_btc.csv', index=False)
