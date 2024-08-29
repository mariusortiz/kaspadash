import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Définir la date de genèse
genesis_date = pd.Timestamp('2009-01-03')  # Date de genèse de Bitcoin
current_date = datetime.now()

# Charger les données de prix
df = pd.read_csv('btc_d.csv')
df['date'] = pd.to_datetime(df['date'])

# Calculer les jours écoulés depuis la genèse
df['days_from_genesis'] = (df['date'] - genesis_date).dt.days

# Paramètres de la Power Law pour Bitcoin
exp = 5.82
fair_coefficient = 1.0117e-17  # Coefficient pour la bande verte (fair price)
bottom_multiplier = 0.42  # Coefficient pour la bande violette (bottom price)
top_multiplier = 1 / bottom_multiplier  # Symétrique de la bande rouge par rapport à la bande verte

# Calculer les prix pour chaque bande
df['fair_price'] = fair_coefficient * (df['days_from_genesis']**exp)
df['bottom_price'] = df['fair_price'] * bottom_multiplier
df['top_price'] = df['fair_price'] * top_multiplier

# Créer des multiplicateurs équidistants entre bottom et fair, et entre fair et top
multipliers_below_fair = np.geomspace(bottom_multiplier, 1, 3)  # 3 bandes en dessous de "fair_price"
multipliers_above_fair = np.geomspace(1, top_multiplier, 3)  # 3 bandes au-dessus de "fair_price"

colors = ['purple', 'light_blue', 'green', 'yellow', 'orange', 'red']
price_columns = {
    'purple': 'bottom_price',
    'green': 'fair_price',
    'red': 'top_price',
    'light_blue': None,  # Sera calculé avec un multiplicateur
    'yellow': None,      # Sera calculé avec un multiplicateur
    'orange': None       # Sera calculé avec un multiplicateur
}

# Ajouter les bandes calculées au DataFrame
for i, color in enumerate(colors):
    if price_columns[color]:
        df[f'{color}_price'] = df[price_columns[color]]
    elif color in ['light_blue', 'yellow']:  # Bandes entre "bottom" et "fair"
        multiplier = multipliers_below_fair[['light_blue', 'yellow'].index(color)]
        df[f'{color}_price'] = df['fair_price'] * multiplier
    elif color in ['orange']:  # Bande entre "fair" et "top"
        multiplier = multipliers_above_fair[0]  # Seulement une bande entre "fair" et "top"
        df[f'{color}_price'] = df['fair_price'] * multiplier

# Générer les données pour chaque bande
rainbow_data = []
for color in colors:
    column = f'{color}_price'
    for date, price in zip(df['date'], df[column]):
        rainbow_data.append({'date': date, 'price': price, 'color': color})

# Ajouter les données futures pour les 24 prochains mois
future_days = np.arange(1, 365 * 2 + 1)  # 24 mois = 365 * 2 jours
last_day_from_genesis = df['days_from_genesis'].max()
future_days_from_genesis = last_day_from_genesis + future_days
future_dates = [df['date'].max() + timedelta(days=int(day)) for day in future_days]

# Calculer les prix futurs pour chaque bande
for color in colors:
    column = f'{color}_price'
    future_prices = df[column].iloc[-1] * (future_days_from_genesis / last_day_from_genesis)**exp
    for date, price in zip(future_dates, future_prices):
        rainbow_data.append({'date': date, 'price': price, 'color': color})

# Créer un DataFrame final pour le Rainbow Chart
rainbow_df = pd.DataFrame(rainbow_data)

# Enregistrer dans un fichier CSV (immuable)
rainbow_df.to_csv('rainbow_chart_data_btc.csv', index=False)
