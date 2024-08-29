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

# Calculer les prix pour chaque bande
df['fair_price'] = fair_coefficient * (df['days_from_genesis']**exp)
df['bottom_price'] = df['fair_price'] * bottom_multiplier

# Répartir les autres bandes uniformément
num_bands = 4  # Nombre de bandes entre la bande "bottom" et "fair", et entre "fair" et "sell"
multipliers_below_fair = np.geomspace(bottom_multiplier, 1, num_bands)
multipliers_above_fair = np.geomspace(1, 3, num_bands)

colors = ['purple', 'blue', 'light_blue', 'green', 'yellow', 'orange', 'red']
price_columns = ['bottom_price', None, None, 'fair_price', None, None, None]

# Ajouter les bandes calculées au DataFrame
for i, color in enumerate(colors):
    if color == 'purple':
        df['purple_price'] = df['bottom_price']
    elif color == 'green':
        df['green_price'] = df['fair_price']
    elif i < len(multipliers_below_fair) + 1:  # Pour les bandes en dessous de "fair_price"
        multiplier = multipliers_below_fair[i - 1] if i > 0 else bottom_multiplier
        df[f'{color}_price'] = df['fair_price'] * multiplier
    else:  # Pour les bandes au-dessus de "fair_price"
        multiplier = multipliers_above_fair[i - len(multipliers_below_fair) - 1]
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
