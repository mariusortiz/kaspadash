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

# Ajuster les multiplicateurs pour écarter les bandes de manière uniforme
blue_multiplier = 0.33  # Pour écarter la bande bleue
green_multiplier = 0.66  # Pour écarter la bande verte
yellow_multiplier = 1.0  # La bande jaune reste inchangée
orange_multiplier = 1.33  # Pour écarter la bande orange
red_multiplier = 1.66  # Pour écarter la bande rouge

# Calculer les prix pour chaque bande
df['fair_price'] = fair_coefficient * (df['days_from_genesis']**exp)

# Créer un DataFrame pour le Rainbow Chart
rainbow_data = []

colors = ['blue', 'green', 'yellow', 'orange', 'red']
multipliers = [blue_multiplier, green_multiplier, yellow_multiplier, orange_multiplier, red_multiplier]

for color, multiplier in zip(colors, multipliers):
    price_series = df['fair_price'] * multiplier
    for date, price in zip(df['date'], price_series):
        rainbow_data.append({'date': date, 'price': price, 'color': color})

# Ajouter les données futures pour les 24 prochains mois
future_days = np.arange(1, 365 * 2 + 1)  # 24 mois = 365 * 2 jours
last_day_from_genesis = df['days_from_genesis'].max()
future_days_from_genesis = last_day_from_genesis + future_days
future_dates = [df['date'].max() + timedelta(days=int(day)) for day in future_days]

for color, multiplier in zip(colors, multipliers):
    future_prices = df['fair_price'].iloc[-1] * multiplier * (future_days_from_genesis / last_day_from_genesis)**exp
    for date, price in zip(future_dates, future_prices):
        rainbow_data.append({'date': date, 'price': price, 'color': color})

# Créer un DataFrame final pour le Rainbow Chart
rainbow_df = pd.DataFrame(rainbow_data)

# Enregistrer dans un fichier CSV (immuable)
rainbow_df.to_csv('rainbow_chart_data_kas.csv', index=False)

