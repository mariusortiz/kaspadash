import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Définir la date de genèse
genesis_date = datetime(2021, 11, 7)
current_date = datetime.now()

# Calculer les jours écoulés depuis la genèse
df = pd.read_csv('kas_d.csv')
df['date'] = pd.to_datetime(df['date'])
df['days_from_genesis'] = (df['date'] - genesis_date).dt.days

# Paramètres de la Power Law
exp = 4.218461
support_coefficient = 10**-13.41344198
resistance_coefficient = 10**-13.10611888
fair_coefficient = 10**-13.25978043

# Calculer les prix pour chaque bande
df['support_price'] = support_coefficient * (df['days_from_genesis']**exp)
df['resistance_price'] = resistance_coefficient * (df['days_from_genesis']**exp)
df['fair_price'] = fair_coefficient * (df['days_from_genesis']**exp)

# Créer un DataFrame pour le Rainbow Chart
rainbow_data = []

colors = ['blue', 'green', 'yellow', 'orange', 'red']
bands = {
    'blue': df['support_price'],
    'green': df['support_price'] * (resistance_coefficient/support_coefficient)**(1/4),
    'yellow': df['fair_price'],
    'orange': df['resistance_price'] * (support_coefficient/resistance_coefficient)**(1/4),
    'red': df['resistance_price']
}

for color, price_series in bands.items():
    for date, price in zip(df['date'], price_series):
        rainbow_data.append({'date': date, 'price': price, 'color': color})

rainbow_df = pd.DataFrame(rainbow_data)

# Enregistrer dans un fichier CSV (immuable)
rainbow_df.to_csv('rainbow_chart_data_kas.csv', index=False)