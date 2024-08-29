import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Charger les données historiques
df = pd.read_csv('btc_d.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Calculer ΔGB (jours depuis le Genesis Block)
genesis_date = pd.Timestamp('2009-01-03')
df['days_since_genesis'] = (df['date'] - genesis_date).dt.days

# Calculer les prix justes selon la loi de puissance pour Bitcoin
df['fair_price'] = 1.0117e-17 * (df['days_since_genesis']**5.82)
df['bottom_price'] = df['fair_price'] * 0.42

# Lissage des prix justes historiques
df['historical_fair_price_smooth'] = savgol_filter(df['fair_price'], window_length=51, polyorder=3)

# Enregistrer les données historiques dans un CSV
df.to_csv('historical_fair_price_btc.csv', index=False)

# Générer des dates futures
future_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=365)
future_df = pd.DataFrame({'date': future_dates})
future_df['days_since_genesis'] = (future_df['date'] - genesis_date).dt.days

# Calculer les prix futurs selon la loi de puissance pour Bitcoin
future_df['fair_price'] = 1.0117e-17 * (future_df['days_since_genesis']**5.82)
future_df['bottom_price'] = future_df['fair_price'] * 0.42
future_df['predicted_price'] = future_df['fair_price']  # Utilisation de 'predicted_price' pour correspondre à votre fonction

# Enregistrer les données futures dans un CSV
future_df.to_csv('future_prices_btc.csv', index=False)