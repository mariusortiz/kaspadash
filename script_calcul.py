import pandas as pd
import numpy as np
import powerlaw
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from scipy.signal import savgol_filter

# Charger les données
df = pd.read_csv('kas_d.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Filtrer les données nécessaires
data = df['close'].dropna().values

# Ajuster la distribution avec powerlaw
results = powerlaw.Fit(data)
alpha = results.power_law.alpha
xmin = results.power_law.xmin
print(f"Alpha: {alpha}, Xmin: {xmin}")

# Application de la régression RANSAC glissante
window_size = 30  # Fenêtre glissante de 30 jours
historical_fair_price = []

for i in range(len(df) - window_size):
    window_data = data[i:i + window_size]
    
    ransac = RANSACRegressor(residual_threshold=5.0)
    log_days = np.log(np.arange(1, len(window_data) + 1)).reshape(-1, 1)
    if len(log_days) == len(window_data):  # Assurez-vous que les longueurs sont cohérentes
        ransac.fit(log_days, np.log(window_data))
        fair_price = np.exp(ransac.predict(np.log([len(window_data)]).reshape(-1, 1)))[0]
        historical_fair_price.append(fair_price)
    else:
        historical_fair_price.append(np.nan)

# Compléter les données manquantes
historical_fair_price = [np.nan] * window_size + historical_fair_price

# Appliquer un filtre de lissage
historical_fair_price_smooth = savgol_filter(historical_fair_price, window_length=51, polyorder=3)

# Ajouter au dataframe
df['historical_fair_price'] = historical_fair_price
df['historical_fair_price_smooth'] = historical_fair_price_smooth

# Charger les données du Rainbow Chart
rainbow_df = pd.read_csv('rainbow_chart_data_kas.csv')
rainbow_df['date'] = pd.to_datetime(rainbow_df['date'])
yellow_band_df = rainbow_df[rainbow_df['color'] == 'yellow']

# Générer des dates futures
future_dates = pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=365)
future_timestamps = future_dates.astype(np.int64) // 10**9
timestamps_rainbow = yellow_band_df['date'].astype(np.int64) // 10**9

# Interpolation pour obtenir la tendance future
interpolated_trend = np.interp(future_timestamps, timestamps_rainbow, yellow_band_df['price'])

# Ajouter un lissage exponentiel
def exponential_smoothing(series, alpha):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result

future_prices_smooth = exponential_smoothing(interpolated_trend, alpha=0.1)

# Créer un dataframe pour les prévisions futures
future_df = pd.DataFrame({'date': future_dates, 'predicted_price': interpolated_trend, 'predicted_price_smooth': future_prices_smooth})

# Enregistrer les données
df.to_csv('historical_fair_price_kas.csv', index=False)
future_df.to_csv('future_prices_kas.csv', index=False)

# Charger les données historiques et futures
historical_df = pd.read_csv('historical_fair_price_kas.csv')
future_df = pd.read_csv('future_prices_kas.csv')

# Convertir les colonnes de date en datetime
historical_df['date'] = pd.to_datetime(historical_df['date'])
future_df['date'] = pd.to_datetime(future_df['date'])

# Combiner les données
combined_df = pd.concat([historical_df, future_df], ignore_index=True)

# Visualiser
plt.figure(figsize=(14, 7))
plt.plot(combined_df['date'], combined_df['close'], label='Actual Price')
plt.plot(combined_df['date'], combined_df['historical_fair_price'], label='Historical Fair Price')
plt.plot(combined_df['date'], combined_df['historical_fair_price_smooth'], label='Smoothed Historical Fair Price')
plt.plot(future_df['date'], future_df['predicted_price'], label='Predicted Future Price', linestyle='--', color='red')
plt.yscale('log')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()