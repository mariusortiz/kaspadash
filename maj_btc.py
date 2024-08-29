from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import pandas as pd

# Initialiser le client CoinGecko
cg = CoinGeckoAPI()

# Calculer la date de la veille (timestamp UNIX)
yesterday = datetime.now() - timedelta(1)
yesterday_str = yesterday.strftime('%d-%m-%Y')  # Format pour get_coin_history_by_id
yesterday_timestamp = int(yesterday.timestamp())

# Récupérer les données OHLC pour Bitcoin
ohlc_data = cg.get_coin_ohlc_by_id(id='bitcoin', vs_currency='usd', days='1')

# Trouver la bougie OHLC pour la veille
ohlc_yesterday = None
for ohlc in ohlc_data:
    if ohlc[0] >= yesterday_timestamp * 1000:  # Convertir en millisecondes pour la comparaison
        ohlc_yesterday = ohlc
        break

# Récupérer les données de volume pour Bitcoin via get_coin_history_by_id
history_data = cg.get_coin_history_by_id(id='bitcoin', date=yesterday_str)
market_data = history_data.get('market_data', {})
volume_btc = market_data.get('total_volume', {}).get('btc', None)

# Calculer le volume si le prix de clôture et le volume en BTC sont disponibles
if ohlc_yesterday and volume_btc:
    timestamp, open_, high, low, close = ohlc_yesterday[:5]

    # Convertir le timestamp en date
    date_str = datetime.utcfromtimestamp(timestamp / 1000).strftime('%m/%d/%Y')

    # Créer les données pour le CSV
    btc_data = {
        "date": date_str,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": f"{volume_btc:.1f} m"  # Volume en BTC
    }

    # Gestion du CSV
    csv_file = 'btc_d.csv'
    try:
        existing_df = pd.read_csv(csv_file)
        if date_str in existing_df['date'].values:
            print(f"Les données pour {date_str} existent déjà dans le fichier CSV. Pas de nouvelle entrée ajoutée.")
        else:
            # Ajouter les nouvelles données
            updated_df = pd.concat([existing_df, pd.DataFrame([btc_data])], ignore_index=True)
            # Sauvegarder le fichier mis à jour
            updated_df.to_csv(csv_file, index=False)
            print(f"Les données de Bitcoin pour {date_str} ont été enregistrées dans 'btc_d.csv'.")
    except FileNotFoundError:
        # Si le fichier n'existe pas, créer un nouveau fichier CSV
        df = pd.DataFrame([btc_data])
        df.to_csv(csv_file, index=False)
        print(f"Fichier CSV créé et les données de Bitcoin pour {date_str} ont été enregistrées.")
else:
    print("Impossible de récupérer toutes les données nécessaires pour la veille.")
