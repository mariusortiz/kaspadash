from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import pandas as pd

# Initialiser le client CoinGecko
cg = CoinGeckoAPI()

# Calculer la date de la veille (timestamp UNIX)
yesterday = datetime.now() - timedelta(1)
yesterday_str = yesterday.strftime('%d-%m-%Y')  # Format pour get_coin_history_by_id
yesterday_timestamp = int(yesterday.timestamp())

# Récupérer les données OHLC pour Kaspa
ohlc_data = cg.get_coin_ohlc_by_id(id='kaspa', vs_currency='usd', days='1')

# Trouver la bougie OHLC pour la veille
ohlc_yesterday = None
for ohlc in ohlc_data:
    if ohlc[0] >= yesterday_timestamp * 1000:  # Convertir en millisecondes pour la comparaison
        ohlc_yesterday = ohlc
        break

# Récupérer les données de volume pour Kaspa via get_coin_history_by_id
history_data = cg.get_coin_history_by_id(id='kaspa', date=yesterday_str)
market_data = history_data.get('market_data', {})
volume_usd = market_data.get('total_volume', {}).get('usd', None)

# Calculer le volume en KAS si le prix de clôture et le volume en USD sont disponibles
if ohlc_yesterday and volume_usd:
    timestamp, open_, high, low, close = ohlc_yesterday[:5]
    volume_native = volume_usd / close / 1_000_000  # Convertir en millions de KAS
    volume_str = f"{volume_native:.1f} m"  # Formater avec une décimale et ajouter 'm'

    # Convertir le timestamp en date
    date_str = datetime.utcfromtimestamp(timestamp / 1000).strftime('%m/%d/%Y')

    # Créer les données pour le CSV
    kaspa_data = {
        "date": date_str,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume_str  # Volume en millions de KAS avec suffixe 'm'
    }

    # Gestion du CSV
    csv_file = 'kas_d.csv'
    try:
        existing_df = pd.read_csv(csv_file)
        if date_str in existing_df['date'].values:
            print(f"Les données pour {date_str} existent déjà dans le fichier CSV. Pas de nouvelle entrée ajoutée.")
        else:
            # Ajouter les nouvelles données
            updated_df = pd.concat([existing_df, pd.DataFrame([kaspa_data])], ignore_index=True)
            # Sauvegarder le fichier mis à jour
            updated_df.to_csv(csv_file, index=False)
            print(f"Les données de Kaspa pour {date_str} ont été enregistrées dans 'kas_d.csv'.")
    except FileNotFoundError:
        # Si le fichier n'existe pas, créer un nouveau fichier CSV
        df = pd.DataFrame([kaspa_data])
        df.to_csv(csv_file, index=False)
        print(f"Fichier CSV créé et les données de Kaspa pour {date_str} ont été enregistrées.")
else:
    print("Impossible de récupérer toutes les données nécessaires pour la veille.")
