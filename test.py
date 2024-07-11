import pandas as pd

# Charger le fichier CSV
rainbow_df = pd.read_csv('rainbow_chart_data.csv')

# Afficher les premières lignes du dataframe pour vérifier les données
print(rainbow_df.head(10))
