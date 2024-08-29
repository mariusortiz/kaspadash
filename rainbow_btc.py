import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Charger les données historiques
df = pd.read_csv('btc_d.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Calculer ΔGB (jours depuis le Genesis Block)
genesis_date = pd.Timestamp('2009-01-03')
df['days_since_genesis'] = (df['date'] - genesis_date).dt.days

# Calculer les prix de support, de résistance et les prix justes selon la loi de puissance
df['support_price'] = 10**(-17.00661888) * (df['days_since_genesis']**5.82)
df['resistance_price'] = df['support_price'] * 2  # La résistance est le double du support
df['fair_price'] = df['support_price'] * 1  # Prix juste, qui est le prix de support pour le BTC

# Créer un DataFrame pour le Rainbow Chart
rainbow_df = pd.DataFrame()
rainbow_df['date'] = df['date']

# Liste des couleurs et leurs multiplicateurs
multipliers = {
    'purple': 0.42,  # Bad news
    'blue': np.exp(-np.log(2) / 2),  # Buy
    'light_blue': np.exp(-np.log(2) / 4),  # Cheap
    'green': 1,  # Fair Price
    'yellow': np.exp(np.log(2) / 4),  # Pricey
    'orange': np.exp(np.log(2) / 2),  # Expensive
    'red': 2  # Sell
}

# Calcule les prix pour chaque bande en utilisant le fair_price comme référence
for color, multiplier in multipliers.items():
    if color == 'green':  # La bande verte (fair_price) est déjà calculée
        rainbow_df[f'{color}_price'] = df['fair_price']
    else:
        rainbow_df[f'{color}_price'] = df['fair_price'] * multiplier

# Enregistrer dans un fichier CSV (immuable)
rainbow_df.to_csv('rainbow_chart_data_btc.csv', index=False)

# Plot the chart
import plotly.graph_objs as go
import streamlit as st

def plot_rainbow_chart(df, rainbow_df):
    fig = go.Figure()

    # Ajouter les bandes colorées
    colors = ['purple', 'blue', 'light_blue', 'green', 'yellow', 'orange', 'red']
    for color in colors:
        fig.add_trace(go.Scatter(x=rainbow_df['date'], y=rainbow_df[f'{color}_price'], mode='lines', name=color.capitalize(), line=dict(color=color)))

    # Ajouter les prix réels
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Actual Price', line=dict(color='cyan')))

    fig.update_layout(title="BTC Rainbow Chart", xaxis_title="Date", yaxis_title="Price", yaxis_type="log")
    st.plotly_chart(fig, use_container_width=True)

# Appel de la fonction pour générer le graphique
plot_rainbow_chart(df, rainbow_df)
