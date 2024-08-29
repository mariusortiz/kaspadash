import pandas as pd
import streamlit as st
from plotly import graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor
from PIL import Image
import streamlit.components.v1 as components

def exponential_smoothing(series, alpha):
    result = [series[0]]  # première valeur est identique à la série
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result


def plot_sma_chart(df, instrument):
    st.markdown(f"<h2 style='text-align: center;'>{instrument} SMA Crossover Chart</h2>", unsafe_allow_html=True)

    # Calculer les moyennes mobiles simples (SMA)
    df['SMA_66'] = df['close'].rolling(window=66).mean()
    df['SMA_85'] = df['close'].rolling(window=85).mean()

    # Identifier les points de croisement
    df['crossover'] = np.where((df['SMA_85'] > df['SMA_66']) & (df['SMA_85'].shift(1) <= df['SMA_66'].shift(1)), 1, 0)
    crossover_dates = df[df['crossover'] == 1]['date']

    fig = go.Figure()

    # Tracer le prix actuel
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        mode='lines',
        name='Price Data',
        line=dict(color='cyan')
    ))

    # Tracer la SMA 66
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['SMA_66'],
        mode='lines',
        name='66DMA',
        line=dict(color='purple')
    ))

    # Tracer la SMA 85
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['SMA_85'],
        mode='lines',
        name='85DMA',
        line=dict(color='yellow')
    ))

    # Ajouter les lignes verticales pour les croisements
    for date in crossover_dates:
        fig.add_vline(x=date.timestamp() * 1000, line=dict(color="red", dash="dot"))

    # Ajuster les ordonnées
    fig.update_yaxes(
        tickvals=[0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        range=[0, 0.30],  # Fixer la plage d'ordonnée de 0 à 0.30
        showgrid=True,
        gridwidth=1,
        title='Price'
    )

    fig.update_layout(
        height=800,
        width=1200,
        xaxis=dict(showgrid=True, gridwidth=1, title='Date', tickangle=-45),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True)
    expander = st.expander('Explications')
    expander.write('''
    #### SMA Crossover Chart

    Le graphique **SMA Crossover Chart** est basé sur l'observation que lorsque la moyenne mobile simple (SMA) sur 85 jours croise la moyenne mobile simple sur 66 jours, le prix de Kaspa (KAS) tend à connaître une forte hausse peu de temps après. Ce type d'indicateur est inspiré par des analyses similaires effectuées sur Bitcoin, notamment l'indicateur "Pi Cycle Top".

    #### Calculs effectués :
    1. **Moyenne Mobile Simple (SMA)** : Nous calculons deux moyennes mobiles simples, l'une sur 66 jours et l'autre sur 85 jours, à partir des données de prix historiques. La SMA sur 66 jours réagit plus rapidement aux changements de prix, tandis que la SMA sur 85 jours est plus lissée et réactive aux tendances à plus long terme.
    2. **Crossover (Croisement)** : Nous identifions les points où la SMA sur 66 jours croise la SMA sur 85 jours. Ces croisements sont particulièrement intéressants car ils peuvent signaler des changements significatifs dans la direction du prix. En particulier, un croisement où la SMA sur 66 jours passe au-dessus de la SMA sur 85 jours est souvent vu comme un signal haussier.
    3. **Visualisation des croisements** : Les dates où ces croisements se produisent sont mises en évidence sur le graphique par des lignes verticales rouges. Cela permet de visualiser facilement les moments où ces signaux se sont produits et d'observer leur impact potentiel sur le prix par la suite.
    4. **Affichage des prix historiques** : En plus des SMA, nous affichons le prix historique réel de Kaspa pour fournir un contexte visuel clair aux croisements des moyennes mobiles.

    #### Utilité et pertinence :
    Le SMA Crossover Chart est particulièrement utile pour les traders et les investisseurs qui cherchent à identifier des points d'entrée ou de sortie basés sur les signaux de croisement des moyennes mobiles. Les croisements des SMA sur 66 et 85 jours offrent un indicateur intermédiaire qui est sensible aux changements de tendance à moyen terme. En utilisant ce graphique, les utilisateurs peuvent repérer les signaux haussiers ou baissiers potentiels et ajuster leurs stratégies d'investissement en conséquence.
''')


def plot_rainbow_chart(df, rainbow_df, instrument):
    st.markdown(f"<h2 style='text-align: center;'>{instrument} Rainbow Chart</h2>", unsafe_allow_html=True)
    
    df['date'] = pd.to_datetime(df['date'])
    rainbow_df['date'] = pd.to_datetime(rainbow_df['date'])

    # Mapping des couleurs aux descriptions
    color_map = {
        'red': ('#FF0000', 'Sell'),
        'orange': ('#FFA500', 'Expensive'),
        'yellow': ('#FFFF00', 'Pricey'),
        'green': ('#008000', 'Fair Price'),
        'light_blue': ('#ADD8E6', 'Cheap'),
        'dark_blue': ('#00008B', 'Buy'),
        'purple': ('#800080', 'Bad news')
    }

    fig = go.Figure()

    # Tracer les bandes du Rainbow Chart avec les nouvelles légendes
    for color, (color_hex, label) in color_map.items():
        color_data = rainbow_df[rainbow_df['color'] == color]
        fig.add_trace(go.Scatter(
            x=color_data['date'],
            y=color_data['price'],
            mode='lines',
            name=label,
            line=dict(color=color_hex)
        ))

    # Tracer le prix actuel
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        mode='lines',
        name='Actual Price',
        line=dict(color='cyan')
    ))

    fig.update_layout(
        height=800,
        width=1200,
        yaxis_type="log",
        xaxis=dict(showgrid=True, gridwidth=1, title='Date', tickangle=-45),
        yaxis_title='Price',
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)
    expander = st.expander('Explications')
    expander.write('''
    #### Rainbow Chart

    Le graphique Rainbow Chart visualise différentes bandes de prix pour Kaspa (KAS) ou Bitcoin (BTC) en plus du prix réel. Chaque bande de couleur représente une plage de prix distincte, basée sur une loi de puissance appliquée au temps écoulé depuis le bloc de genèse de Kaspa. Cela offre un moyen visuel de comprendre la dynamique des prix au fil du temps.

    #### Calculs effectués :
    1. **Loi de Puissance (Power Law)** : Nous utilisons une loi de puissance pour modéliser le prix en fonction du temps écoulé depuis le bloc de genèse. La formule est basée sur l'équation : `Prix = Coefficient * (Jours depuis le bloc de genèse)^Exposant`, où le coefficient et l'exposant sont dérivés de l'analyse des données historiques.
    2. **Multiplicateurs pour les bandes de couleurs** : Chaque bande de couleur est générée en appliquant un multiplicateur spécifique au prix "juste" calculé par la loi de puissance. Par exemple, la bande verte représente le prix "juste", tandis que les autres bandes (bleu, jaune, orange, rouge, etc.) représentent des niveaux de prix différents basés sur des multiples de cette valeur "juste".
    3. **Prévision des prix futurs** : Le modèle est utilisé non seulement pour tracer les prix historiques, mais aussi pour extrapoler les prix futurs sur plusieurs mois en fonction du même modèle de loi de puissance.
    4. **Visualisation continue** : Les valeurs sont interpolées pour couvrir toute la plage de dates, ce qui permet une visualisation continue et lisse des bandes de couleurs sur le graphique.

    #### Utilité et pertinence :
    Le Rainbow Chart est particulièrement pertinent car il permet aux investisseurs de visualiser facilement les zones de support et de résistance basées sur un modèle de loi de puissance. Les différentes bandes colorées aident à identifier les niveaux de prix potentiels où le marché peut trouver un soutien ou une résistance. Cette visualisation est très utile pour les traders et les analystes techniques qui cherchent à identifier les tendances de prix et à prendre des décisions éclairées basées sur ces niveaux critiques.
''')


def plot_future_power_law(df, instrument, historical_fair_price_df, predicted_prices_df):
    try:
        days_from_today = st.sidebar.slider('Select number of days from today for prediction:', 
                                            min_value=1, 
                                            max_value=365,  # Changer à 10 ans
                                            value=30)
        st.markdown(f"<h2 style='text-align: center;'>{instrument} Power Law Predictions</h2>", unsafe_allow_html=True)

        chart_type = st.sidebar.select_slider(
            'Select scale type',
            options=['Linear', 'Logarithmic'],
            value="Linear"
        )

        # Assurez-vous que les dates sont dans le bon format
        historical_fair_price_df['date'] = pd.to_datetime(historical_fair_price_df['date'])
        predicted_prices_df['date'] = pd.to_datetime(predicted_prices_df['date'])

        # Merge des données pour inclure le prix historique et les prédictions futures
        df = df.merge(historical_fair_price_df, on='date', how='left', suffixes=('', '_fair_price'))
        df = df.merge(predicted_prices_df, on='date', how='left', suffixes=('', '_predicted'))

        # Calculer les dates futures
        last_date = df['date'].max()
        future_date = last_date + timedelta(days=days_from_today)

        future_price_row = predicted_prices_df[predicted_prices_df['date'] == future_date]
        if future_price_row.empty:
            st.error(f"No data available for the selected future date: {future_date.strftime('%Y-%m-%d')}")
            return

        predicted_price_on_future_date = future_price_row['predicted_price'].values[0]
        today_price = df.dropna(subset=['close'])['close'].values[-1]

        st.markdown(f"<h4 style='text-align: center;'>Predicted price {days_from_today} days from the last available date ({last_date.strftime('%Y-%m-%d')}) is: ${predicted_price_on_future_date:.5f},  {((predicted_price_on_future_date-today_price)/today_price)*100:.0f}% difference</h4>", unsafe_allow_html=True)

        fig = go.Figure()
        df_to_plot = df[df['date'] <= future_date]

        fig.add_trace(go.Scatter(x=df_to_plot['date'], y=df_to_plot['close'], mode='lines', name='Actual Price'))
        fig.add_trace(go.Scatter(x=df_to_plot['date'], y=df_to_plot['predicted_price'], mode='lines', name='Predicted Future Price', line=dict(color='red', dash='dot')))
        fig.add_trace(go.Scatter(x=df_to_plot['date'], y=df_to_plot['historical_fair_price_smooth'], mode='lines', name='Smoothed Historical Fair Price', line=dict(color='orange')))

        fig.add_vline(x=future_date.timestamp() * 1000, line=dict(color="purple", dash="dash"), annotation_text=f"Predicted price: {predicted_price_on_future_date:.5f}")
        fig.add_trace(go.Scatter(x=[future_date], y=[predicted_price_on_future_date], mode='markers', marker=dict(color='red', size=10), name='Predicted Price'))

        if chart_type == "Linear":
            fig.update_layout(xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
        elif chart_type == "Logarithmic":
            fig.update_layout(xaxis_title='Date', yaxis=dict(type='log', title='Price'), xaxis_rangeslider_visible=False)

        st.plotly_chart(fig, use_container_width=True)
        expander = st.expander('Explications')
        expander.write('''
        #### Future Power Law

        Le graphique Future Power Law utilise une loi de puissance pour prédire le prix futur de Kaspa (KAS) ou Bitcoin (BTC), en se basant sur des données historiques et des modèles de régression.

        #### Calculs effectués :
        1. **Séries temporelles historiques** : Nous utilisons les données de prix historiques pour ajuster un modèle de loi de puissance. La loi de puissance est une relation mathématique où une quantité varie comme une puissance d'une autre quantité.
        2. **Régression RANSAC** : Nous appliquons l'algorithme de régression RANSAC pour ajuster un modèle logarithmique aux données historiques, en identifiant une "fair price" historique.
        3. **Lissage exponentiel** : Nous appliquons un lissage exponentiel aux prix historiques pour obtenir une courbe plus régulière et prévisible.
        4. **Interpolation et extrapolation** : En utilisant les modèles ajustés, nous interpolons les tendances historiques et les extrapolons pour prédire les prix futurs. Cette prédiction est ensuite lissée pour réduire la volatilité et améliorer la précision.

        #### Utilité et pertinence :
        La visualisation du risque est essentielle pour évaluer le potentiel de perte ou de gain associé à une cryptomonnaie. En montrant comment le prix actuel se compare à sa tendance historique, cette visualisation aide les investisseurs à identifier les périodes de surachat ou de survente. Les zones de risque colorées fournissent une représentation visuelle claire des niveaux de risque, aidant ainsi à la prise de décision stratégique en matière d'investissement et de gestion des risques.
    ''')

    except Exception as e:
        st.error(f"An error occurred: {e}")

def plot_risk_visualization(df, instrument):
    st.markdown(f"<h2 style='text-align: center;'>{instrument} Risk Visualization</h2>", unsafe_allow_html=True)
    
    chart_type = st.sidebar.select_slider(
        'Select scale type',
        options=['Linear', 'Logarithmic'],
        value="Logarithmic"
    )
    df['Value'] = df['close']
    df = df[df['Value'] > 0]
    diminishing_factor = 0.395
    moving_average_days = 365
    df['MA'] = df['Value'].rolling(moving_average_days, min_periods=1).mean().dropna()
    df['Preavg'] = (np.log(df.Value) - np.log(df['MA'])) * df.index**diminishing_factor
    df['avg'] = (df['Preavg'] - df['Preavg'].cummin()) / (df['Preavg'].cummax() - df['Preavg'].cummin())
    annotation_text = f"Updated: {df['date'].iloc[-1].strftime('%Y-%m-%d')} | Price: {round(df['Value'].iloc[-1], 5)} | Risk: {round(df['avg'].iloc[-1], 2)}"
    if chart_type == "Linear":
        fig = go.Figure(data=go.Scatter(x=df['date'], y=df['Value'], mode='markers', marker=dict(size=8, color=df['avg'], colorscale='Jet', showscale=True)))
        fig.update_yaxes(title='Price ($USD)', showgrid=True)
        fig.update_layout(height=600, width=800, margin=dict(l=50, r=50, t=50, b=50))
    else:
        fig = go.Figure(data=go.Scatter(x=df['date'], y=df['Value'], mode='markers', marker=dict(size=8, color=df['avg'], colorscale='Jet', showscale=True)))
        fig.update_yaxes(title='Price ($USD)', showgrid=True, type='log')
        fig.update_layout(height=600, width=800, margin=dict(l=50, r=50, t=50, b=50))
    fig.update_layout(title=annotation_text)
    st.plotly_chart(fig, use_container_width=True)
    # Deuxième graphique avec des zones de risque colorées
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=df['date'], y=df['Value'], name='Price', line=dict(color='cyan', width=3)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['avg'], name='Risk', line=dict(color='purple', width=2)), secondary_y=True)
    opacity = 0.2
    for i in range(5, 0, -1):
        opacity += 0.05
        fig.add_hrect(y0=i*0.1, y1=((i-1)*0.1), line_width=0, fillcolor='green', opacity=opacity, secondary_y=True)
    for i in range(6, 10):
        opacity += 0.1
        fig.add_hrect(y0=i*0.1, y1=((i+1)*0.1), line_width=0, fillcolor='red', opacity=opacity, secondary_y=True)
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Price ($USD)', type='log', showgrid=False)
    fig.update_yaxes(title='Risk', type='linear', secondary_y=True, showgrid=True, tick0=0.0, dtick=0.1, range=[0, 1])
    fig.update_layout(template='plotly_dark', title={'text': annotation_text, 'y': 0.9, 'x': 0.5})

    st.plotly_chart(fig, use_container_width=True)
    expander = st.expander('Explications')
    expander.write('''
        #### Risk Visualization

        La visualisation des risques utilise des méthodes statistiques pour mesurer et afficher le risque associé aux fluctuations des prix de Kaspa (KAS) ou Bitcoin (BTC).

        #### Calculs effectués :
        1. **Moyenne mobile** : Nous calculons la moyenne mobile des prix de clôture sur une période de 365 jours pour lisser les fluctuations à court terme et mettre en évidence les tendances à long terme.
        2. **Pré-moyenne logarithmique** : Nous transformons les prix en utilisant une échelle logarithmique et ajustons les valeurs en fonction de l'indice des données pour tenir compte de la variabilité temporelle.
        3. **Indice de risque** : Nous calculons l'indice de risque en mesurant la différence entre la pré-moyenne logarithmique et ses valeurs minimales et maximales cumulatives. Cela donne une mesure normalisée du risque qui varie entre 0 et 1.
        4. **Visualisation colorée** : Nous utilisons une échelle de couleurs pour représenter l'indice de risque, avec des couleurs allant du vert (faible risque) au rouge (risque élevé), et ajoutons des zones de risque colorées pour une visualisation plus intuitive.

        #### Utilité et pertinence :
        La visualisation des risques est essentielle pour les investisseurs qui cherchent à comprendre la volatilité et le risque associés à une cryptomonnaie. En offrant une mesure visuelle et normalisée du risque, ce graphique permet aux utilisateurs d'identifier les périodes de haute et basse volatilité, de prendre des décisions d'investissement éclairées et de gérer leurs portefeuilles de manière plus efficace. La visualisation colorée facilite également la compréhension rapide des niveaux de risque actuels et historiques.
    ''')

# Charger les données
def load_data(currency):
    kas_d_csv = f'{currency}_d.csv'
    rainbow_chart_csv = f'rainbow_chart_data_{currency}.csv'
    historical_fair_price_csv = f'historical_fair_price_{currency}.csv'
    predicted_prices_csv = f'future_prices_{currency}.csv'

    df = pd.read_csv(kas_d_csv)
    rainbow_df = pd.read_csv(rainbow_chart_csv)
    historical_fair_price_df = pd.read_csv(historical_fair_price_csv)
    predicted_prices_df = pd.read_csv(predicted_prices_csv)

    df['date'] = pd.to_datetime(df['date'])
    df['days_from_genesis'] = (df['date'] - df['date'].min()).dt.days
    
    return df, rainbow_df, historical_fair_price_df, predicted_prices_df

def main():
    st.set_page_config(layout="wide")

    # Charger les images des logos
    kaspa_logo = Image.open("images/kaspa_logo.png")
    bitcoin_logo = Image.open("images/bitcoin_logo.png")

    # Utiliser HTML/CSS pour créer des boutons personnalisés avec logos
    st.sidebar.markdown("### Choix de la monnaie")
    
    kaspa_html = f"""
    <style>
    .button {{
        background-color: transparent;
        border: none;
        text-align: center;
        padding: 5px;
        display: inline-block;
    }}
    .button img {{
        width: 30px;
        height: 30px;
    }}
    .button span {{
        color: white;
        font-size: 18px;
        margin-left: 10px;
        vertical-align: middle;
    }}
    </style>
    <button class="button" onclick="window.location.href='?selected_currency=kas'">
        <img src="data:image/png;base64,{st.image(kaspa_logo, use_column_width=False).data}"/>
        <span>Kaspa (KAS)</span>
    </button>
    """

    bitcoin_html = f"""
    <style>
    .button {{
        background-color: transparent;
        border: none;
        text-align: center;
        padding: 5px;
        display: inline-block;
    }}
    .button img {{
        width: 30px;
        height: 30px;
    }}
    .button span {{
        color: white;
        font-size: 18px;
        margin-left: 10px;
        vertical-align: middle;
    }}
    </style>
    <button class="button" onclick="window.location.href='?selected_currency=btc'">
        <img src="data:image/png;base64,{st.image(bitcoin_logo, use_column_width=False).data}"/>
        <span>Bitcoin (BTC)</span>
    </button>
    """

    # Affichage des boutons avec les logos
    col1, col2 = st.sidebar.columns(2)
    with col1:
        components.html(kaspa_html, height=60)
    with col2:
        components.html(bitcoin_html, height=60)

    # Vérification de la monnaie sélectionnée
    selected_currency = st.experimental_get_query_params().get('selected_currency', ['kas'])[0]

    st.sidebar.markdown("### Choix du dashboard")

    # Liste de dashboards avec des boutons stylisés
    dashboard = st.sidebar.selectbox(
        label="",
        options=['Rainbow Chart', 'Risk Visualization', 'Future Power Law', 'SMA Chart'],
        index=0,
        format_func=lambda x: f"🎯 {x}"  # Ajout d'une icône à chaque option pour la rendre plus attrayante
    )

    df, rainbow_df, historical_fair_price_df, predicted_prices_df = load_data(selected_currency)

    if dashboard == 'Rainbow Chart':
        plot_rainbow_chart(df, rainbow_df, selected_currency)
    elif dashboard == 'Risk Visualization':
        plot_risk_visualization(df, selected_currency)
    elif dashboard == 'Future Power Law':
        plot_future_power_law(df, selected_currency, historical_fair_price_df, predicted_prices_df)
    elif dashboard == 'SMA Chart':  
        plot_sma_chart(df, selected_currency)

if __name__ == "__main__":
    main()