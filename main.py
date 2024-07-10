import pandas as pd
import streamlit as st
from plotly import graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")

# Charger le fichier CSV
csv_file = 'kas_d.csv'  # Remplacez par le chemin de votre fichier CSV
df = pd.read_csv(csv_file)
df['date'] = pd.to_datetime(df['date'])

instrument = "Kaspa (KAS)"  # Remplacez par l'instrument souhaité si nécessaire

dashboard = st.sidebar.selectbox(
    label='Select dashboard',
    options=['Rainbow chart', 'Risk Visualization']
)

# Fonction pour le Rainbow Chart
def plot_rainbow_chart(df, instrument):
    st.markdown(f"<h2 style='text-align: center;'>{instrument} Rainbow Chart</h2>", unsafe_allow_html=True)
    pct_change = st.sidebar.slider('Select increase/decrease in % for prediction:', min_value=-99, max_value=500, value=0)
    colors = ['blue','green','yellow', 'orange', 'red' ]

    if instrument == "Kaspa (KAS)":
        genesis_date = datetime(2021, 11, 7)
        start_date = '2022-01-04'
        future_days = np.arange(1, 365*2)

    df['date'] = pd.to_datetime(df['date'])
    max_date_with_close = df.dropna(subset=['close'])['date'].max()
    df = df[df["date"] <= max_date_with_close]

    start_date_for_slider = df['date'].iloc[99].date()
    end_date_for_slider = df['date'].iloc[-1].date()

    cut_off_date = st.sidebar.slider(
        "Select the cut-off date:",
        value=end_date_for_slider,
        min_value=start_date_for_slider,
        max_value=end_date_for_slider,
    )

    df['days_from_genesis'] = (df['date'] - genesis_date).dt.days
    df = df[df['days_from_genesis'] >= 0]

    df['log_close'] = np.log(df['close'])
    df['log_days_from_genesis'] = np.log(df['days_from_genesis'])

    X = df[['log_days_from_genesis']]
    y = df['log_close']
    model = LinearRegression().fit(X, y)
    df['predicted_log_close'] = model.predict(X)

    cut_off_date = pd.to_datetime(cut_off_date)
    start_date = pd.to_datetime(start_date)

    df_filtered_for_fit = df[df['date'] <= cut_off_date]

    X_fit = df_filtered_for_fit[['log_days_from_genesis']]
    y_fit = df_filtered_for_fit['log_close']
    model = LinearRegression().fit(X_fit, y_fit)

    df_after_start = df[df['date'] >= start_date]
    df['predicted_log_close'] = model.predict(df[['log_days_from_genesis']])
    df_after_start['residuals'] = df_after_start['log_close'] - df_after_start['predicted_log_close']

    highest_residual_index = df_after_start['residuals'].idxmax()
    lowest_residual_index = df_after_start['residuals'].idxmin()

    slope = model.coef_[0]
    intercept_high = df.loc[highest_residual_index, 'log_close'] - (slope * df.loc[highest_residual_index, 'log_days_from_genesis'])
    intercept_low = df.loc[lowest_residual_index, 'log_close'] - (slope * df.loc[lowest_residual_index, 'log_days_from_genesis'])

    slope_increase_percentage = pct_change
    adjusted_slope = slope * (1 + slope_increase_percentage / 100)

    last_day_from_genesis = np.log(df_filtered_for_fit['days_from_genesis'].max() + 1)
    intercept_high_adjusted = df.loc[highest_residual_index, 'log_close'] - (adjusted_slope * last_day_from_genesis)
    intercept_low_adjusted = df.loc[lowest_residual_index, 'log_close'] - (adjusted_slope * last_day_from_genesis)

    future_log_days_from_genesis = np.log(df_filtered_for_fit['days_from_genesis'].max() + 1 + future_days)
    future_days_from_genesis = np.exp(future_log_days_from_genesis)

    future_dates = [genesis_date + timedelta(days=int(day)) for day in future_days_from_genesis]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=np.exp(df['log_close']), mode='lines', name='Log of Close Prices', marker=dict(color='lightgray')))
    num_bands = 3

    intercepts_original = []

    for i in range(num_bands + 2):
        intercept_band = intercept_low + i * (intercept_high - intercept_low) / (num_bands + 1)
        y_values = slope * df_filtered_for_fit['log_days_from_genesis'] + intercept_band
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(x=df_filtered_for_fit['date'], y=np.exp(y_values), mode='lines', line=dict(color=color, dash='solid')))
    for i in range(num_bands + 2):
        intercept_band = intercept_low + i * (intercept_high - intercept_low) / (num_bands + 1)
        last_y_value = slope * last_day_from_genesis + intercept_band

        intercepts_original.append(last_y_value - adjusted_slope * last_day_from_genesis)

    for i, intercept_adjusted in enumerate(intercepts_original):
        y_values = adjusted_slope * future_log_days_from_genesis + intercept_adjusted
        color = colors[i % len(colors)]

        fig.add_trace(go.Scatter(x=future_dates, y=np.exp(y_values), mode='lines', name='Adjusted Bands' if i == 0 else "", line=dict(color=color, dash='dot')))

    fig.update_layout(
        height=800,
        width=1200,
        yaxis_type="log",
        xaxis=dict(showgrid=True, gridwidth=1, title='Date', tickangle=-45),
        yaxis_title='Close Price',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)
    expander = st.expander('About the model')
    expander.write('''
    This model calculates the linear regression on the log-log scaled data and then creates an envelope around the price action - for the date specified. It is a gross simplification, as the middle band **is not** the fair price as dictated by power law, but just middle between the lowest 
    and highest regression lines. Also both "support" and "resistance" lines have the same slope, a proper model should find a separate fit for the bottoms and tops. To view such a model check [this](https://hcburger.com/blog/poweroscillator/index.html) and to watch a more complex rainbow chart go  [here](https://www.blockchaincenter.net/en/bitcoin-rainbow-chart/).

    The aim of this graphic is to demonstrate how do such predictions change with more data.
    ''')

# Fonction pour la visualisation des risques
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

    # Calculate the `Risk Metric`
    df['MA'] = df['Value'].rolling(moving_average_days, min_periods=1).mean().dropna()
    df['Preavg'] = (np.log(df.Value) - np.log(df['MA'])) * df.index**diminishing_factor

    # Normalization to 0-1 range
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
    fig.add_trace(go.Scatter(x=df['date'], y=df['Value'], name='Price', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['avg'], name='Risk', line=dict(color='white')), secondary_y=True)

    # Ajout des zones colorées
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


# Affichage des graphiques en fonction de la sélection de l'utilisateur
if dashboard == 'Rainbow chart':
    plot_rainbow_chart(df, instrument)
elif dashboard == 'Risk Visualization':
    plot_risk_visualization(df, instrument)
