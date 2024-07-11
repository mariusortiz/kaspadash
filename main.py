import pandas as pd
import streamlit as st
from plotly import graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor

def exponential_smoothing(series, alpha):
    result = [series[0]]  # première valeur est identique à la série
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result

# Charger les données depuis les CSV générés
historical_fair_price_df = pd.read_csv('historical_fair_price.csv')
predicted_prices_df = pd.read_csv('predicted_prices.csv')


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
    model = RANSACRegressor().fit(X, y)
    df['predicted_log_close'] = model.predict(X)

    cut_off_date = pd.to_datetime(cut_off_date)
    start_date = pd.to_datetime(start_date)

    df_filtered_for_fit = df[df['date'] <= cut_off_date]

    X_fit = df_filtered_for_fit[['log_days_from_genesis']]
    y_fit = df_filtered_for_fit['log_close']
    model = RANSACRegressor().fit(X_fit, y_fit)

    df_after_start = df[df['date'] >= start_date]
    df['predicted_log_close'] = model.predict(df[['log_days_from_genesis']])
    df_after_start['residuals'] = df_after_start['log_close'] - df_after_start['predicted_log_close']

    highest_residual_index = df_after_start['residuals'].idxmax()
    lowest_residual_index = df_after_start['residuals'].idxmin()

    slope = model.estimator_.coef_[0]
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

def plot_past_power_law(df, instrument):
    st.markdown(f"<h2 style='text-align: center;'>{instrument} Historical Power Law Predictions</h2>", unsafe_allow_html=True)
    
    chart_type = st.sidebar.select_slider(
        'Select scale type',
        options=['Linear', 'Logarithmic'],
        value="Linear"
    )
    
    max_date_with_close = df.dropna(subset=['close'])['date'].max()
    df = df[df["date"] <= max_date_with_close]
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'Actual vs Predicted Prices - {instrument}', 'Percentage Difference between Actual and Historical Predicted Prices'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Actual Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['historical_fair_price'], mode='lines', name='Predicted Next Day Price', line=dict(color='cyan')), row=1, col=1)
    
    fig.add_annotation(
        text="KASPING.STREAMLIT.APP",
        align='left',
        opacity=0.4,
        font=dict(color="red", size=35),
        xref='paper',
        yref='paper',
        x=0.5,
        y=0.5,
        showarrow=False
    )
    
    differences = 100 * (df['close'] - df['historical_fair_price']) / df['historical_fair_price']
    fig.add_trace(go.Scatter(x=df['date'], y=differences, mode='lines', name='Difference (%)'), row=2, col=1)
    fig.add_hline(y=0, line=dict(dash='dash', color='red'), row=2, col=1)

    fig.update_layout(height=800, width=1000, showlegend=True)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Difference (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    if chart_type == "Linear":
        fig.update_layout(yaxis_title='Price', xaxis_rangeslider_visible=False)
    elif chart_type == "Logarithmic":
        fig.update_layout(yaxis=dict(type='log', title='Price'), xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)
    expander = st.expander('About the chart')
    expander.write('''
    You might find it surprising to see the predicted value fluctuate. Typically, power law charts depict the fair price as a constant, straight line (on log-log charts) because they are curve-fitted on the past data for the best fit.

    However, this doesn't reveal past predictions, which is crucial for assessing the reliability of these forecasts.

    This chart is designed differently. It shows predictions as they would have been made using all available data at each point in the past. The goal is to demonstrate the degree to which power law predictions can vary, giving you insight into their consistency.
    ''')

def plot_future_power_law(df, historical_fair_price_df, predicted_prices_df):
    days_from_today = st.sidebar.slider('Select number of days from today for prediction:', 
                                        min_value=1, 
                                        max_value=3650,  # Changer à 10 ans
                                        value=30)
    st.markdown(f"<h2 style='text-align: center;'>Kaspa (KAS) Power Law Predictions</h2>", unsafe_allow_html=True)

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
    fig.add_trace(go.Scatter(x=df_to_plot['date'], y=df_to_plot['historical_fair_price'], mode='lines', name='Historical Fair Price', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df_to_plot['date'], y=df_to_plot['predicted_price'], mode='lines', name='Predicted Future Price', line=dict(color='red', dash='dash')))

    fig.add_vline(x=future_date.timestamp() * 1000, line=dict(color="purple", dash="dash"), annotation_text=f"Predicted price: {predicted_price_on_future_date:.5f}")
    fig.add_trace(go.Scatter(x=[future_date], y=[predicted_price_on_future_date], mode='markers', marker=dict(color='red', size=10), name='Predicted Price'))

    if chart_type == "Linear":
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
    elif chart_type == "Logarithmic":
        fig.update_layout(xaxis_title='Date', yaxis=dict(type='log', title='Price'), xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)
    expander = st.expander('About the chart')
    expander.write('''
    You might find it surprising to see the predicted value fluctuate. Typically, power law charts depict the fair price as a constant, straight line (on log-log charts) because they are curve-fitted on the past data for the best fit.

    However, this doesn't reveal past predictions, which is crucial for assessing the reliability of these forecasts.

    This chart is designed differently. It shows predictions as they would have been made using all available data at each point in the past. The goal is to demonstrate the degree to which power law predictions can vary, giving you insight into their consistency.
    ''')

# Charger les données
df = pd.read_csv('kas_d.csv')
historical_fair_price_df = pd.read_csv('historical_fair_price.csv')
predicted_prices_df = pd.read_csv('future_prices.csv')

# Appeler la fonction pour tracer les graphiques
plot_future_power_law(df, historical_fair_price_df, predicted_prices_df)

def main():
    st.set_page_config(layout="wide")

    # Charger le fichier CSV du prix actuel
    csv_file = 'kas_d.csv'
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])

    instrument = "Kaspa (KAS)"
    df['days_from_genesis'] = (df['date'] - df['date'].min()).dt.days

    dashboard = st.sidebar.selectbox(
        label='Select dashboard',
        options=['Rainbow chart', 'Risk Visualization', 'Past Power Law', 'Future Power Law']
    )

    if dashboard == 'Future Power Law':
        plot_future_power_law(df, historical_fair_price_df, predicted_prices_df)

if __name__ == "__main__":
    main()