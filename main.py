import pandas as pd
import streamlit as st
from plotly import graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
import numpy as np
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
import base64
import io
import os
st.set_page_config(layout="wide")


instrument = st.sidebar.selectbox(
    label='Select instrument',
    options=["Kaspa (KAS)", "Bitcoin (BTC)"]
)


st.warning("**Follow us on** [Twitter](https://twitter.com/KaspingApp)", icon="💸")  # Adjust font color and size

ENCRYPTION_PASSWORD =st.secrets["ENCRYPTION_PASSWORD"]

def un_encrypt_file(file_name):
# Read the encrypted file and extract the salt
    try:
        with open(fr"{file_name}", 'rb') as encrypted_file:
            file_content = encrypted_file.read()
        salt = file_content[:16]  # Extract the salt (assuming you used a 16-byte salt)
        encrypted_content = file_content[16:]
    
        # Re-create the KDF instance for decryption
        password = ENCRYPTION_PASSWORD.encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
    
        # Derive the key using the same password and the extracted salt
        key = base64.urlsafe_b64encode(kdf.derive(password))
    
        # Decrypt the content
        cipher_suite = Fernet(key)
        decrypted_content = cipher_suite.decrypt(encrypted_content)
    
        # Convert the decrypted content back to a DataFrame
        decrypted_df = pd.read_csv(io.StringIO(decrypted_content.decode()))
    
        # Display the head of the DataFrame
        return decrypted_df
    except:
        decrypted_df = pd.read_csv('data/kas_real_PL_extended.csv')
        return decrypted_df


if instrument == "Kaspa (KAS)":
    dashboard = st.sidebar.selectbox(
        label='Select dashboard',
        options=[
            'Past Power Law',
            'Future Power Law',
            'Risk Visualization',
            'Trend Predictor',
            'DCA Simulator - *** Coming Soon ***','Smart DCA Automation - *** Coming Soon ***'
        ])

    
    df = un_encrypt_file('data/kas_real_PL_extended.csv')

else:
    dashboard = st.sidebar.selectbox(
        label='Select dashboard',
        options=[
            'Past Power Law',
            'Future Power Law',
            'Risk Visualization',
            'Trend Predictor - *** Coming Soon ***,',
            'DCA Simulator - *** Coming Soon ***','Smart DCA Automation - *** Coming Soon ***'])
    df = un_encrypt_file('data/btc_real_PL_extended.csv')

if dashboard in ('DCA Simulator - *** Coming Soon ***', 'Smart DCA Automation - *** Coming Soon ***', "Trend Predictor - *** Coming Soon ***,"):
    st.title(f'Coming soon')

    

df['date'] = pd.to_datetime(df['date'])
max_date = df["date"].max()  # Get the last date in the DataFrame
max_date_with_close = df.dropna(subset=['close'])['date'].max()
    
    # Calculate the difference from today to max_date
days_difference = (max_date - datetime.today()).days
    # Slider for selecting the number of days from today for prediction
df = df.reset_index(drop=True)

if dashboard == 'Past Power Law':
    # Load in the data for the dash
    st.markdown(f"<h2 style='text-align: center;'>{instrument} Historical Power Law Predictions</h2>", unsafe_allow_html=True)


    chart_type = st.sidebar.select_slider(
        'Select scale type',
        options=['Linear', 'Logarithmic'],
        value = "Linear")

    df = df[df["date"] <= max_date_with_close]

    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'Actual vs Predicted Prices - {instrument}', 'Percentage Difference between Actual and  Historical Predicted Prices'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Actual Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['date'], y=df['predicted_next_day_price'], mode='lines', name='Predicted Next Day Price', line=dict(color='white')), row=1, col=1)
    fig.add_annotation(
        text="KASPING.STREAMLIT.APP",  # The watermark text
        align='left',
        opacity=0.2,  # Adjust opacity to make the watermark lighter
        font=dict(color="yellow", size=35),  # Adjust font color and size
        xref='paper',  # Position the watermark relative to the entire figure
        yref='paper',
        x=0.5,  # Centered horizontally
        y=0.5,  # Centered vertically
        showarrow=False)  # Do not show an arrow pointing to the text)
    differences = 100 * (df['close'] - df['predicted_next_day_price']) / df['predicted_next_day_price']
    fig.add_trace(go.Scatter(x=df['date'], y=differences, mode='lines', name='Difference (%)'), row=2, col=1)
    fig.add_hline(y=0, line=dict(dash='dash', color='red'), row=2, col=1)

    # Update layout
    fig.update_layout(height=800, width=1000,  showlegend=True)
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
















# Assuming your CSV has been loaded into `df`

if dashboard == 'Future Power Law':


    days_from_today = st.sidebar.slider('Select number of days from today for prediction:', 
                            min_value=1, 
                            max_value=days_difference, 
                            value=30)
    # Slider for selecting the number of days from today for prediction
    st.markdown(f"<h2 style='text-align: center;'>{instrument} Power Law Predictions</h2>", unsafe_allow_html=True)

    chart_type = st.sidebar.select_slider(
        'Select scale type',
        options=['Linear', 'Logarithmic'],
        value = "Linear")
    # Calculate the date for the specified number of days from today
    today = datetime.today()
    future_date = today + timedelta(days=(days_from_today-1))

    # Find the closest date in the dataframe to the future date
    closest_future_date = df[df['date'] >= future_date].iloc[0]['date']

    # Retrieve the predicted price for that date
    predicted_price_on_future_date = df[df['date'] == closest_future_date]['predicted_price'].values[0]
    today_price = df.dropna(subset=['close'])['close'].values[-1]


    # Display the predicted price
    #st.write(f"Predicted price {days_from_today} days from today ({future_date.strftime('%Y-%m-%d')}) is: ${predicted_price_on_future_date:.5f},  {((predicted_price_on_future_date-today_price)/today_price)*100:.0f}% difference")
    st.markdown(f"<h4 style='text-align: center;'>Predicted price {days_from_today} days from today ({future_date.strftime('%Y-%m-%d')}) is: ${predicted_price_on_future_date:.5f},  {((predicted_price_on_future_date-today_price)/today_price)*100:.0f}% difference</h4>", unsafe_allow_html=True)

    # Plot the actual and predicted prices using Plotly
    fig = go.Figure()
    df = df[df['date'] <= future_date]

    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=df['date'], y=df['predicted_next_day_price'],name='Historical Fair Price', mode='lines', line=dict(color='white')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['predicted_price'], mode='lines', name='Future Fair Price', line=dict(color='red')))


    # Highlight the future date and predicted price
    fig.add_vline(x=future_date.timestamp() * 1000, line=dict(color="purple", dash="dash"), annotation_text=f"Predicted price: {predicted_price_on_future_date:.5f}")
    fig.add_trace(go.Scatter(x=[closest_future_date], y=[predicted_price_on_future_date], mode='markers', marker=dict(color='red', size=10), name='Predicted Fair Price'))
    # Update the layout based on the chart type
    if chart_type == "Linear":
        fig.update_layout(xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=False)
    elif chart_type == "Logarithmic":
        fig.update_layout(xaxis_title='Date', yaxis=dict(type='log', title='Price'), xaxis_rangeslider_visible=False)
    # Assuming you've already completed your figure setup above, now add a watermark
    fig.add_annotation(
        text="KASPING.STREAMLIT.APP",  # The watermark text
        align='left',
        opacity=0.2,  # Adjust opacity to make the watermark lighter
        font=dict(color="yellow", size=35),  # Adjust font color and size
        xref='paper',  # Position the watermark relative to the entire figure
        yref='paper',
        x=0.5,  # Centered horizontally
        y=0.5,  # Centered vertically
        showarrow=False,  # Do not show an arrow pointing to the text
    )

    st.plotly_chart(fig, use_container_width=True)
    expander = st.expander('About the chart')
    expander.write('''
    You might find it surprising to see the predicted value fluctuate. Typically, power law charts depict the fair price as a constant, straight line (on log-log charts) because they are curve-fitted on the past data for the best fit. 
    
    However, this doesn't reveal past predictions, which is crucial for assessing the reliability of these forecasts.
    
    This chart is designed differently. It shows predictions as they would have been made using all available data at each point in the past. The goal is to demonstrate the degree to which power law predictions can vary, giving you insight into their consistency.
    ''')


if dashboard == 'Risk Visualization':

    st.markdown(f"<h2 style='text-align: center;'>{instrument} Risk Visualization</h2>", unsafe_allow_html=True)


    
    chart_type = st.sidebar.select_slider(
        'Select scale type',
        options=['Linear', 'Logarithmic'],
        value = "Logarithmic")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=(f'Actual vs Predicted Prices - {instrument}', 'Percentage Difference between Actual and Predicted Prices'))

    # Data preprocessing
    df['Value'] = df['close']  # Rename 'close' to 'Value'
    df = df[df['Value'] > 0]  # Filter out data points without a price
    if instrument == "Bitcoin (BTC)":
        
        df = df[df.index > 1200]
        df['MA'] = df['predicted_next_day_price'].rolling(2, min_periods=1).mean().dropna()
        df['Preavg'] = (np.log(df.Value) - np.log(df['MA'])) * df.index**.395
        
        # Normalization to 0-1 range
        df['avg'] = (df['Preavg'] - df['Preavg'].cummin()) / (df['Preavg'].cummax() - df['Preavg'].cummin())

    else:
        # Calculate the Risk Metric
        df['Preavg'] = ((np.log(df.Value) - (df['predicted_next_day_price'])) /np.log(df['predicted_next_day_price'])) ### balanced advisor
        # Normalization to 0-1 range
        df['avg'] = (df['Preavg'] - df['Preavg'].cummin()) / (df['Preavg'].cummax() - df['Preavg'].cummin())
        df['avg'] =1-df['avg']

    # Title for the plots
    annotation_text = f"Updated: {df['date'].iloc[-1].strftime('%Y-%m-%d')} | Price: {round(df['Value'].iloc[-1], 5)} | Risk: {round(df['avg'].iloc[-1], 2)}"


    if chart_type == "Linear":
        # Scatter plot of Price colored by Risk values
        fig = go.Figure(data=go.Scatter(x=df['date'], y=df['Value'], mode='markers', marker=dict(size=8, color=df['avg'], colorscale='Jet', showscale=True)))
        fig.update_yaxes(title='Price ($USD)', showgrid=True)
        fig.update_layout(template='plotly_dark', title_text=annotation_text)
        st.plotly_chart(fig, use_container_width=True)
    if chart_type == "Logarithmic":
        # Scatter plot of Price colored by Risk values
        fig = go.Figure(data=go.Scatter(x=df['date'], y=(df['Value']), mode='markers', marker=dict(size=8, color=df['avg'], colorscale='Jet', showscale=True)))
        fig.update_yaxes(title='Price ($USD)',type = "log", showgrid=True)
        fig.update_layout(template='plotly_dark', title_text=annotation_text)

    fig.add_annotation(
        text="KASPING.STREAMLIT.APP",  # The watermark text
        align='left',
        opacity=0.2,  # Adjust opacity to make the watermark lighter
        font=dict(color="yellow", size=35),  # Adjust font color and size
        xref='paper',  # Position the watermark relative to the entire figure
        yref='paper',
        x=0.5,  # Centered horizontally
        y=0.5,  # Centered vertically
        showarrow=False,  # Do not show an arrow pointing to the text
    )

        
    st.plotly_chart(fig, use_container_width=True)

    # Plot Price and Risk Metric
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Scatter(x=df['date'], y=df['Value'], name='Price', line=dict(color='gold')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['avg'], name='Risk', line=dict(color='white')), secondary_y=True)

    # Add colored risk zones
    opacity = 0.2
    for i in range(5, 0, -1):
        opacity += 0.05
        fig.add_hrect(y0=i*0.1, y1=((i-1)*0.1), line_width=0, fillcolor='green', opacity=opacity, secondary_y=True)
    for i in range(6, 10):
        opacity += 0.1
        fig.add_hrect(y0=i*0.1, y1=((i+1)*0.1), line_width=0, fillcolor='red', opacity=opacity, secondary_y=True)

    # Update layout
    fig.update_xaxes(title='Date')
    fig.update_yaxes(title='Price ($USD)', type='log', showgrid=False)
    fig.update_yaxes(title='Risk', type='linear', secondary_y=True, showgrid=True, tick0=0.0, dtick=0.1, range=[0, 1])
    fig.update_layout(template='plotly_dark', title={'text': annotation_text, 'y': 0.9, 'x': 0.5})

    fig.add_annotation(
        text="KASPING.STREAMLIT.APP",  # The watermark text
        align='left',
        opacity=0.2,  # Adjust opacity to make the watermark lighter
        font=dict(color="yellow", size=35),  # Adjust font color and size
        xref='paper',  # Position the watermark relative to the entire figure
        yref='paper',
        x=0.5,  # Centered horizontally
        y=0.5,  # Centered vertically
        showarrow=False,  # Do not show an arrow pointing to the text
    )



    st.plotly_chart(fig, use_container_width=True)





if dashboard == 'Trend Predictor':
    st.markdown(f"<h2 style='text-align: center;'>{instrument} Machine Learning Trend Predictor v1</h2>", unsafe_allow_html=True)

    df = un_encrypt_file('data/kas_d_with_predictions.csv')
    df['date'] = pd.to_datetime(df['date'])
    trend_thresh = st.sidebar.slider(
        label='Trend Probability >',
        value=0.7,
        min_value=0.0,
        max_value=0.99,    
    )


    highlight_color = 'yellow'

    highlight_opacity = 0.2


    df = df.reset_index(drop=True)
    up_prob = df['pred_prob'].round(2).values[-1]
    date_updated = df['date'].max().strftime('%Y-%m-%d')
    st.markdown(f"<h4 style='text-align: center;'>Uptrend probability for tomorrow: {up_prob}, Updated: {date_updated}</h4>", unsafe_allow_html=True)
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            showlegend=False,
        )   
    )

    # Create the threshold column, so that we can see the effect of changing
    # the model "surety" - this is needed to generate the highlighted regions
    df['threshold'] = df['pred_prob'] >= float(trend_thresh)

    # Find the intervals where the model has detected the pattern
    df_pattern = (
        df[df['threshold']]
        .groupby((~df['threshold']).cumsum())
        ['date']
        .agg(['first', 'last'])
    )
    df_pattern['first'] += pd.Timedelta(days=1)
    df_pattern['last'] += pd.Timedelta(days=1)
    # For each interval, plot as a highlighted section
    for idx, row in df_pattern.iterrows():
        fig.add_vrect(
            x0=row['first'], 
            x1=row['last'],
            line_width=0,
            fillcolor=highlight_color,
            opacity=highlight_opacity,
        )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        xaxis_title="Date",
        yaxis_title=f"{instrument} Price ($)",
        width=1200,
        height=700,
    )
    fig.add_annotation(
        text="KASPING.STREAMLIT.APP",  # The watermark text
        align='left',
        opacity=0.2,  # Adjust opacity to make the watermark lighter
        font=dict(color="yellow", size=35),  # Adjust font color and size
        xref='paper',  # Position the watermark relative to the entire figure
        yref='paper',
        x=0.5,  # Centered horizontally
        y=0.5,  # Centered vertically
        showarrow=False,  # Do not show an arrow pointing to the text
    )
    st.plotly_chart(fig, use_container_width=True)



    # Assuming df_test has been previously loaded or defined
    # Ensure the 'pred' column is based on a threshold of 0.9 on 'pred_prob'
    df['pred'] = df['pred_prob'] >= float(trend_thresh)

    # Shift the 'Close' and 'Open' values to get the next day's values for calculations
    df['next_day_close'] = df['close'].shift(-1)
    df['next_day_open'] = df['open'].shift(-1)

    # Calculate the returns for the next day, based on the prediction being 1 (true) or 0 (false)
    df['return_after_pred_1'] = df.apply(lambda x: x['next_day_close'] - x['next_day_open'] if x['pred'] else 0, axis=1)
    df['return_after_pred_0'] = df.apply(lambda x: x['next_day_close'] - x['next_day_open'] if not x['pred'] else 0, axis=1)

    # Calculate win rates for predictions equal to 1 and predictions equal to 0
    win_rate_pred_1 = df[df['pred']]['return_after_pred_1'].gt(0).mean()
    win_rate_pred_0 = df[~df['pred']]['return_after_pred_0'].gt(0).mean()

    # Determine the number of trades for both scenarios
    num_trades_pred_1 = df['pred'].sum()
    num_trades_pred_0 = (~df['pred']).sum()

    # Output the win rates and number of trades
    print(f"Win rate after pred=1: {win_rate_pred_1:.2%} (Number of trades: {num_trades_pred_1})")
    print(f"Win rate after pred=0: {win_rate_pred_0:.2%} (Number of trades: {num_trades_pred_0})")



    # Assuming df_test is your DataFrame and it's already been prepared as per your provided code
    df['pred'] = df['pred_prob'] >= trend_thresh
    # Determine the cumulative returns and generate x-axis values for trades after pred=1 and pred=0
    df['cumulative_return_after_pred_1'] = df[df['pred']]['return_after_pred_1'].cumsum()
    df['cumulative_return_after_pred_0'] = df[~df['pred']]['return_after_pred_0'].cumsum()

    # Generate trade number sequences for plotting
    trade_numbers_pred_1 = np.arange(1, df['pred'].sum() + 1)
    trade_numbers_pred_0 = np.arange(1, (~df['pred']).sum() + 1)

    # Create Plotly Go figure
    fig = go.Figure()

    # PnL Chart for predictions = 1
    fig.add_trace(go.Scatter(
        x=trade_numbers_pred_1,
        y=df['cumulative_return_after_pred_1'].dropna(),
        mode='lines+markers',
        name='Prediction > threshold',
        marker=dict(color='green'),  # Set marker color for markers
        line=dict(color='green')  # Set line color for lines
    ))

    # PnL Chart for predictions = 0
    fig.add_trace(go.Scatter(
        x=trade_numbers_pred_0,
        y=df['cumulative_return_after_pred_0'].dropna(),
        mode='lines+markers',
        name='Prediction < threshold'
    ))
    st.markdown(f"<h4 style='text-align: center;'></h4>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'></h4>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'></h4>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'></h4>", unsafe_allow_html=True)


    st.markdown(f"<h4 style='text-align: center;'>Price Action after Prediction</h4>", unsafe_allow_html=True)

    # Update layout for a shared x-axis
    fig.update_layout(

        xaxis_title='Trade Number',
        yaxis_title='Returns',
        legend_title='Scenario',
        template='plotly_dark',  # Choose a theme, if desired
    )

    fig.add_annotation(
        text="KASPING.STREAMLIT.APP",  # The watermark text
        align='left',
        opacity=0.2,  # Adjust opacity to make the watermark lighter
        font=dict(color="yellow", size=35),  # Adjust font color and size
        xref='paper',  # Position the watermark relative to the entire figure
        yref='paper',
        x=0.5,  # Centered horizontally
        y=0.5,  # Centered vertically
        showarrow=False,  # Do not show an arrow pointing to the text
    )
    st.plotly_chart(fig, use_container_width=True)

    # Calculate win rates for predictions equal to 1 and predictions equal to 0
    win_rate_pred_1 = df[df['pred']]['return_after_pred_1'].gt(0).mean()
    win_rate_pred_0 = df[~df['pred']]['return_after_pred_0'].gt(0).mean()

    # Determine the number of trades for both scenarios
    num_trades_pred_1 = df['pred'].sum()
    num_trades_pred_0 = (~df['pred']).sum()


    st.markdown(f"Win rate for prediction > {trend_thresh}: **{win_rate_pred_1:.2%}** (# trades: {num_trades_pred_1})")
    st.markdown(f"Win rate for prediction < {trend_thresh}: **{win_rate_pred_0:.2%}**(# trades: {num_trades_pred_0})")



    expander = st.expander('About the model')
    expander.write('''
    The model uses a Random Forest algorithm to try to predict whether the price will close above the 20-day SMA in 5 days. If the prediction "probability" (not an actual probability, but a way to quantify how confident the model is) is above the threshold (0.6-0.8 recommended; paradoxically, the highest values are not always the best as that usually indicates the trend is at its peak and may reverse), we enter at the next day's open and exit at the next day's close.
    **Why is it not showing complete KAS history:** The model needs to be trained on some data, and it is unfair to then say "look how well the model does (on the data it has already seen)" that is why the performance is shown only on the interval, which the model was not trained on.
    **DISCLAIMER:** If you trade using only this model, expect to lose all your money. Anything you manage to keep should be considered a miracle.
    
    If you decide to trade using this early version model, make sure that you only take long positions and that the slope of the 20-day SMA is positive.               ''')
    
st.warning("**Follow us on** [Twitter](https://twitter.com/KaspingApp)", icon="💸")  # Adjust font color and size

expander = st.expander('**Contact**')
expander.write('''
               



The aim of this project is to create a suite of tools for Kaspa (and other) investors to manage their positions intelligently, connect with like-minded people. You can get in touch on [Twitter](https://twitter.com/AlgoTradevid) or [join the beta waitlist](https://form.jotform.com/240557098994069).



''')

expander = st.expander('ReadME - about the project')
expander.write('''
               

This project was built on the code shared by [Danny Groves Ph.D.](https://twitter.com/DrDanobi), link to the original [github repository](https://github.com/GrovesD2/market_monitor_trend_dash/tree/main#readme). The idea for the Trend Predictor was also built on "Dr. Danobi's" code and concept.

The concept that the price of some cryptocurrencies is well explained by the [Power law](https://en.wikipedia.org/wiki/Power_law) relationship was proposed in 2018 by [Giovanni Santostasi](https://twitter.com/Giovann35084111) on Reddit and has recently started to gain popularity on X, due to its accuracy in predicting Bitcoin's price. This law seems to work very well for KAS too, although we have a much shorter price history for it.

For the risk metric visualization, the code from [Bitcoin Raven](https://github.com/BitcoinRaven/Bitcoin-Risk-Metric-V2) was used.

The aim of this project is to create a suite of tools for Kaspa (and other) investors to manage their positions intelligently, connect with like-minded people, and improve my skills in dashboard creation and machine learning. You can get in touch on [Twitter](https://twitter.com/AlgoTradevid) or [join the beta waitlist](https://form.jotform.com/240557098994069).



''')

