## Kaspa Dashboard
This dashboard is an enhanced version of the work shared by [Danny Groves Ph.D.](https://twitter.com/DrDanobi), whose original code can be found in this GitHub repository [github repository](https://github.com/GrovesD2/market_monitor_trend_dash/tree/main#readme). The concept for the Trend Predictor was also inspired by the work and ideas of "Dr. Danobi."

## Key Features

## Power Law / Rainbow Chart
The concept that the price of some cryptocurrencies follows a Power Law relationship was first proposed in 2018 by Giovanni Santostasi on Reddit. This concept has gained significant popularity on social media platforms like X (formerly Twitter) due to its accuracy in predicting Bitcoin's price. Although the price history for Kaspa (KAS) is much shorter, this law appears to apply well to KAS too.

## Future Power Law
This chart allows you to check the expected future price of KAS or BTC using the current Power Law parameters. It helps visualize where the price might be heading based on historical trends.

## Risk Visualizer
This proprietary metric evaluates how far the current trading price deviates from the expected value as per the Power Law model. Importantly, this metric is not repainted, meaning it remains consistent over time. The code for this feature was adapted from Bitcoin Raven.

## SMA Crossover Chart
The SMA Crossover Chart is based on the observation that whenever the 85-day Simple Moving Average (SMA) crosses over the 66-day SMA, the price of Kaspa tends to surge shortly afterward. This tool is inspired by the Bitcoin Pi Cycle Top Indicator and uses SMA crossovers as a potential signal for significant price moves.
