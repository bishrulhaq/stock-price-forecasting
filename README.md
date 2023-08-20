# Stock Price Forecasting Web Application 📈

This repository contains a Stock Price Forecasting Web Application built using Streamlit and Python. The application is created to predict future stock prices based on historical data. It allows users to select different parameters, visualize stock price trends, and analyze potential profits.

## Functionality 🛠️

1. **Data Loading and Company Selection** 🏢
   - The user can select a stock symbol from a list of companies.
   - The user can input an initial investment amount.
   - The user can choose between predicting stock prices for a number of months or days.
   - The user can choose between different prediction models (Machine Learning - Random Forest Regression or Neural Network - LSTM).

2. **Data Preprocessing and Visualization** 📊
   - The application fetches historical stock price data using the Yahoo Finance API.
   - The data is preprocessed to include relevant features such as day, month, and year.
   - Interactive plots are generated using Plotly to visualize historical stock prices, moving averages, and actual vs. predicted prices.

3. **Stock Price Prediction** 📈
   - The application predicts future stock prices using either a Random Forest Regression model or an LSTM neural network.
   - The predictions are displayed in interactive line charts, showing both actual and predicted stock prices.
   - For the Random Forest model, the application predicts open and close prices.

4. **Portfolio Management and Profit Calculation** 💰
   - The application calculates the portfolio value based on the predicted stock prices.
   - It displays the initial investment, number of shares bought, and the resulting profit or loss.

5. **List of Profitable Stock Companies** 💹
   - The application provides a list of profitable stock companies based on the selected criteria.
   - It displays the predicted stock prices, shares bought, and the calculated profit for each company.

## Getting Started

1. Clone this repository: `git clone https://github.com/bishrulhaq/stock-price-forecasting.git`
2. Navigate to the project directory: `cd stock-price-forecasting`
3. Install the required packages: `pip install -r requirements.txt`
4. Run the Streamlit app: `streamlit run app.py`

## Usage

1. Open your web browser and navigate to `http://localhost:8501` (default Streamlit port).
2. Use the sidebar to select a stock ticker, prediction period, and initial investment.
3. Visualize historical stock prices, moving averages, and predicted trends.
4. Analyze potential profits and losses based on the selected investment strategy.
5. Explore the list of profitable stock companies based on predictions.


## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Streamlit](https://streamlit.io/): Python library for creating interactive web apps.
- [yfinance](https://github.com/ranaroussi/yfinance): Fetch historical stock data from Yahoo Finance.
- [numpy](https://numpy.org/): Essential package for scientific computing in Python.
- [scikit-learn](https://scikit-learn.org/): Machine learning tools for Python.
- [plotly](https://plotly.com/): Create interactive and appealing data visualizations.
- [tensorflow](https://www.tensorflow.org/): Open-source machine learning framework.
- [pandas](https://pandas.pydata.org/): Data manipulation and analysis in Python.


## Author 👨‍💻

[Bishrul Haq](https://github.com/bishrulhaq)
