import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from datetime import timedelta
import pandas as pd

st.set_page_config(page_title="Home", page_icon="📈")

st.sidebar.header("Welcome to Stock Prediction Application")

st.sidebar.write(
    """This Application is created to Predict the stock prices of the companies."""
)

start_date = pd.Timestamp('2022-01-01')
end_date = pd.Timestamp.today() - pd.DateOffset(days=1)

box_style = (
    "border: 2px solid #333;"
    "background-color: #f0f0f0;"
    "color: #000000;"
    "border-radius: 10px;"
    "padding: 10px;"
    "margin-top: 20px;"
    "font-weight: bold;"
)

hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

st.markdown("<h1 style='color: yellow;'>Stock Market Price Forecasting Application</h1>", unsafe_allow_html=True)
st.write(
    """Created By : Bishrul Haq"""
)

data_load_state = st.text('Loading ...')

# Fetch stock data using yfinance
def fetch_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data

# Scale the features using MinMaxScaler
def scale_input(X):
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Sidebar - Company selection
ticker_list = pd.read_csv('data/company_list.csv')
ticker_list = ticker_list['Company']
symbol = st.sidebar.selectbox('Stock ticker', ticker_list)

selected_date = st.date_input("Select a date to Train from:",value=start_date)
start_date = pd.Timestamp(selected_date)

st.text('Data Trained from {} to {}'.format(start_date.date(), end_date.date()))

initial_investment = st.sidebar.text_input("Initial Investment ($)", value="100", key="initial_investment")

selected_tickers = st.sidebar.slider("Filter the number of companies which makes profit", 1, 30, 3)

prediction_type = st.sidebar.radio("Select prediction type", ["Months", "Days"], index=0)

num_months = 1
num_days = 1

if prediction_type == "Months":
    num_months = st.sidebar.slider("Select number of months for prediction", 1, 6, 3)
else:
    num_days = st.sidebar.slider("Select number of days for prediction", 1, 100, 30)


model_selected = st.sidebar.radio("Select prediction model", ["Machine Learning (Random Forest Regression)","Neural Network (LSTM)"], index=0)
# Fetch stock data
stock_data = fetch_stock_data(symbol, start_date, end_date)

# Preprocess data
stock_data['Date'] = pd.to_datetime(stock_data.index)  # Convert index to datetime
stock_data['day'] = stock_data['Date'].dt.day
stock_data['month'] = stock_data['Date'].dt.month
stock_data['year'] = stock_data['Date'].dt.year
X = stock_data[['day', 'month', 'year']]

y_close = stock_data['Close'].values
y_open = stock_data['Open'].values

tickerData = yf.Ticker(symbol)
string_name = symbol +' - '+tickerData.info['longName']
st.header('**%s**' % string_name)

if 'country' in tickerData.info:
    st.write('Country : '+ tickerData.info['country'])

if 'Industry' in tickerData.info:
    st.write('Industry : '+ tickerData.info['Industry'])

if 'Currency' in tickerData.info:
    st.write('Currency : '+ tickerData.info['Currency'])

if 'financial Currency' in tickerData.info:
    st.write('financialCurrency : '+ tickerData.info['financialCurrency'])

data_load_state.text('')
st.write('---')

# Calculate additional features
stock_data['DailyReturn'] = stock_data['Adj Close'].pct_change()  # Calculate daily returns
stock_data['MovingAvg_10'] = stock_data['Adj Close'].rolling(window=10).mean()  # 10-day moving average
stock_data['MovingAvg_50'] = stock_data['Adj Close'].rolling(window=50).mean()  # 50-day moving average

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Adj Close'], mode='lines', name='Adj Close'))
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MovingAvg_10'], mode='lines', name='10-day Moving Avg'))
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MovingAvg_50'], mode='lines', name='50-day Moving Avg'))
fig.update_layout(xaxis_title='Date', yaxis_title='Price', title='Stock Price and Moving Averages')
st.plotly_chart(fig)

st.write('Click Run Forcast on the sidebar to Predict the Stock Price 🗠')

if not initial_investment.isdigit():
    st.sidebar.error("Please enter a valid initial investment.")
    initial_investment = 0

days_or_month = num_months if prediction_type == "Months" else num_days

if not stock_data.empty and initial_investment.isdigit() and st.sidebar.button('Run Forecast'):

    data_load_state = st.text('Loading data...')
    st.subheader('Raw data')
    st.write(stock_data[['Close','Open','High','Low','Adj Close']].tail())
    data_load_state.text('Loading data... done!')

    X_train, X_test, y_train_close, y_test_close, y_train_open, y_test_open = train_test_split(scale_input(X), y_close, y_open, test_size=0.2, random_state=42)

    test_days = len(y_test_open)

    # Select Forecast Period
    new_dates = pd.date_range(end_date + pd.DateOffset(days=1), end_date + pd.DateOffset(months=3)).date

    if prediction_type == "Months":
        new_dates = pd.date_range(end_date + pd.DateOffset(days=1), end_date + pd.DateOffset(months=num_months)).date
    else:
        new_dates = pd.date_range(end_date + pd.DateOffset(days=1), end_date + pd.DateOffset(days=num_days)).date

    if model_selected == "Machine Learning (Random Forest Regression)":
        model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
        model.fit(X_train, list(zip(y_train_close, y_train_open)))
        predicted = model.predict(scale_input(X_test))
        predicted_close = predicted[:, 0]
        predicted_open = predicted[:, 1]

        fig_close = go.Figure()
        fig_close.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual Close Prices'))
        fig_close.add_trace(go.Scatter(x=stock_data.index[X_train.shape[0]:], y=predicted_close, mode='lines', name='Predicted Close Prices'))
        fig_close.update_layout(title='Stock Close Price Prediction', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_close)

        fig_open = go.Figure()
        fig_open.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Open'], mode='lines', name='Actual Open Prices'))
        fig_open.add_trace(go.Scatter(x=stock_data.index[X_train.shape[0]:], y=predicted_open, mode='lines', name='Predicted Open Prices'))
        fig_open.update_layout(title='Stock Open Price Prediction', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_open)

        days = [date.day for date in new_dates]
        months = [date.month for date in new_dates]
        years = [date.year for date in new_dates]

        date_info_df = pd.DataFrame({'day': days, 'month': months, 'year': years})
        predicted_unseen = model.predict(scale_input(date_info_df))

        # Extracting the predicted close and open prices
        predicted_new_close = predicted_unseen[:, 0]
        predicted_new_open = predicted_unseen[:, 1]

        # Display predicted open and close prices for new dates using Plotly
        fig_pred_close = go.Figure()
        fig_pred_close.add_trace(go.Scatter(x=new_dates, y=predicted_new_close, mode='lines', name='Predicted Close Prices'))
        fig_pred_close.update_layout(title='Predicted Stock Close Prices for last {} days'.format(test_days),xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_pred_close)

        fig_pred_open = go.Figure()
        fig_pred_open.add_trace(go.Scatter(x=new_dates, y=predicted_new_open, mode='lines', name='Predicted Open Prices'))
        fig_pred_open.update_layout(title='Predicted Stock Open Prices for Next {} fays'.format(test_days),xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_pred_open)

        # Create interactive Plotly graph combining actual and predicted data for close prices
        fig_close = go.Figure()
        fig_close.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual Close Prices'))
        fig_close.add_trace(go.Scatter(x=stock_data.index[X_train.shape[0]:], y=predicted_close, mode='lines',name='Predicted Close Prices'))
        fig_close.add_trace(go.Scatter(x=new_dates, y=predicted_new_close, mode='lines',name='Predicted Close Prices for Next {} {}'.format(days_or_month,prediction_type)))
        fig_close.update_layout(title='Stock Close Price Prediction', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_close)

        # Create interactive Plotly graph combining actual and predicted data for open prices
        fig_open = go.Figure()
        fig_open.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Open'], mode='lines', name='Actual Open Prices'))
        fig_open.add_trace(go.Scatter(x=stock_data.index[X_train.shape[0]:], y=predicted_open, mode='lines',name='Predicted Open Prices'))
        fig_open.add_trace(go.Scatter(x=new_dates, y=predicted_new_open, mode='lines',name='Predicted Open Prices for Next {} {}'.format(days_or_month,prediction_type)))
        fig_open.update_layout(title='Stock Open Price Prediction', xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig_open)

        if prediction_type == "Months":
            new_end_date = end_date + pd.DateOffset(months=num_months)
        else:
            new_end_date = end_date + pd.DateOffset(days=num_days)

        new_dates = pd.date_range(start=end_date + pd.DateOffset(days=1), end=new_end_date)
        predicted_new_close = predicted_unseen[:, 0]

        data = {'Date': new_dates, 'Close': predicted_new_close}
        predictions_df = pd.DataFrame(data)
        predictions_df.set_index('Date', inplace=True)  # Set 'date' column as index

        merged_stocks_data = pd.concat([stock_data[['Close']], predictions_df[['Close']]])
        yesterday_date = pd.to_datetime('today').normalize() - pd.Timedelta(days=1)
        selected_data = merged_stocks_data.loc[yesterday_date:]

        merged_stocks_data['DailyReturn'] = merged_stocks_data['Close'].pct_change()
        merged_stocks_data['CumulativeReturn'] = (1 + merged_stocks_data['DailyReturn']).cumprod()

        initial_price = stock_data['Close'].iloc[-1]
        shares_bought = float(initial_investment) / float(initial_price)
        merged_stocks_data['PortfolioValue'] = merged_stocks_data['CumulativeReturn'] * shares_bought * merged_stocks_data['Close']

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=merged_stocks_data.index, y=merged_stocks_data['PortfolioValue'], mode='lines', name='Portfolio Value', line=dict(color='green')))
        fig.update_layout(title='Portfolio Value Over Time', xaxis_title='Date', yaxis_title='Value')
        st.plotly_chart(fig)

        initial_price = stock_data['Close'].iloc[-1]
        final_value = shares_bought * merged_stocks_data['Close'].iloc[-1]
        profit = final_value - float(initial_investment)
        st.markdown(f"<div style='{box_style}'>Initial Investment : ${initial_investment} </div>",unsafe_allow_html=True)
        st.markdown(f"<div style='{box_style}'>Number of shares Bought : {shares_bought:.2f} </div>",unsafe_allow_html=True)
        st.markdown(f"<div style='{box_style}'>Profit or Loss : ${profit:.2f} </div>", unsafe_allow_html=True)

    else:
        with st.spinner("Predicting..."):
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

            sequence_length = 10
            generator = TimeseriesGenerator(scaled_data, scaled_data, length=sequence_length, batch_size=1)

            model = Sequential([
                LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')
            model.fit(generator, epochs=10)

            # Forecasting
            forecast_days = len(new_dates)
            forecast = []

            current_sequence = scaled_data[-sequence_length:]
            for _ in range(forecast_days):
                prediction = model.predict(current_sequence.reshape(1, sequence_length, 1))
                forecast.append(scaler.inverse_transform(prediction)[0, 0])
                current_sequence = np.append(current_sequence[1:], prediction[0])

            forecast_dates = [stock_data.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
            forecast_data = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
            forecast_data = forecast_data.set_index('Date')

            # Calculate Investment Profit
            if len(stock_data) > 0:
                initial_price = stock_data['Close'].iloc[-1]
                final_price = forecast_data['Forecast'].iloc[-1]
                shares_bought = float(initial_investment) / float(initial_price)
                final_portfolio_value = shares_bought * final_price

            # Display Results
            st.line_chart(stock_data['Close'], use_container_width=True)

            if prediction_type == "Months":
                st.title(f"Forecast for the Next {num_months} Months")
            else:
                st.title(f"Forecast for the Next {num_days} Days")

            st.line_chart(forecast_data, use_container_width=True)

            if len(stock_data) > 0:
                st.markdown(f"<div style='{box_style}'>Initial Investment : ${initial_investment} </div>",unsafe_allow_html=True)
                st.markdown(f"<div style='{box_style}'>Number of shares Bought : {shares_bought:.2f} </div>",unsafe_allow_html=True)
                st.markdown(f"<div style='{box_style}'>Profit or Loss : ${final_portfolio_value:.2f} </div>", unsafe_allow_html=True)


    profitable_stocks = []

    # Display the list of profitable stocks
    st.title('List of Profitable Stock Companies')
    count = 0
    # Loop through each ticker symbol and check if the company is predicted to be profitable
    with st.spinner('Loading company data...'):
        for symbol in ticker_list:
            stock_data = fetch_stock_data(symbol, start_date, end_date)

            if not stock_data.empty:
                if count == selected_tickers:
                    break
                # Preprocess data
                stock_data['Date'] = pd.to_datetime(stock_data.index)
                stock_data['day'] = stock_data['Date'].dt.day
                stock_data['month'] = stock_data['Date'].dt.month
                stock_data['year'] = stock_data['Date'].dt.year

                X = stock_data[['day', 'month', 'year']]

                y_close = stock_data['Close'].values
                y_open = stock_data['Open'].values

                tickerData = yf.Ticker(symbol)

                model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
                model.fit(scale_input(X), list(zip(y_close, y_open)))

                days = [date.day for date in new_dates]
                months = [date.month for date in new_dates]
                years = [date.year for date in new_dates]

                date_info_df = pd.DataFrame({'day': days, 'month': months, 'year': years})
                predicted_unseen = model.predict(scale_input(date_info_df))

                # Extracting the predicted close and open prices
                predicted_new_close = predicted_unseen[:, 0]
                predicted_new_open = predicted_unseen[:, 1]

                if predicted_new_close[-1] > y_close[-1]:
                    count += 1

                    shares_bought = float(initial_investment) / float(y_close[-1])
                    final_portfolio_value = shares_bought * predicted_new_close[-1]
                    profit = final_portfolio_value - float(initial_investment)

                    predicted_df = pd.DataFrame({
                        'Date': new_dates,
                        'Predicted Close Price': predicted_new_close,
                        'Predicted Open Price': predicted_new_open
                    })

                    # Display the predicted results in a line chart
                    st.subheader('Predicted Stock Prices for {}'.format(symbol))
                    st.line_chart(predicted_df.set_index('Date'))

                    st.write(f"{symbol} ( {tickerData.info['shortName']} )")
                    st.write(f" Shared Bought: {shares_bought:.2f}")
                    st.write(f" Profit of ${profit:.2f}")
                    st.write('--')