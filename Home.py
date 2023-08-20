import streamlit as st
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import plotly.graph_objs as go

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

    X_train, X_test, y_train_close, y_test_close, y_train_open, y_test_open = train_test_split(
    X, y_close, y_open, test_size=0.2, random_state=42)

    model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
    model.fit(X_train, list(zip(y_train_close, y_train_open)))

    test_days = len(y_test_open)

    predicted = model.predict(X_test)
    predicted_close = predicted[:, 0]
    predicted_open = predicted[:, 1]

    # Display graphs for historical predictions using Plotly
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

    new_dates = pd.date_range(end_date + pd.DateOffset(days=1), end_date + pd.DateOffset(months=3)).date

    if prediction_type == "Months":
        new_dates = pd.date_range(end_date + pd.DateOffset(days=1), end_date + pd.DateOffset(months=num_months)).date
    else:
        new_dates = pd.date_range(end_date + pd.DateOffset(days=1), end_date + pd.DateOffset(days=num_days)).date

    # Generate new dates for the next three months
    new_dates = pd.date_range(end_date + pd.DateOffset(days=1), end_date + pd.DateOffset(months=num_months)).date

    days = [date.day for date in new_dates]
    months = [date.month for date in new_dates]
    years = [date.year for date in new_dates]

    date_info_df = pd.DataFrame({'day': days, 'month': months, 'year': years})
    predicted_unseen = model.predict(date_info_df)

    # Extracting the predicted close and open prices
    predicted_new_close = predicted_unseen[:, 0]
    predicted_new_open = predicted_unseen[:, 1]

    # Display predicted open and close prices for new dates using Plotly
    fig_pred_close = go.Figure()
    fig_pred_close.add_trace(go.Scatter(x=new_dates, y=predicted_new_close, mode='lines', name='Predicted Close Prices'))
    fig_pred_close.update_layout(title='Predicted Stock Close Prices for last {} days'.format(test_days), xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_pred_close)

    fig_pred_open = go.Figure()
    fig_pred_open.add_trace(go.Scatter(x=new_dates, y=predicted_new_open, mode='lines', name='Predicted Open Prices'))
    fig_pred_open.update_layout(title='Predicted Stock Open Prices for Next {} fays'.format(test_days), xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_pred_open)

    # Create interactive Plotly graph combining actual and predicted data for close prices
    fig_close = go.Figure()
    fig_close.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Actual Close Prices'))
    fig_close.add_trace(go.Scatter(x=stock_data.index[X_train.shape[0]:], y=predicted_close, mode='lines', name='Predicted Close Prices'))
    fig_close.add_trace(go.Scatter(x=new_dates, y=predicted_new_close, mode='lines', name='Predicted Close Prices for Next {} {}'.format(days_or_month,prediction_type)))
    fig_close.update_layout(title='Stock Close Price Prediction', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_close)

    # Create interactive Plotly graph combining actual and predicted data for open prices
    fig_open = go.Figure()
    fig_open.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Open'], mode='lines', name='Actual Open Prices'))
    fig_open.add_trace(go.Scatter(x=stock_data.index[X_train.shape[0]:], y=predicted_open, mode='lines', name='Predicted Open Prices'))
    fig_open.add_trace(go.Scatter(x=new_dates, y=predicted_new_open, mode='lines', name='Predicted Open Prices for Next {} {}'.format(days_or_month,prediction_type)))
    fig_open.update_layout(title='Stock Open Price Prediction', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig_open)

    new_end_date = end_date + pd.DateOffset(months=num_months)
    new_dates = pd.date_range(start=end_date + pd.DateOffset(days=1), end=new_end_date)
    predicted_new_close = predicted_unseen[:, 0]
    data = {'Date': new_dates,'Close': predicted_new_close}
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

    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged_stocks_data.index, y=merged_stocks_data['PortfolioValue'], mode='lines', name='Portfolio Value', line=dict(color='green')))
    fig.update_layout(title='Portfolio Value Over Time', xaxis_title='Date', yaxis_title='Value')
    st.plotly_chart(fig)

    initial_price = stock_data['Close'].iloc[-1]
    final_value = shares_bought * merged_stocks_data['Close'].iloc[-1]
    profit = final_value - float(initial_investment)
    st.markdown(f"<div style='{box_style}'>Initial Investment : ${initial_investment} </div>", unsafe_allow_html=True)
    st.markdown(f"<div style='{box_style}'>Number of shares Bought : {shares_bought:.2f} </div>", unsafe_allow_html=True)
    st.markdown(f"<div style='{box_style}'>Profit or Loss : ${profit:.2f} </div>", unsafe_allow_html=True)

    profitable_stocks = []

    # Display the list of profitable stocks
    st.title('List of Profitable Stock Companies')
    count = 0
    # Loop through each ticker symbol and check if the company is predicted to be profitable
    with st.spinner('Loading company data...'):
        for symbol in ticker_list:
            stock_data = fetch_stock_data(symbol, pd.Timestamp('2022-01-01'), pd.Timestamp.today())

            if not stock_data.empty:
                if count == selected_tickers:
                    break
                # Preprocess data
                stock_data['Date'] = stock_data.index
                stock_data['Date'] = stock_data['Date'].apply(lambda x: x.toordinal())

                X = stock_data['Date'].values.reshape(-1, 1)
                y_close = stock_data['Close'].values
                y_open = stock_data['Open'].values

                tickerData = yf.Ticker(symbol)

                # Fetching new dates for prediction
                num_months = 3
                new_end_date = pd.Timestamp.today() + pd.DateOffset(months=num_months)
                new_dates = pd.date_range(start=pd.Timestamp.today(), end=new_end_date)
                new_dates_ordinal = [date.toordinal() for date in new_dates]

                # Creating the model
                model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
                model.fit(X, list(zip(y_close, y_open)))

                # Predicting open and close prices for new dates
                predicted_unseen = model.predict(np.array(new_dates_ordinal).reshape(-1, 1))
                predicted_new_close = predicted_unseen[:, 0]
                predicted_new_open = predicted_unseen[:, 1]

                shares_bought = float(initial_investment) / float(y_close[-1])
                final_portfolio_value =  shares_bought * predicted_new_close[-1]
                profit = final_portfolio_value - float(initial_investment)

                if predicted_new_close[-1] > y_close[-1]:
                    count += 1
                    st.write(f"{symbol} ( {tickerData.info['shortName']} )")
                    st.write(f" Shared Bought: {shares_bought:.2f}")
                    st.write(f" Profit of ${profit:.2f}")
                    st.write('--')
