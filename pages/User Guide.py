import streamlit as st

st.set_page_config(page_title="User Guide", page_icon="📈")

hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

st.title("Stock Prediction Application User Guide")

st.header("Welcome to the Stock Prediction Application!")
st.write("This guide will help you navigate and use the features of the application.")

st.header("1. Company Selection and Date Range")
st.write("On the sidebar, you can select a stock symbol from the dropdown menu. This will determine which company's stock data to analyze.")
st.write("You can also choose a starting date to train the model and specify the prediction period in terms of months or days.")

st.header("2. Initial Investment")
st.write("Enter the initial investment amount you want to use for calculating profit/loss in the 'Initial Investment' field.")

st.header("3. Run Forecast")
st.write("After configuring your preferences, click the 'Run Forecast' button to generate stock price predictions and insights.")

st.header("4. Results and Visualization")
st.write("The application will display various visualizations, including historical stock prices, predicted prices, and portfolio value over time.")
st.write("You'll also see the predicted closing and opening prices for the specified prediction period.")

st.header("5. Profitable Stocks")
st.write("At the bottom of the page, you'll find a list of companies that are predicted to be profitable based on the selected prediction model.")

st.header("6. Interacting with Visualizations")
st.write("You can interact with the visualizations to zoom in and out, pan, and more. Hover over data points to see detailed information.")

st.header("7. Experiment with Different Settings")
st.write("Feel free to adjust the prediction model, prediction period, and other settings to see how they impact the results.")

st.header("8. Contact")
st.write("If you have any questions or feedback, please feel free to contact the application creator, Bishrul Haq.")

st.header("9. Have Fun!")
st.write("Explore the application and gain insights into stock price predictions. Happy forecasting!")