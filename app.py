# import libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px 
import datetime

from datetime import date,timedelta
from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller

# Title
app_name = 'Stock Market Forecasting'
st.title(app_name)

# Sub Header
st.subheader('This app is created to forecast the stock market price of the selected company')

# Add image from online source
# st.image("https://media.istockphoto.com/id/467147904/vector/abstract-diagrams-stock-media-image-digital-graphs.jpg?s=612x612&w=0&k=20&c=6YvItLVH_Rh04MfZAgmmuZ85285V5C9UDVSL6ecsT-I=")

# take input from the user of start and end date

#sidebar
st.sidebar.header('Select Parameters from Below')

start_date = st.sidebar.date_input('Start Date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End Date', date(2020, 12, 31))

# Add Ticker symbol list 
ticker_list =["AAPL" , "MSFT" , "GOOG","GOOGL" , "META", "TSLA" , "NVDA", "ADBE", "PYPL" , "INTC" , "CMCSA","NFLX","PEP"]
ticker = st.sidebar.selectbox('Select the company' , ticker_list)

# Add image from online source in sidebar
st.sidebar.image("https://media.istockphoto.com/id/467147904/vector/abstract-diagrams-stock-media-image-digital-graphs.jpg?s=612x612&w=0&k=20&c=6YvItLVH_Rh04MfZAgmmuZ85285V5C9UDVSL6ecsT-I=" , width=200)


# Fetch Data from user inputs usinf yfinance library

data = yf.download(ticker,start=start_date,end=end_date) 

# Add date as a column to the dataframe
data.insert(0,"Date" , data.index , True)
data.reset_index(drop=True,inplace=True)

st.write('Data from' ,start_date , 'to', end_date)

st.write(data)

# Plot of the data
st.header('Data Visualization')
st.subheader('Plot of the data')

st.write("**Note Select Specific Date Range from the sidebar, or zoom in on the plot and select your specific column")

fig = px.line(data, x='Date',y=data.columns, title='Closing price of the stock',width=1000,height=600)
st.plotly_chart(fig)

# Add a select box to select column from the data
column = st.selectbox('Select the column to be used for forecasting',data.columns[1:])

# subsetting the data

data = data[['Date',column]]
st.write("Selected Data")
st.write(data)

# ADF test check stationarity 
st.header("Is data Stationary?")
# st.write('**Note:** If p-value is less than 0.05, then data is stationary')
st.write(adfuller(data[column])[1] < 0.05)

# Decompose the data 
st.header('Decomposition of the data')
decomposition = seasonal_decompose(data[column], model = 'additive' , period = 15)
st.write(decomposition.plot())

# make the same plots in plotly
st.write("## Plotting the decomposition in plotly")
st.plotly_chart(px.line(x=data["Date"], y=decomposition.trend,title="Trend" ,width=1000,    height=400, labels={'x':'Date','y':'Price'}).update_traces(line_color='Blue'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.seasonal,title="Seasonality" ,width=1000,    height=400, labels={'x':'Date','y':'Price'}).update_traces(line_color='Green'))
st.plotly_chart(px.line(x=data["Date"], y=decomposition.resid,title="Residuals" ,width=1000,    height=400, labels={'x':'Date','y':'Price'}).update_traces(line_color='Red', line_dash='dot'))

# Lets run the model
# user input for three parameters of the model and seasonal order
p = st.slider('Select the value of p',0,5,2)
d = st.slider('Select the value of d',0,5,1)
q = st.slider('Select the value of q',0,5,2)

seasonal_order = st.number_input('Select the value of seasonal p', 0, 24, 12)

model = sm.tsa.statespace.SARIMAX(data[column],order=(p,d,q),seasonal_order=(p,d,q,seasonal_order))
model  = model.fit()

# Print Model Summary 
st.header('Model Summary')
st.write(model.summary())
st.write("---")

# predict the future values (Forecasting)
st.write("<p style = 'color:green; font-size:50px; font-weight:bold;'> Forecasting the data </p>" , unsafe_allow_html=True)
forecast_period = st.number_input('Slect the number of days to forecast', 1, 365, 10)

# predict the future values
predictions = model.get_prediction(start=len(data),end=len(data)+forecast_period)
predictions = predictions.predicted_mean
# st.write(predictions)

# Add index to the predictions
predictions.index = pd.date_range(start = end_date , periods = len(predictions), freq = 'D')
predictions = pd.DataFrame(predictions)
predictions.insert(0 , "Date" , predictions.index , True)
predictions.reset_index(drop=True,inplace=True)
st.write("Predictions" , predictions)
st.write("Actual Data" , data)

st.write("---")

# plot the data
fig = go.Figure()

# Add actual data to the plot 
fig.add_trace(go.Scatter(x=data["Date"],y=data[column],mode='lines',name='Actual',line=dict(color='Blue')))

# Add predicted data to the plot 
fig.add_trace(go.Scatter(x=predictions["Date"],y=predictions["predicted_mean"],mode='lines', name = 'Predicted', line=dict(color='red')))

# set the title and axis labels
fig.update_layout(title='Actual vs Predicted', xaxis_title='Date',yaxis_title='Price',width= 1000,height=400)

# Display the plot
st.plotly_chart(fig)

# Buttons to show/Hide Seperate Plots
show_plots = False
if st.button('Show Seperate Plots'):
    if not show_plots:
        st.write(px.line(x=data["Date"], y=data[column],title="Actual" , width=1000 , height=400, labels={'x':'Date','y':'Price'}).update_traces(line_color='Green'))
        st.write(px.line(x=predictions["Date"],y=predictions["predicted_mean"],title="predicted" , width=1000 , height=400 , labels={'x':'Date','y':'Price'}).update_traces(line_color='Red', line_dash='dot'))
        show_plots = True
    else:
        show_plots = False

# Add hide plot button
hide_plots = False
if st.button("Hide Seperate Plots"):
    if not hide_plots:
        hide_plots = True
    else: 
        hide_plots = False

st.write("---")

