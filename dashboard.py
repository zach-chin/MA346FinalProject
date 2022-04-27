# importing modules to work with avocado data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import streamlit as st
# for fitting to a curve
from scipy.optimize import curve_fit
# for creating a time series analysis
import datetime
import seaborn as sns
import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

### title
st.title("Predicting Future Prices of Avocados")
st.subheader("By: Zachary Chin")
st.write("LINKS TO REPORT AND CODE")

# importing the data set
avo = pd.read_csv("avocado.csv")

# converting date to datetime and renaming the first column
avo["Date"] = pd.to_datetime(avo["Date"])
avo = avo.rename(columns={"Unnamed: 0": "Week"})

# add season column using datetime
season_dict = {1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",
               7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall",12:"Winter"}
avo["Season"] = avo["Date"].dt.month.map(season_dict)

# Splitting into organic and conventional type avocodos and indexing by date
def index_by_date(df):
    df = df.sort_values("Date", ascending=True)
    df.index = df["Date"]
    return df

# splitting
avoOrg = avo[avo["type"]=="organic"]
avoOrg = index_by_date(avoOrg)
avoCon = avo[avo["type"]=="conventional"]
avoCon = index_by_date(avoCon)

# inputs for predicting
testRegion = st.selectbox("Choose a Region to predict prices in:", avo["region"].unique(), index=21)
testType = st.radio("Choose the type of avocado:", ["Organic","Conventional"])
testSeason = st.selectbox("Choose a season to predict the price during:", ["Winter", "Spring", "Summer", "Fall"])

# this is the curve chosen to fit the price prediction of avocados
## use this curve?? https://otexts.com/fpp2/holt-winters.html
def logistic_curve ( x, β0, β1, β2 ):
        return β0 / ( 1 + np.exp(β1 *(-x+β2) ))

# ε * sin(α + β * t) + γt + ρ
# this curve with several variables fits them to manipulate a sin curve to fit the data
# ε for amplitude, α for shift left/right, β for frequency, γ for slope, ρ is intercept
def sin_curve (t, α, β, γ, ε, ρ):
    return ε * (np.sin((α + β * t))) + (γ * t) + ρ

def fit_test_to_curve(region, avotype):
    """
    creates a model from several filters and then uses that model to predict the price of an avocado given the certain filters
    """
    # setting it as the conventional series or an organic series
    if avotype == "Conventional":
        series = pd.Series(avoCon.loc[avoCon["region"]==region, "AveragePrice"])
        series.index = pd.Series(avoCon.loc[avoCon["region"]==region, "Date"])
    else:
        series = pd.Series(avoOrg.loc[avoOrg["region"]==region, "AveragePrice"])
        series.index = pd.Series(avoOrg.loc[avoOrg["region"]==region, "Date"])

    dateseries = series.index

    # fitting the data to the curve
    # resetting index to make it work with curve_fit
    series = series.reset_index(drop=True).dropna()

    # setting x as the date, and ys as the average price
    xs = series.index
    ys = series

    # using the curve_fit function to get betas
    beta_guess = [0, 0.1, -1, 0.5, 1.5]
    found_params, covariance = curve_fit(sin_curve, xs,ys,p0=beta_guess, maxfev=2000)
    α, β, γ, ε, ρ = found_params

    # find SSE
    fit_model = lambda x: sin_curve(x, α, β, γ, ε, ρ)
    SSE = np.sum((fit_model(series.index)-series)**2)

    # returning betas
    return fit_model, SSE, series
    
predicted, curveSSE, series = fit_test_to_curve(testRegion, testType)

# puts a timer for the user while the prediction and graph load
with st.spinner('Please wait for the curve to be fit'):
    # time series to use
    def fit_to_timeseries(region, avotype):
        # fits a time-series ARIMA model to selected data
        timeseries = pd.DataFrame()
        # setting it as the conventional series or an organic series
        if avotype == "Conventional":
            timeseries["AveragePrice"] = pd.Series(avoCon.loc[avoCon["region"]==region, "AveragePrice"])
            timeseries.index = pd.Series(avoCon.loc[avoCon["region"]==region, "Date"])
        else:
            timeseries["AveragePrice"] = pd.Series(avoOrg.loc[avoOrg["region"]==region, "AveragePrice"])
            timeseries.index = pd.Series(avoOrg.loc[avoOrg["region"]==region, "Date"])

        # fitting to a model
        mod = sm.tsa.statespace.SARIMAX(timeseries["AveragePrice"], trend='n', order=(0,0,1), seasonal_order=(0,1,1,52))
        results = mod.fit()

        # series of the predicted values
        timeseries["forecast"] = results.predict(start=53, end=169, dynamic=True)

        # finding the SSE of the model
        timeseries = timeseries.reset_index(drop=True).dropna()
        SSE = np.sum((timeseries.loc[53:, "forecast"] - timeseries.loc[53:, "AveragePrice"])**2)

        # returning predicted values and SSE
        return results, timeseries["forecast"], SSE
        
    results, forecast, timeSSE = fit_to_timeseries(testRegion, testType)

    # fitting the predictions to the index in the data
    if timeSSE > curveSSE:
        if testSeason == "Spring":
            predictedPrice = predicted(170+52) # for spring 2019
        elif testSeason == "Summer":
            predictedPrice = predicted(182+52) # summer
        elif testSeason == "Fall":
            predictedPrice = predicted(194+52) # fall
        else:
            predictedPrice = predicted(206+52) # winter
    else:
        if testSeason == "Spring":
            predictedPrice = results.predict(170+52)[0] # for spring 2019
        elif testSeason == "Summer":
            predictedPrice = results.predict(182+52)[0] # summer
        elif testSeason == "Fall":
            predictedPrice = results.predict(194+52)[0] # fall
        else:
            predictedPrice = results.predict(206+52)[0] # winter

    # printing out the predicted price in the selected season

    if testType == "Conventional":
        subheadstring = f"The predicted value for a conventional avocado in {testRegion} during {testSeason} 2019 is ${predictedPrice:.2f}"
    else:
        subheadstring = f"The predicted value for an organic avocado in {testRegion} during {testSeason} 2019 is ${predictedPrice:.2f}"

    st.success(subheadstring)

    # plotting the predictions to show them
    if timeSSE > curveSSE:
        plt.title(f"Prices of Avocados between 04/01/2015 and 03/25/2018")
        plt.xlabel(f"Days since 04/01/2015")
        plt.ylabel("Price ($)")
        plt.scatter(series.index, series)
        plt.plot(series.index, predicted(series.index), c="red")
        plt.legend(labels=["Actual", "Predicted"])
        st.pyplot(plt.gcf())
    else:
        # making the forecast a usable length for plotting
        precedingNulls = pd.Series([np.nan]*53)
        forecast = precedingNulls.append(forecast)
        # plotting
        plt.title(f"Prices of Avocados between 04/01/2015 and 03/25/2018")
        plt.xlabel(f"Days since 04/01/2015")
        plt.ylabel("Price ($)")
        plt.scatter(series.index, series)
        plt.plot(series.index, forecast, c="red")
        plt.legend(labels=["Actual", "Predicted"])
        st.pyplot(plt.gcf())