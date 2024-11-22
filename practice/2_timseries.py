import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import SimpleExpSmoothing, AutoReg
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from math import sqrt

df = pd.read_csv("../practical8/IPG2211A2N.csv", index_col="DATE", parse_dates=True)
plt.figure(figsize=(10, 5))
plt.plot(df, label="Electric and Gas Utilities")
plt.title("Industrial Production")
plt.xlabel("Date")
plt.ylabel("Production")
plt.legend()
plt.grid()
plt.show()

decompos_results = seasonal_decompose(df, period=12, model="multiplicative")
decompos_results.plot()
plt.show()

plt.figure(figsize=(12, 6))
decompos_results = seasonal_decompose(df, period=12, model="multiplicative")
decompos_results.trend.plot()
plt.show()

plt.figure(figsize=(12, 6))
decompos_results = seasonal_decompose(df, period=12, model="multiplicative")
decompos_results.seasonal.plot()
plt.show()

plt.figure(figsize=(12, 6))
decompos_results = seasonal_decompose(df, period=12, model="multiplicative")
decompos_results.resid.plot()
plt.show()

plt.figure(figsize=(12, 6))
df1 = df.copy()
df1["MA_3"] = df1["IPG2211A2N"].rolling(window=3).mean()
df1["MA_6"] = df1["IPG2211A2N"].rolling(window=6).mean()
df1["MA_12"] = df1["IPG2211A2N"].rolling(window=12).mean()
df1["EMA"] = df1["IPG2211A2N"].ewm(com=10).mean()

plt.plot(df1["IPG2211A2N"], label="Original")
plt.plot(df1["MA_3"], label="Moving Average - 3 months", color="green")
plt.plot(df1["MA_6"], label="Moving Average - 6 months", color="purple")
plt.plot(df1["MA_12"], label="Moving Average - 12 months", color="orange")
plt.plot(df1["EMA"], label="Exponential Moving Average", color="black")

plt.title("Moving Average")
plt.xlabel("Date")
plt.ylabel("Production")
plt.legend()
plt.grid()
plt.show()

df2 = pd.read_csv("../practical10/CarPrice_Assignment.csv")
df2["time_index"] = np.arange(len(df2))
df2["price"] = pd.to_numeric(df2["price"], errors="coerce")
df2.dropna(subset=["price"], inplace=True)

exp_model = ExponentialSmoothing(
    df2["price"], seasonal=None, trend=None, damped_trend=False
).fit(smoothing_level=0.5)
exp_forecast = exp_model.forecast(steps=12)

plt.figure(figsize=(10, 6))
plt.plot(df2["time_index"], df2["price"], label="Original")
plt.plot(df2["time_index"], exp_model.fittedvalues, label="Exponential Smoothing")
plt.plot(range(len(df2), len(df2) + 12), exp_forecast, label="Forecast", linestyle="--")
plt.legend()
plt.title("Exponential Smoothing Forecast")
plt.show()

linear_model = LinearRegression()
linear_model.fit(df2[["time_index"]], df2["price"])
df2["linear_trend"] = linear_model.predict(df2[["time_index"]])

plt.figure(figsize=(10, 6))
plt.plot(df2["time_index"], df2["price"], label="Original")
plt.plot(df2["time_index"], df2["linear_trend"], label="Linear Trend")
plt.legend()
plt.title("Linear Trend Fit")
plt.show()

ar_model = AutoReg(df2["price"], lags=5).fit()
ar_forecast = ar_model.predict(start=len(df2), end=len(df2) + 11)

plt.figure(figsize=(10, 6))
plt.plot(df2["time_index"], df2["price"], label="Original")
plt.plot(ar_model.fittedvalues.index, ar_model.fittedvalues, label="AR Fitted Values")
plt.plot(
    range(len(df2), len(df2) + 12), ar_forecast, label="AR Forecast", linestyle="--"
)
plt.legend()
plt.title("Autoregressive Model Forecast")
plt.show()

rmse = sqrt(mean_squared_error(ar_model.fittedvalues[:12], ar_forecast))
rmse

train = df["IPG2211A2N"][: len(df["IPG2211A2N"]) - 7]
test = df["IPG2211A2N"][len(df["IPG2211A2N"]) - 7 :]
adfuller(df["IPG2211A2N"], autolag="AIC")

pacf = plot_pacf(df["IPG2211A2N"], lags=22)
pcf = plot_acf(df["IPG2211A2N"], lags=22)

autoreg = AutoReg(train, lags=5).fit()
autoreg.summary()

import statsmodels.api as sm

df3 = pd.read_csv("../practical9/CarPrice_Assignment.csv")
df3

X_1 = df3["horsepower"]
y_1 = df3["citympg"]
X_1 = sm.add_constant(X_1)
model_1 = sm.OLS(y_1, X_1).fit()
model_1.summary()

X_2 = df3["horsepower"]
y_2 = df3["highwaympg"]
X_2 = sm.add_constant(X_2)
model_2 = sm.OLS(y_2, X_2).fit()
model_2.summary()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df3["horsepower"], y=df3["citympg"], color="blue")
plt.plot(
    df3["horsepower"],
    model_1.predict(sm.add_constant(df3["horsepower"])),
    color="red",
    label="Regression Line",
)
plt.title("Scatterplot of City MPG vs Horsepower")
plt.xlabel("Horsepower")
plt.ylabel("City MPG")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df3["horsepower"], y=df3["highwaympg"], color="green")
plt.plot(
    df3["horsepower"],
    model_2.predict(sm.add_constant(df3["horsepower"])),
    color="red",
    label="Regression Line",
)
plt.title("Scatterplot of Highway MPG vs Horsepower")
plt.xlabel("Horsepower")
plt.ylabel("Highway MPG")
plt.legend()
plt.show()

X_citympg = sm.add_constant(df3["citympg"])
model_price_citympg = sm.OLS(df3["price"], X_citympg).fit()
print("Model 1: price vs citympg")
print(model_price_citympg.summary())

X_highwaympg = sm.add_constant(df3["highwaympg"])
model_price_highwaympg = sm.OLS(df3["price"], X_highwaympg).fit()
print("\nModel 2: price vs highwaympg")
print(model_price_highwaympg.summary())

X_enginesize = df3[["enginesize"]]
y_price = df3["price"]
X_enginesize = sm.add_constant(X_enginesize)
model_enginesize = sm.OLS(y_price, X_enginesize).fit()
print(model_enginesize.summary())

X_curbweight = df3[["curbweight"]]
y_price = df3["price"]
X_curbweight = sm.add_constant(X_curbweight)
model_curbweight = sm.OLS(y_price, X_curbweight).fit()
print(model_curbweight.summary())

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=df3["enginesize"], y=df3["price"], color="blue")
plt.plot(
    df3["enginesize"],
    model_enginesize.predict(sm.add_constant(df3["enginesize"])),
    color="red",
    label="Regression Line",
)
plt.title("Scatterplot of Engine Size vs Price")
plt.xlabel("Engine Size (L)")
plt.ylabel("Price ($)")
plt.legend()

plt.subplot(1, 2, 2)
sns.scatterplot(x=df3["curbweight"], y=df3["price"], color="green")
plt.plot(
    df3["curbweight"],
    model_curbweight.predict(sm.add_constant(df3["curbweight"])),
    color="orange",
    label="Regression Line",
)
plt.title("Scatterplot of Curb Weight vs Price")
plt.xlabel("Curb Weight (lbs)")
plt.ylabel("Price ($)")
plt.legend()

plt.tight_layout()
plt.show()

numeric_df = df3.drop(["price", "citympg", "highwaympg"], axis=1).select_dtypes(
    include=[float, int]
)
independent_vars = sm.add_constant(numeric_df)

vif_data = pd.DataFrame()
vif_data["Feature"] = independent_vars.columns
vif_data["VIF"] = [
    variance_inflation_factor(independent_vars.values, i)
    for i in range(independent_vars.shape[1])
]
print(vif_data)

y_pred = model_curbweight.predict(sm.add_constant(df3["curbweight"]))
residuals = y_price - y_pred
residuals

plt.scatter(y_pred, residuals, color="blue", edgecolor="k", alpha=0.7)
plt.axhline(y=0, color="red", linestyle="--")
plt.title("Residuals vs Fitted Values")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()

plt.hist(residuals, bins=10, edgecolor="k", alpha=0.7)
plt.title("Histogram of Residuals")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
