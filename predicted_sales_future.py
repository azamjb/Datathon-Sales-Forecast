import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Importing Libraries ^

FILE_PATH    = "retail_data.xlsx"
PRODUCT_CODE = "23166"

# Defining data file and product stock code ^

df = (pd.read_excel(FILE_PATH, engine="openpyxl")
        .assign(InvoiceDate=lambda d: pd.to_datetime(d["InvoiceDate"], errors="coerce"))
)

df = df[df["Quantity"] > 0].copy()
df["StockCode"] = (df["StockCode"].astype(str).str.strip().str.upper())

# Loading file, cleaning data ^

ts = (df.loc[df["StockCode"] == PRODUCT_CODE, ["InvoiceDate", "Quantity"]]
        .set_index("InvoiceDate")
        .resample("D").sum()
        .asfreq("D"))

ts["Quantity"] = ts["Quantity"].fillna(0)

p99 = ts["Quantity"].quantile(0.99)
ts["Quantity"] = ts["Quantity"].clip(upper=p99)

median_day = ts["Quantity"].median()
ts["Spike"] = (ts["Quantity"] > 3 * median_day).astype(int)

# Building time series data set, filtered to specified product^

cutoff = ts.index.max() - pd.DateOffset(months=3)
train = ts.loc[cutoff:] # <--- DIFFERENCE: only last 3 months for training

# No need to split train/valid separately now ^

first_sale = df.loc[df["StockCode"] == PRODUCT_CODE, "InvoiceDate"].min()
print("First ever sale of this SKU:", first_sale)

train_df = train.reset_index().rename(columns={"InvoiceDate": "ds", "Quantity": "y"})
train_df["Spike"] = train["Spike"].values

m = Prophet(weekly_seasonality=True, yearly_seasonality=False, seasonality_mode="additive")
m.add_regressor("Spike")
m.fit(train_df)

# Building prophet forecasting model ^

future = m.make_future_dataframe(periods=90, freq="D")
future["Spike"] = future["ds"].map(ts["Spike"]).fillna(0).astype(int)

fcst = m.predict(future).set_index("ds")

# Predicting total quantity of units over next 3 months ^

forecast_3m = fcst.loc[train.index.max() + pd.Timedelta(days=1):]

print(f"\nExpected units shipped (Prophet prediction) "
      f"for {PRODUCT_CODE} in next 90 days: {forecast_3m['yhat'].sum():,.0f}")

print("\nSummary statistics for next 3 months:")
print("Mean daily forecast sales :", forecast_3m["yhat"].mean())
print("Median daily forecast sales:", forecast_3m["yhat"].median())

# Building visual plot ^

plt.figure(figsize=(12,5))
plt.plot(train.index, train["Quantity"], label="Actual (Past)", lw=2)
plt.plot(fcst.index, fcst["yhat"], "--", label="Forecast (Next 90 days)", color="orange")
plt.fill_between(fcst.index, fcst["yhat_lower"], fcst["yhat_upper"],
                 color="lightgray", alpha=0.4, label="Forecast 80% band")
plt.axvspan(train.index.max(), fcst.index.max(), color="lightblue", alpha=0.3, label="Forecast period")
plt.title(f"Future Sales Forecast – Medium Ceramic Top Storage Jar")
plt.xlabel("Date"); plt.ylabel("Units / Day"); plt.legend(); plt.tight_layout()
plt.show()

# FINAL FORECAST ^
