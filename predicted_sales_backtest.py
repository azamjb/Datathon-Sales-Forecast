import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# Importing Libraries ^


FILE_PATH    = "retail_data.xlsx"
PRODUCT_CODE = "23166"   


# Defining data file and product stock code ^


df = (pd.read_excel(FILE_PATH, engine="openpyxl")
      
        .assign(InvoiceDate=lambda d: pd.to_datetime(d["InvoiceDate"], errors="coerce"))
)

df = df[df["Quantity"] > 0].copy() # removing negative values for simplicity purposes
df["StockCode"] = (df["StockCode"].astype(str).str.strip().str.upper()) # normalizing StockCodes


# Loading file, cleaning data ^


ts = (df.loc[df["StockCode"] == PRODUCT_CODE,["InvoiceDate", "Quantity"]]
        .set_index("InvoiceDate")
        .resample("D").sum() # Sum rows by each day
        .asfreq("D"))  

ts["Quantity"] = ts["Quantity"].fillna(0)   # 0 sales on blank days

p99 = ts["Quantity"].quantile(0.99)
ts["Quantity"] = ts["Quantity"].clip(upper=p99)

median_day = ts["Quantity"].median()
ts["Spike"] = (ts["Quantity"] > 3 * median_day).astype(int)  # simple spike flag


# Building time series data set, filtered to specified product^


cutoff = ts.index.max() - pd.DateOffset(months=3)
train  = ts.loc[:cutoff]
valid  = ts.loc[cutoff + pd.Timedelta(days=1):]


# Splitting the data into a training and validation set (last 3 months for validation) ^


first_sale = df.loc[df["StockCode"] == PRODUCT_CODE, "InvoiceDate"].min()
print("First ever sale of this SKU:", first_sale)

train_df = train.reset_index().rename(columns={"InvoiceDate": "ds", "Quantity": "y"})
train_df["Spike"] = train["Spike"].values                        # add regressor column

m = Prophet(weekly_seasonality=True, yearly_seasonality=True, seasonality_mode="multiplicative")
m.add_regressor("Spike")                                         # register Spike
m.fit(train_df)                                                  # feeding the model historical 'train' data


# Building prophet forecasting model ^


future = m.make_future_dataframe(periods=len(valid), freq="D") 
future["Spike"] = future["ds"].map(ts["Spike"]).fillna(0).astype(int)   # pass Spike to future

fcst   = (m.predict(future).set_index("ds")["yhat"]) # use model to forecast 3 months into future (exact length of 'valid')

pred_valid = fcst.loc[valid.index] # select dates that align with 'valid' for comparison


# Using model for forecasting ^


mae  = mean_absolute_error(valid["Quantity"], pred_valid)
rmse = mean_squared_error(valid["Quantity"], pred_valid, squared=False)
smape = (pred_valid - valid["Quantity"]).abs() / \
        ((pred_valid.abs() + valid["Quantity"].abs()) / 2)
smape = smape.mean()

print(f"\nValidation scores for SKU {PRODUCT_CODE}")
print(f"  MAE  : {mae:,.2f} units")
print(f"  RMSE : {rmse:,.2f} units")
print(f"SMAPE : {smape:.2%}")


# accuracy metrics ^


total_pred = pred_valid.sum()
print(f"\nExpected units shipped (Prophet prediction) "
      f"for {PRODUCT_CODE} in that 3-month period: {total_pred:,.0f}")


print("Mean daily sales :", valid["Quantity"].mean())
print("Median daily sales:", valid["Quantity"].median())


# Predicting total quantity of units over 3-month window ^


plt.figure(figsize=(10,4))
plt.plot(valid.index, valid["Quantity"], label="Actual", lw=2)
plt.plot(pred_valid.index, pred_valid,   label="Predicted", ls="--")
plt.title(f"Sales Forecast – Medium Ceramic Top Storage Jar")
plt.xlabel("Date"); plt.ylabel("Units / Day"); plt.legend(); plt.tight_layout()
plt.show()

# Building visual plot ^



# BACK TEST ^
