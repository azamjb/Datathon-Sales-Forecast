

import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# Importing Libraries ^


FILE_PATH    = "retail_data.xlsx"
PRODUCT_CODE = "20750"   


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


# Building time series data set, filtered to specified product^


cutoff = ts.index.max() - pd.DateOffset(months=3)
train  = ts.loc[:cutoff]
valid  = ts.loc[cutoff + pd.Timedelta(days=1):]


# Splitting the data into a training and validation set (last 3 months for validation) ^


first_sale = df.loc[df["StockCode"] == PRODUCT_CODE, "InvoiceDate"].min()
print("First ever sale of this SKU:", first_sale)

m = Prophet(weekly_seasonality=True, yearly_seasonality=True, seasonality_mode="multiplicative") # configuring empty prophet model

m.fit(train.reset_index().rename(columns={"InvoiceDate": "ds", "Quantity": "y"})) # feeding the model historical 'train' data


# Building prophet forecasting model ^


future = m.make_future_dataframe(periods=len(valid), freq="D") 

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
plt.title(f"Prophet validation – {PRODUCT_CODE}")
plt.xlabel("Date"); plt.ylabel("Units / Day"); plt.legend(); plt.tight_layout()
plt.show()

# Building visual plot ^



# BACK TEST ^
