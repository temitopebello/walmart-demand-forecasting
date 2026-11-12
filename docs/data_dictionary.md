# Data Dictionary

Source: Walmart Recruiting Store Sales Forecasting (Kaggle)
URL: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting

## train.csv (421,570 rows)

| Column | Type | Description | Module 2 Fishbone reference |
|---|---|---|---|
| Store | integer | Store identifier (1 to 45) | People bone |
| Dept | integer | Department identifier (1 to 99) | Process bone |
| Date | date | Week ending date | Measurement bone |
| Weekly_Sales | numeric | Weekly sales in USD (target variable) | - |
| IsHoliday | boolean | Holiday-weighted week flag (×5 in WMAE) | Measurement bone |

## features.csv

| Column | Type | Description | Cadence |
|---|---|---|---|
| Store | integer | Store identifier | weekly |
| Date | date | Week ending date | weekly |
| Temperature | numeric | Store-region temperature in Fahrenheit | weekly |
| Fuel_Price | numeric | Store-region fuel price | weekly |
| MarkDown1 to MarkDown5 | numeric | Promotional markdown amounts (sparse) | weekly (sparse) |
| CPI | numeric | Consumer Price Index for the region | monthly (interpolated to weekly) |
| Unemployment | numeric | Unemployment rate for the region | monthly (interpolated to weekly) |
| IsHoliday | boolean | Holiday flag | weekly |

## stores.csv (45 rows)

| Column | Type | Description | Fairness relevance |
|---|---|---|---|
| Store | integer | Store identifier | key |
| Type | categorical | Store tier (A, B, C) | primary fairness attribute (Module 1 objective) |
| Size | integer | Store square footage | correlates with Type |
