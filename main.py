import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r'C:\Users\dell\OneDrive\Desktop\Avacado Price Pridiction\avocado.csv\avocado.csv')

# Convert date to datetime and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Filter for TotalUS region
df_us = df[df['region'] == 'TotalUS'].copy()

# Plot: Price over time
plt.figure(figsize=(12, 6))
plt.plot(df_us['Date'], df_us['AveragePrice'], color='green')
plt.title('Avocado Prices Over Time - Total US')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.grid(True)
plt.tight_layout()
plt.show()

# Add month & year columns
df_us['Month'] = df_us['Date'].dt.month
df_us['Year'] = df_us['Date'].dt.year

# Monthly average price
monthly_avg = df_us.groupby('Month')['AveragePrice'].mean()
monthly_avg.plot(kind='bar', color='skyblue')
plt.title('Average Avocado Price by Month')
plt.xlabel('Month')
plt.ylabel('Avg Price')
plt.grid(True)
plt.tight_layout()
plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Convert date to number
df_us['DateOrdinal'] = df_us['Date'].apply(lambda x: x.toordinal())

# Prepare features and target
X = df_us[['DateOrdinal']]
y = df_us['AveragePrice']
dates = df_us['Date']

# Split with dates
X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X, y, dates, test_size=0.2, shuffle=False
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Plot actual and predicted
plt.figure(figsize=(12, 6))
plt.plot(df_us['Date'], df_us['AveragePrice'], label='Actual (Full Data)', alpha=0.5)
plt.plot(dates_test, y_pred, label='Predicted (Test Data)', color='red', linestyle='--')
plt.title('Actual vs Predicted Avocado Prices (Linear Regression)')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


