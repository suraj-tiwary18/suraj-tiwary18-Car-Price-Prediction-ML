# Multiple Linear Regression (Car Price Prediction)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# DataSet :-
data = pd.read_csv("car_dataset.csv")

# Data Frame :-
df = pd.DataFrame(data)

# Handle missing values :-
df.drop(['model', 'transmission', 'reg_year', 'overall_cost', 'has_insurance', 'spare_key', 'reg_number', 'title'], axis = 1, inplace = True)

df['engine_capacity(CC)'] = df['engine_capacity(CC)'].fillna(df['engine_capacity(CC)'].median())
df['km_driven'] = df['km_driven'].fillna(df['km_driven'].median())

df['brand'] = df['brand'].fillna(df['brand'].mode()[0])
df['fuel_type'] = df['fuel_type'].fillna(df['fuel_type'].mode()[0])
df['ownership'] = df['ownership'].fillna(df['ownership'].mode()[0])

current_year = 2026 #2026 is current year 
df['make_year'] = pd.to_numeric(df['make_year'], errors='coerce') # to_nueric -> to convert into numeric values int/float , errors -> if any values not convert into int/float so it will change value into nan
df['car_age'] = current_year - df['make_year'] 
df.drop('make_year', axis = 1, inplace = True)
df = df.dropna(subset=['car_age']) # dropna -> it will drop nan values in columns 
# df['car_age'] = df['car_age'].fillna(df['car_age'].median())

# maping :-
df['ownership'] = df['ownership'].str.strip().str.lower() # .str.strip() -> it will removes the extra starting and ending spaces
df['ownership'] = df['ownership'].map({ # .map -> it will conver string into numbers by the help of dictionary
    '1st owner': 1,
    '2nd owner': 2,
    '3rd owner': 3,
    '4th owner': 4
})

# One-Hot Encoded :-
df = pd.get_dummies(df, columns=['brand', 'fuel_type'], drop_first=True, dtype=int) # pd.get_dummies -> it will convert catagorical(string) data into numarical columns 

# # Seaborn Graphs :- 
# # Price Distribution
# sns.histplot(df['price'], kde=True)
# plt.title("Car Price Distribution")
# plt.show()

# # car age vs price
# sns.scatterplot(x='car_age', y = 'price', data = df)
# plt.title("Car Age vs Price")
# plt.show()

# # KM Driven vs Price
# sns.scatterplot(x='km_driven', y='price', data=df)
# plt.title("KM Driven vs Price")
# plt.show()

# Split Data :-
X = df.drop('price', axis = 1) # axis = 0 -> rows and axis = 1 -> columns 
y = df['price']

# Train Test Split 
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Featuer Scaling :-
scaler = StandardScaler()
numerical_cols = ['car_age', 'engine_capacity(CC)', 'km_driven']

x_train[numerical_cols] = scaler.fit_transform(x_train[numerical_cols]) # fit() -> it will calculate mean and std , transform() -> x - mean / std to scale values
x_test[numerical_cols] = scaler.transform(x_test[numerical_cols]) # in testing only use transform() 

# print(df.columns)

# Model Train
model = LinearRegression() # The model is ready to make predictions but hasn't learned anything yet
model.fit(x_train, y_train) # trains a linear regression model using training data and learns the relationship between the features and the target.

# Prediction :-

# User input 
engine = float(input("Enter Engine Capacity (CC): "))
km = float(input("Enter KM Driven: "))
owner = int(input("Enter Ownership (1-4): "))
age = int(input("Enter Car Age: "))
brand = input("Enter Brand: ").strip().title()
fuel = input("Enter Fuel Type: ").strip().title()

new_data = pd.DataFrame(0, index=[0], columns=X.columns)

new_data["engine_capacity(CC)"] = engine
new_data['km_driven'] = km
new_data['ownership'] = owner
new_data["car_age"] = age

brand_col = "brand_" + brand
fuel_col = "fuel_type_" + fuel

if brand_col not in new_data.columns:
    print("Brand not found in training data.")
    exit()

if fuel_col not in new_data.columns:
    print("Fuel type not found in training data.")
    exit()

new_data[brand_col] = 1
new_data[fuel_col] = 1

new_data[numerical_cols] = scaler.transform(new_data[numerical_cols]) # Scale user input using the same scaler fitted on the training data so the model will predict accurate

# Prediction model
prediction = model.predict(new_data) # a linear regression model to predict the estimated price of a car based on user inputs.
print(f"\nPredicted Car Price: ₹{prediction[0]:,.2f}")

y_pred = model.predict(x_test) # for check how model is predict accurate

sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--' # red line in graph 
)

plt.show()

joblib.dump(model, "car_price_model.joblib")
joblib.dump(scaler, "scaler.joblib")