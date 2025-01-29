import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


data_path = "C:/Users/aarit/Downloads/mergedData.csv"
df = pd.read_csv(data_path)

# customizable weights for gold, silver, bronze
gold_weight = 2
silver_weight = 1
bronze_weight = 0.5

# calculate the weighted medal count with weights
df['Weighted_Medal_Count'] = (
    df['Gold'] * gold_weight + 
    df['Silver'] * silver_weight + 
    df['Bronze'] * bronze_weight
)

# create a new column shifting each medal count up one to represent 'previous weighted medal count'
df['Prev_Weighted_Medal_Count'] = df.groupby('Country_Name')['Weighted_Medal_Count'].shift(1)
df['Prev_Weighted_Medal_Count'] = df['Prev_Weighted_Medal_Count'].fillna(0).astype(int)

# customizable weights for each sports category
aquatic_weight = 0.75  
gymnastics_weight = 0.25  
athletics_weight = 0.75 
other_weight = 0.1  


df['Weighted_Athletes'] = (df['# aquatics athletes'] * aquatic_weight + 
                           df['# gymnastics athletes'] * gymnastics_weight + 
                           df['# athletics athletes'] * athletics_weight + 
                           df['# other athletes'] * other_weight)

df = df[df['Year'] >= 1964]

# df_2024 = df[df['Year'] == 2024]
# df_2024 = df_2024[['Country_Name', 'Prev_Weighted_Medal_Count', 'Weighted_Athletes']]
# df_2024.to_csv("C:/Users/aarit/Downloads/2024resultDataIMRPOVED.csv")

# Multiple Linear Regression with 2 independent variables
X = df[['Weighted_Athletes', 'Prev_Weighted_Medal_Count']]  
y = df['Total']  

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the total medals for 2028
# medals_last_olympics = np.array([[257, 125]])  # Provide both features: Weighted_Athletes and Prev_Weighted_Medal_Count
# predicted_medals = model.predict(medals_last_olympics)
# print(f"Predicted Total Medals for 2028 (given {medals_last_olympics[0][0]} weighted athletes and {medals_last_olympics[0][1]} previous weighted medals):", predicted_medals[0])

# see results of model
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)
print("Coefficient (slope):", model.coef_)
print("Intercept:", model.intercept_)


