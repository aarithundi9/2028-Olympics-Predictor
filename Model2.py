import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Upload and read file
data_path = "C:/Users/aarit/Downloads/mergedData.csv"
df = pd.read_csv(data_path)

# Define new weights for gold, silver, and bronze
gold_weight = 2
silver_weight = 1
bronze_weight = 0.5

# Recalculate the Weighted_Medal_Count with new weights
df['Weighted_Medal_Count'] = (
    df['Gold'] * gold_weight + 
    df['Silver'] * silver_weight + 
    df['Bronze'] * bronze_weight
)

# Shift all values up to determine the previous weighted medal count (create new column)
df['Prev_Weighted_Medal_Count'] = df.groupby('Country_Name')['Weighted_Medal_Count'].shift(1)

# Fill NaN values with 0 and convert to integer
df['Prev_Weighted_Medal_Count'] = df['Prev_Weighted_Medal_Count'].fillna(0).astype(int)

# Define the weights for each sport
aquatic_weight = 0.75  # Example weight, adjust as necessary
gymnastics_weight = 0.25  # Example weight, adjust as necessary
athletics_weight = 0.75  # Example weight, adjust as necessary
other_weight = 0.1  # Example weight, adjust as necessary

# Calculate the weighted sum
df['Weighted_Athletes'] = (df['# aquatics athletes'] * aquatic_weight + 
                           df['# gymnastics athletes'] * gymnastics_weight + 
                           df['# athletics athletes'] * athletics_weight + 
                           df['# other athletes'] * other_weight)

df = df[df['Year'] >= 1968]

# Filter data for 2024 and onwards (assuming you already have the data for 2024)
df_2024 = df[df['Year'] == 2024]
# Create a new dataframe with only the relevant columns (Country and Total Medals in 2024)
df_2024 = df_2024[['Country_Name', 'Prev_Weighted_Medal_Count', 'Weighted_Athletes']]
df_2024.to_csv("C:/Users/aarit/Downloads/2024resultDataIMRPOVED.csv")

# Linear Regression
# Correct X definition as a 2D array, passing both columns as separate features
# X = df[['Weighted_Athletes', 'Prev_Weighted_Medal_Count']]  # Use both features for regression
# y = df['Total']  # Target variable (y)

# # Create and train the linear regression model
# model = LinearRegression()
# model.fit(X, y)

# # Model coefficients and intercept
# print("Coefficient (slope):", model.coef_)
# print("Intercept:", model.intercept_)

# # # Predict the total medals for 2028
# # medals_last_olympics = np.array([[257, 125]])  # Provide both features: Weighted_Athletes and Prev_Weighted_Medal_Count
# # predicted_medals = model.predict(medals_last_olympics)
# # print(f"Predicted Total Medals for 2028 (given {medals_last_olympics[0][0]} weighted athletes and {medals_last_olympics[0][1]} previous weighted medals):", predicted_medals[0])

# # Evaluate the model
# y_pred = model.predict(X)
# mse = mean_squared_error(y, y_pred)
# r2 = r2_score(y, y_pred)

# print("Mean Squared Error (MSE):", mse)
# print("R-squared (R2):", r2)

# # # Optionally: Plotting the linear regression (useful if you're comparing 2D data)
# # plt.figure(figsize=(8, 6))
# # plt.scatter(df['Weighted_Athletes'], y, color='blue', label='Actual Data')
# # plt.plot(df['Weighted_Athletes'], y_pred, color='red', label='Regression Line (Predicted Medal Count)')
# # plt.title('Total Medals vs. Weighted Athlete Value (Starting in 1964)')
# # plt.xlabel('Weighted Athlete Value')
# # plt.ylabel('Total Medal Count')
# # plt.legend()
# # plt.show()
