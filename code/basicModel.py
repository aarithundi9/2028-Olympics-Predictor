import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import seaborn as sns
import matplotlib.pyplot as plt


data_path = "C:/Users/aarit/Downloads/totalMedalCountSorted.csv"
df = pd.read_csv(data_path)

#creates a new data frame with only the results after 1964
df_last50 = df[df['Year'] >= 1964]
df_last50 = df_last50.drop(df_last50.columns[-1], axis=1)

#new dataframe with a new index
df_last50 = df_last50.reset_index(drop=True)
print(df_last50.head())

# Define the variables in the regression model, X = independent, y = dependent 
X = df_last50[['Medals_Last_Olympics']]  
y = df_last50['Total_Medal_Count']      

# create regression model
model = LinearRegression()
model.fit(X, y)


print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)

# for sample predictions
# medals_last_olympics = np.array([[126]])  
# predicted_medals = model.predict(medals_last_olympics)
# print(f"Predicted Total Medals for 2028 (given {medals_last_olympics[0][0]} medals in last Olympics):", predicted_medals[0])

# evaluate the model
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)


# saving 2024 data to run 2028 predictions 
# df_2024 = df[df['Year'] == 2024]
# df_2024 = df_2024[['Name_Of_Country', 'Total_Medal_Count']]
# df_2024.to_csv("C:/Users/aarit/Downloads/2024resultData.csv")


# Plotting the linear regression
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line (Predicted Medal Count)')
plt.title('Total Medals vs. Medals in Last Olympics (Starting in 1964)')
plt.xlabel('Medals in the Last Olympics')
plt.ylabel('Total Medal Count')
plt.legend()
plt.show()


# df_last50.to_csv("C:/Users/aarit/Downloads/totalMedalCountLast50.csv")
