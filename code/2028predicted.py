import pandas as pd
import numpy as np

data_file = "C:/Users/aarit/Downloads/2024resultDataIMRPOVED.csv"
df = pd.read_csv(data_file)

df['Predicted_medal_count'] = 0.0

# coefficients
beta_0 = -1.7693290390592704 
beta_1 = 0.49295382  
beta_2 = 0.17588152

# Calculate the predicted medal count using the linear regression equation with coefficients above
df['Predicted_medal_count'] = beta_0 + beta_1 * df['Prev_Weighted_Medal_Count'] + beta_2 * df['Weighted_Athletes']

# root mean squared value
rmse = 8.5491

# Add Medal_Range column with rounded values
df['Medal_Range'] = df['Predicted_medal_count'].apply(
    lambda x: f"[{max(0, round(x - rmse))}, {round(x + rmse)}]"
)

df['Predicted_medal_count'] = df['Predicted_medal_count'].round().astype(int)

for i in range(82):
    if df.loc[i, 'Predicted_medal_count'] < 0:
        df.loc[i, 'Predicted_medal_count'] = 0

df = df.sort_values(by='Predicted_medal_count', ascending=False)


df.reset_index(drop=True, inplace=True)
df.index += 1  

print(df)

df = df[['Country_Name', 'Predicted_medal_count', 'Medal_Range']]

# to convert to a table to use in latex
latex_table = df.to_latex(index=True, longtable=True)
print(latex_table)

