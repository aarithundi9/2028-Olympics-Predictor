# the goal of this file is to clean the program data and have a list in order of sports with the most events

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = "C:/Users/aarit/Downloads/summerOly_athletes.csv"
df = pd.read_csv(data_path)

# print(df.head())

# Define the sports categories
athletics_sports = ['Athletics']
aquatics_sports = ['Swimming']
gymnastics_sports = ['Gymnastics']

# Add a new column to categorize sports into athletics, aquatics, gymnastics, or other
def categorize_sport(sport):
    if sport in athletics_sports:
        return 'Athletics'
    elif sport in aquatics_sports:
        return 'Aquatics'
    elif sport in gymnastics_sports:
        return 'Gymnastics'
    else:
        return 'Other'

# Apply the categorization function to the 'Sport' column
df['Sport_Category'] = df['Sport'].apply(categorize_sport)

# Group by 'Country' and 'Year' and count the occurrences of each category
grouped_df = df.groupby(['Team', 'Year', 'Sport_Category']).size().unstack(fill_value=0)

# Reset the index to make 'Country' and 'Year' columns
grouped_df = grouped_df.reset_index()

# Rename columns to match the desired format
grouped_df.columns.name = None  # Remove the 'Sport_Category' index
grouped_df = grouped_df.rename(columns={'Athletics': '# athletics athletes', 
                                         'Aquatics': '# aquatics athletes', 
                                         'Gymnastics': '# gymnastics athletes', 
                                         'Other': '# other athletes'})

grouped_df = grouped_df[grouped_df['Year'] >= 1968]

# Define the weights for each sport
aquatic_weight = 2  # Example weight, adjust as necessary
gymnastics_weight = 1  # Example weight, adjust as necessary
athletics_weight = 2  # Example weight, adjust as necessary
other_weight = 0.5  # Example weight, adjust as necessary

# Calculate the weighted sum
grouped_df['Weighted_Athletes'] = (grouped_df['# aquatics athletes'] * aquatic_weight + 
                                    grouped_df['# gymnastics athletes'] * gymnastics_weight + 
                                    grouped_df['# athletics athletes'] * athletics_weight + 
                                    grouped_df['# other athletes'] * other_weight)

# Display the resulting dataframe

grouped_df.rename(columns={'Team': 'Country_Name'}, inplace=True)

print(grouped_df)

grouped_df.to_csv("C:/Users/aarit/Downloads/newDataSet.csv")