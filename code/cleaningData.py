# the goal of this file is to clean the program data and have a list in order of sports with the most events

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_path = "C:/Users/aarit/Downloads/summerOly_athletes.csv"
df = pd.read_csv(data_path)

# print(df.head())

athletics_sports = ['Athletics']
aquatics_sports = ['Swimming']
gymnastics_sports = ['Gymnastics']

# method to 
def categorize_sport(sport):
    if sport in athletics_sports:
        return 'Athletics'
    elif sport in aquatics_sports:
        return 'Aquatics'
    elif sport in gymnastics_sports:
        return 'Gymnastics'
    else:
        return 'Other'

# s
df['Sport_Category'] = df['Sport'].apply(categorize_sport)

# finds the amount of athletes in each unique sports category and then separates the 4 categories
grouped_df = df.groupby(['Team', 'Year', 'Sport_Category']).size().unstack(fill_value=0)
grouped_df = grouped_df.reset_index()

#renames the newly createed columns for each sport category
grouped_df.columns.name = None  
grouped_df = grouped_df.rename(columns={'Athletics': '# athletics athletes', 
                                         'Aquatics': '# aquatics athletes', 
                                         'Gymnastics': '# gymnastics athletes', 
                                         'Other': '# other athletes'})


grouped_df = grouped_df[grouped_df['Year'] >= 1964]

# weights for each sport 
aquatic_weight = 2  
gymnastics_weight = 1  
athletics_weight = 2 
other_weight = 0.5  

# create the weighted athletes column based on weights
grouped_df['Weighted_Athletes'] = (grouped_df['# aquatics athletes'] * aquatic_weight + 
                                    grouped_df['# gymnastics athletes'] * gymnastics_weight + 
                                    grouped_df['# athletics athletes'] * athletics_weight + 
                                    grouped_df['# other athletes'] * other_weight)


grouped_df.rename(columns={'Team': 'Country_Name'}, inplace=True)

# print(grouped_df)

# grouped_df.to_csv("C:/Users/aarit/Downloads/newDataSet.csv")
