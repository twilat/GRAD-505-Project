''' This file is here to make it easier for me to run all the dataset cleans to initialize the datasets before my other work.'''

import pandas as pd
import matplotlib.pyplot as plt

# Perform exploratory data analysis
# Identify any issues that exist with your dataset (e.g. missing or erroneous data)
# Assess whether your dataset can effectively answer your research questions
# Create preliminary visualizations addressing your questions

# read csv into a Pandas dataframe
nest_attempts_df = pd.read_csv('../dataset_files/attempts_locs-20230518.csv')
nest_visits_df = pd.read_csv('../dataset_files/checks-20230518.csv')

# Show the unique columns and select relevant columns for each question, will add Parasite column later for relevant question
nest_attempts_1 = nest_attempts_df[['Attempt ID', 'Species Code', 'Young Fledged', 'Clutch Size', 'Habitat 1m', 'Substrate']]
nest_attempts_2 = nest_attempts_df[['Attempt ID', 'Species Code', 'Young Fledged', 'Clutch Size', 'Substrate']]
nest_attempts_3 = nest_attempts_df[['Attempt ID', 'Species Code', 'Young Fledged', 'Clutch Size']]

# Function to show counts of data, counts of missing data in the relevant variables for each question
# See relevant columns and drop rows where there's missing data in the relevant variables
def clean_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    def drop_missing_data(column: str, df: pd.DataFrame) -> pd.DataFrame:
            value_counts = df[column].isnull().value_counts()
            print(f"Missing data for {column}: \n{value_counts}")
            if (True in value_counts):
                print(f"There are {value_counts[True]} attempts missing data for {column} column")
            else:
                print(f"There is no missing data for {column} column")
            return df[df[column].isnull() == False]
    results_df = dataframe.copy(deep=True)
    for col in dataframe.columns.unique():
        results_df = drop_missing_data(col, results_df)
    return results_df.reset_index(drop=True) # reset the index after dropping rows with missing data

''' Research Question 1 
Predictors: habitat, substrate
Response: proportion fledged '''
# Clean the data and add Proportion Fledged column
cleaned_attempts_1 = clean_dataset(nest_attempts_1)
cleaned_attempts_1.head()
cleaned_attempts_1 = cleaned_attempts_1[cleaned_attempts_1['Clutch Size'] > 0] # take out the entries where Clutch Size = 0
cleaned_attempts_1['Proportion Fledged'] = cleaned_attempts_1['Young Fledged'] / cleaned_attempts_1['Clutch Size']

''' Research Question 2 
Predictors: substrate, host species
Response: # of parasite eggs '''
# Clean the data
cleaned_attempts_2 = clean_dataset(nest_attempts_2)

# To incorporate the parasite count variable, we need to:
# 1) disregard all the rows that don't have a value for Brood Parasite Eggs Count
# 2) get only the unique attempt IDs for those
# 3) join with the nest_attempts_df on attempt ID
parasites_counted_df = nest_visits_df[nest_visits_df['Brood Parasite Eggs Count'].notnull()]
unique_parasite_eggs_counted_df = parasites_counted_df.drop_duplicates(subset='Attempt ID', ignore_index=True)
unique_parasites_counted_df = unique_parasite_eggs_counted_df[['Attempt ID', 'Brood Parasite Eggs Count']]

# joined dataframe
attempts_with_parasite_count_2 = pd.merge(cleaned_attempts_2, unique_parasite_eggs_counted_df, how='inner', on = 'Attempt ID')

''' Research Question 3'''
# These results will be slightly different from the other similar ones before since we're filtering out the null data
# only based on what columns are relevant to each question. If we want to make it more consistent across all the
# questions, we can filter out all the null data for all the columns, but then we would lose data that exists for
# one question but not another. For instance, question 2 filters out null substrate rows, but we keep that data
# in this question since we are not looking at substrate.

# To incorporate the parasite count variable, we need to:
# 1) disregard all the rows that don't have a value for Brood Parasite Live Young Count
# 2) get only the unique attempt IDs for those
# 3) join with the nest_attempts_df on attempt ID
parasites_counted_df = nest_visits_df[nest_visits_df['Brood Parasite Live Young Count'].notnull()]
unique_parasites_counted_df = parasites_counted_df.drop_duplicates(subset='Attempt ID', ignore_index=True)
unique_parasites_counted_df = unique_parasites_counted_df[['Attempt ID', 'Brood Parasite Live Young Count']]

# joined dataframe
attempts_with_parasite_count_2 = pd.merge(cleaned_attempts_2, unique_parasites_counted_df, how='inner', on = 'Attempt ID')

# Clean the data and join it with the parasites dataframe
cleaned_attempts_3 = clean_dataset(nest_attempts_3)
attempts_with_parasite_count_3 = pd.merge(cleaned_attempts_3, unique_parasites_counted_df, how='inner', on = 'Attempt ID')

# TODO We can also do a proportion of the parasite young count out of the young total.
