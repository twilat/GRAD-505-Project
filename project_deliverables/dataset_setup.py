''' This file is here to make it easier for me to run all the dataset cleans to initialize the datasets before my other work.'''

# All import statements for all files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.graphics.regressionplots import abline_plot
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, MaxNLocator)

# read csv into a Pandas dataframe
nest_attempts_df = pd.read_csv('dataset_files/attempts_locs-20230518.csv')
nest_visits_df = pd.read_csv('dataset_files/checks-20230518.csv')

# Filter data to only include records past and including 2020
nest_attempts_df = nest_attempts_df[nest_attempts_df['Year'] >= 2020]

# Select relevant columns for each question, will add Parasite column later for relevant question
nest_attempts_1 = nest_attempts_df[['Attempt ID', 'Species Code', 'Year', 'Young Fledged', 'Clutch Size', 'Habitat 1m']]
nest_attempts_2 = nest_attempts_df[['Attempt ID', 'Species Code', 'Year', 'Young Fledged', 'Clutch Size', 'Substrate']]
nest_attempts_3 = nest_attempts_df[['Attempt ID', 'Species Code', 'Year', 'Young Fledged', 'Clutch Size']]

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
Response: young fledged '''
# Clean the data
cleaned_attempts_1 = clean_dataset(nest_attempts_1)
cleaned_attempts_1.head()
cleaned_attempts_1 = cleaned_attempts_1[cleaned_attempts_1['Clutch Size'] > 0] # take out the entries where Clutch Size = 0

''' There are a few notable outliers in the young fledged-
33614   A1443815       purmar  2021           62.0         71.0      grass    poptow
22109   A1410605       purmar  2020           47.0         48.0      beach    nesbox
55840   A1614433       easblu  2023           44.0          5.0       golf    nesbox
15137   A1410608       purmar  2020           42.0         46.0      grass    nesbox

The third one shows an attempt where 44 young were fledged from a clutch size of 5. This seems unlikely and may be due
to human error from the observer, so we will take it out. We will keep the other attempts where clutch size < young fledged
because in those cases it's possible that the parasitic birds were also being counted. 44 however is an extreme number that 
seems pretty unlikely for a single nest in one year.

The other three outliers with a significantly larger clutch size and young fledged than other attempts seem to be unlikely for
those breeds. For the purmar (purple martin), according to https://www.allaboutbirds.org/guide/Purple_Martin/lifehistory,
the clutch size can be max 6 eggs, with the number of broods ranging from 1-2. While it's possible that some of the other
attempts may exceed this maximum, we will take these specific instances out because they are very extreme and skew the data. '''
cleaned_attempts_1 = cleaned_attempts_1.drop(cleaned_attempts_1[cleaned_attempts_1['Attempt ID']
                                                                .isin(['A1443815', 'A1410605', 'A1614433', 'A1410608']) == True].index)
cleaned_attempts_1 = cleaned_attempts_1.reset_index(drop=True)

''' Research Question 2 
Predictors: substrate, host species
Response: # of parasite eggs '''
# Clean the data
cleaned_attempts_2 = clean_dataset(nest_attempts_2)

# Drop outliers and clutch sizes of 0
cleaned_attempts_2 = cleaned_attempts_2[cleaned_attempts_2['Clutch Size'] > 0] # take out the entries where Clutch Size = 0

# Check outliers
cleaned_attempts_2['Young Fledged'].sort_values(ascending=False).head(10)
cleaned_attempts_2.iloc[30401] # example of finding attempt IDs for each attempt
cleaned_attempts_2 = cleaned_attempts_2.drop(cleaned_attempts_2[cleaned_attempts_2['Attempt ID']
                                                                .isin(['A1468450', 'A1443815', 'A1410605', 'A1614433', 'A1410608']) == True].index)
cleaned_attempts_2 = cleaned_attempts_2.reset_index(drop=True)

# To incorporate the parasite count variable, we need to:
# 1) disregard all the rows that don't have a value for Brood Parasite Eggs Count
# 2) get only the unique attempt IDs for those
# 3) join with the nest_attempts_df on attempt ID
parasites_counted_df = nest_visits_df[nest_visits_df['Brood Parasite Eggs Count'].notnull()]
unique_parasite_eggs_counted_df = parasites_counted_df.drop_duplicates(subset='Attempt ID', ignore_index=True)
unique_parasites_counted_df_2 = unique_parasite_eggs_counted_df[['Attempt ID', 'Brood Parasite Eggs Count']]

# joined dataframe
attempts_with_parasite_count_2 = pd.merge(cleaned_attempts_2, unique_parasites_counted_df_2, how='inner', on = 'Attempt ID')

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

# Clean the data and join it with the parasites dataframe
cleaned_attempts_3 = clean_dataset(nest_attempts_3)

# Similar to question 1 we will take out the extreme outliers for the Young Fledged column
#       Attempt ID Species Code  Year  Young Fledged  Clutch Size
# 41529   A1443815       purmar  2021           62.0         71.0
# 32306   A1468450       easblu  2021           53.0          5.0
# 26977   A1410605       purmar  2020           47.0         48.0
# 70936   A1614433       easblu  2023           44.0          5.0
# 19072   A1410608       purmar  2020           42.0         46.0
cleaned_attempts_3 = cleaned_attempts_3.drop(cleaned_attempts_3[cleaned_attempts_3['Attempt ID'].isin(['A1443815', 'A1468450', 'A1410605', 'A1614433', 'A1410608']) == True].index)
attempts_with_parasite_count_3 = pd.merge(cleaned_attempts_3, unique_parasites_counted_df, how='inner', on = 'Attempt ID')