import pandas as pd
import matplotlib.pyplot as plt

# Perform exploratory data analysis
# Identify any issues that exist with your dataset (e.g. missing or erroneous data)
# Assess whether your dataset can effectively answer your research questions
# Create preliminary visualizations addressing your questions

# read csv into a Pandas dataframe
nest_attempts_df = pd.read_csv('../dataset_files/attempts_locs-20230518.csv')
nest_visits_df = pd.read_csv('../dataset_files/checks-20230518.csv')

''' Initial Data Analysis '''
# Show the unique columns and select relevant columns for each question, will add Parasite column later for relevant question
nest_attempts_df.columns.unique()
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

''' Research Question 1 '''
# Clean the data
cleaned_attempts_1 = clean_dataset(nest_attempts_1)
cleaned_attempts_1.head()

# visualization 1 (histogram for habitat)
habitat_freq = cleaned_attempts_1[cleaned_attempts_1['Habitat 1m'].notnull()]['Habitat 1m'].value_counts()
habitat_freq.plot(kind='bar')
plt.title("Histogram of Habitat 1m")
plt.show()

# visualization 2 (histogram for substrate)
substrate_freq = cleaned_attempts_1[cleaned_attempts_1['Substrate'].notnull()]['Substrate'].value_counts()
substrate_freq.plot(kind='bar')
plt.title("Histogram of Substrate")
plt.show()

# summary table 1 (# young fledged / clutch size)
# I want to take the proportion of young fledged instead of just the raw number, since different nests may have
# different numbers of eggs, and this may differ by species
cleaned_attempts_1['Proportion Fledged'] = cleaned_attempts_1['Young Fledged'] / cleaned_attempts_1['Clutch Size']
cleaned_attempts_1[cleaned_attempts_1['Proportion Fledged'] == cleaned_attempts_1['Proportion Fledged'].max()]
cleaned_attempts_1['Proportion Fledged'].describe()
# This initial describe shows us that the max is infinity for some. Upon closer inspection, we see
# that some entries report clutch sizes of 0 but also have a number > 0 for the Young Fledged.
# This appears to be user error, and is causing a divide by 0 error, which is why the mean and max
# appear to be infinity.

# We will take out the entries where Clutch Size is equal to 0.
cleaned_attempts_1 = cleaned_attempts_1[cleaned_attempts_1['Clutch Size'] > 0]
cleaned_attempts_1['Proportion Fledged'] = cleaned_attempts_1['Young Fledged'] / cleaned_attempts_1['Clutch Size']
cleaned_attempts_1[cleaned_attempts_1['Proportion Fledged'] == cleaned_attempts_1['Proportion Fledged'].max()]
cleaned_attempts_1['Proportion Fledged'].describe()

''' Research Question 2 '''
# Clean the data
cleaned_attempts_2 = clean_dataset(nest_attempts_2)

# frequency table (species)
species_freq = cleaned_attempts_2[cleaned_attempts_2['Species Code'].notnull()]['Species Code'].value_counts()
print(f"There are {species_freq.count()} species after filtering out the nulls, too many to display in a histogram.")

# To incorporate the parasite count variable, we need to:
# 1) disregard all the rows that don't have a value for Brood Parasite Eggs Count
# 2) get only the unique attempt IDs for those
# 3) join with the nest_attempts_df on attempt ID
parasites_counted_df = nest_visits_df[nest_visits_df['Brood Parasite Eggs Count'].notnull()]
unique_parasite_eggs_counted_df = parasites_counted_df.drop_duplicates(subset='Attempt ID', ignore_index=True)
unique_parasites_counted_df = unique_parasite_eggs_counted_df[['Attempt ID', 'Brood Parasite Eggs Count']]

# joined dataframe
attempts_with_parasite_count_2 = pd.merge(cleaned_attempts_2, unique_parasite_eggs_counted_df, how='inner', on = 'Attempt ID')

# summary table 2, 3 (parasites)
attempts_with_parasite_count_2['Brood Parasite Eggs Count'].describe()
attempts_with_parasite_count_2['Young Fledged'].describe()

# visualization 3 (box plot of Brood Parasite Eggs Count)
plt.boxplot(attempts_with_parasite_count_2['Brood Parasite Eggs Count'])
plt.title('Brood Parasite Eggs Count')
plt.show()

# visualization 4 (young fledged)
plt.boxplot(attempts_with_parasite_count_2['Young Fledged'])
plt.title('Young Fledged')
plt.show()


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

# summary table 4, 5
attempts_with_parasite_count_3['Brood Parasite Live Young Count'].describe()
attempts_with_parasite_count_3['Young Fledged'].describe()

# visualization 5 (box plot of Brood Parasite Live Young Count)
plt.boxplot(attempts_with_parasite_count_3['Brood Parasite Live Young Count'])
plt.title('Brood Parasite Live Young Count')
plt.show()

# create visualization 4 (parasite young count vs. young fledged)
plt.scatter(x=attempts_with_parasite_count_3['Brood Parasite Live Young Count'], y=attempts_with_parasite_count_3['Young Fledged'])
plt.xlabel('Brood Parasite Live Young Count')
plt.ylabel('Young Fledged')
plt.show()

# We can also do a proportion of the parasite young count out of the young total. TODO
