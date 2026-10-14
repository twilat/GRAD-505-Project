# Before running the code here, run dataset set up file to set up the variables here

''' Research Question 1 '''
cleaned_attempts_1.head()
print(f"Dataset count (question 1): {cleaned_attempts_1.count()}")

# visualization 1 (histogram for habitat)
habitat_freq = cleaned_attempts_1[cleaned_attempts_1['Habitat 1m'].notnull()]['Habitat 1m'].value_counts()
habitat_freq.plot(kind='bar', color='orange')
plt.title("Histogram of Habitat 1m")
plt.show()

# summary table 1 (# young fledged)
cleaned_attempts_1[cleaned_attempts_1['Young Fledged'] == cleaned_attempts_1['Young Fledged'].max()]
cleaned_attempts_1['Young Fledged'].describe()

''' Research Question 2 '''
# frequency table (species)
species_freq = cleaned_attempts_2[cleaned_attempts_2['Species Code'].notnull()]['Species Code'].value_counts()
print(f"There are {species_freq.count()} species after filtering out the nulls, too many to display in a histogram.")

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
