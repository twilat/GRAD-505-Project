import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.graphics.regressionplots import abline_plot
from matplotlib.ticker import (MultipleLocator)

''' Initial data setup and cleanup'''

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
                print(f"There are {value_counts[True]} attempts that are missing data for {column} column")
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


''' Preliminary visualizations from Project Deliverable 2'''

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

''' Hypothesis tests and final visualizations '''
''' Research question 1: What is the effect of the nesting conditions on the outcome of the nest attempt?
H0: The means of young fledged are equal across each habitat. (In other words, the type of habitat has no effect 
on how many young fledged).
Ha: The means of young fledged are not equal across each habitat.

Statistical test: ANOVA test'''

'''Perform statistical test (ANOVA test)'''
cleaned_attempts_1 = cleaned_attempts_1.rename(columns={"Habitat 1m": "Habitat", "Young Fledged": "Young_Fledged"})
Habitat = cleaned_attempts_1['Habitat']
Young_Fledged = cleaned_attempts_1['Young_Fledged']

# Get dummies for the categorical variable Habitat
df_1 = pd.get_dummies(cleaned_attempts_1, columns=['Habitat'], drop_first=True) # drop one category to avoid multicollinearity
cols_1 = df_1.columns.unique()

# Convert dummy variables to 1 and 0
for habitat in cols_1[6:]:
# for habitat in cols_1[7:]:
    df_1[habitat] = df_1[habitat].astype(int)

# Perform ANOVA
model = ols('Young_Fledged ~ C(Habitat)', data=df_1).fit()
anova_results = anova_lm(model)
print(anova_results)

'''Verify assumptions for statistical test (ANOVA test)'''
# Extract residuals and fitted values from the model
residuals = model.resid
fitted = model.fittedvalues

# Normality test: q-q plot
sm.qqplot(residuals, line="s")
plt.title("Q-Q Plot of ANOVA Residuals")
plt.show()

# Independence test: plot residuals vs run/order
plt.scatter(df_1.index, residuals)
plt.xlabel('Run Order')
plt.ylabel('Residuals')
plt.title('Residuals vs Run Order')
plt.axhline(0, color='red', linestyle='-')
plt.savefig("Residuals_vs_RunOrder.svg")
plt.show()

# Equal variance test: plot residuals vs fitted values
plt.scatter(fitted, residuals)
plt.axhline(0, color='red', linestyle="--")
plt.title("Residuals vs Fitted")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.savefig("Residuals_vs_Fitted.svg")
plt.show()

# Final visualizations
fig, ax = plt.subplots(figsize=(10, 10))
boxplot = ax.boxplot(
    [group['Young_Fledged'] for _, group in cleaned_attempts_1.groupby('Habitat')],
    patch_artist=True,
    tick_labels=cleaned_attempts_1['Habitat'].unique()
)
ax.set_title("Boxplots of young fledged per habitat")
plt.show()


''' Research question 2: Is there a relationship between the nesting substrates, the host species, and the presence of 
parasites in the nest?
H0: There is no effect of the nest substrate and host species on the number of parasite eggs in the nesting attempt.
Ha: The nest substrate and host species have an effect on the number of parasite eggs in the nesting attempt.

Statistical test: Multiple regression '''

''' Perform statistical test (multiple regression)'''
# Rename columns without whitespace and save predictor variable
attempts_with_parasite_count_2 = attempts_with_parasite_count_2.rename(columns={"Species Code": "Species", "Brood Parasite Eggs Count": "Parasite_Eggs"})
Species = attempts_with_parasite_count_2['Species']
Substrate = attempts_with_parasite_count_2['Substrate']

# Get dummies for the categorical variable Habitat
df_2 = pd.get_dummies(attempts_with_parasite_count_2, columns=['Species', 'Substrate'], drop_first=True) # drop one category to avoid multicollinearity
cols_2 = df_2.columns.unique()

# Convert dummy variables to 1 and 0
for c in cols_2[5:]:
    df_2[c] = df_2[c].astype(int)

# Perform regression
X = df_2[df_2.columns[5:]]  # Predictor variables
y = df_2['Parasite_Eggs']   # Response variable
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()  # Fit the linear regression model

print(model.summary()) # r stat is 0.041

'''Verify assumptions for statistical test (multiple regression)'''
# Extract residuals and fitted values from the model
residuals = model.resid
predictions = model.predict(X)

# Normality test: q-q plot
sm.qqplot(residuals, line="s")
plt.title("Q-Q Plot of Residuals")
plt.show()

# Independence test: plot residuals vs run/order
plt.scatter(df_2.index, residuals)
plt.xlabel('Run Order')
plt.ylabel('Residuals')
plt.title('Run Order vs. Residuals')
plt.axhline(0, color='red', linestyle='-')
plt.savefig("Residuals_vs_RunOrder.svg")
plt.show()

# Constant variance test: plot predictions vs. residuals
plt.scatter(predictions, residuals)
plt.title('Predicted Values vs. Residuals')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

# Final visualizations

# Substrate vs. Mean Parasite Eggs
mean_parasite_eggs = [substrate['Parasite_Eggs'].mean() for _, substrate in attempts_with_parasite_count_2.groupby('Substrate')]

plt.scatter(attempts_with_parasite_count_2['Substrate'].unique(), mean_parasite_eggs)
plt.title('Substrate vs. Mean Parasite Eggs')
plt.xlabel('Substrate')
plt.ylabel('Mean Parasite Eggs')
plt.show()

# Species vs. Mean Parasite Eggs
fig, ax = plt.subplots()
ax.scatter(species_count, mean_parasite_eggs)
ax.xaxis.set_major_locator(MultipleLocator(57))
ax.xaxis.set_major_formatter('{x:.0f}')
ax.xaxis.set_minor_locator(MultipleLocator())
ax.set_xlim(0, 172)
ax.set_title('Species vs. Mean Parasite Eggs')
ax.set_xlabel('Species Index (Full Names Too Long to Display)')
ax.set_ylabel('Mean Parasite Eggs')
plt.show()

''' Research question 3: Does the presence of parasites in the nest have an effect on the final outcome of the nest attempt?
H0: There is no correlation between the presence of parasites and the number of young fledged.
Ha: There is a correlation between the presence of parasites and the number of young fledged.

Statistical test: Simple linear regression '''

'''Perform statistical test (simple lienar regression)'''
# Rename columns without whitespace and save predictor variables
df_3 = attempts_with_parasite_count_3.rename(columns={"Young Fledged": "Young_Fledged", "Brood Parasite Live Young Count": "Parasite_Young"})
Young_Fledged = df_3['Young_Fledged']
Parasite_Young = df_3['Parasite_Young']

# Perform regression
X = df_3['Parasite_Young']  # Predictor variables
y = df_3['Young_Fledged']   # Response variable
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()  # Fit the linear regression model

print(model.summary())

# Verify assumptions (simple linear regression)
residuals = model.resid
predictions = model.predict(X)

# Normality test: q-q plot
sm.qqplot(residuals, line="s")
plt.title("Q-Q Plot of Residuals")
plt.show()

# Independence test: plot residuals vs run/order
plt.scatter(df_3.index, residuals)
plt.xlabel('Run Order')
plt.ylabel('Residuals')
plt.title('Residuals vs Run Order')
plt.axhline(0, color='red', linestyle='-')
plt.savefig("Residuals_vs_RunOrder.svg")
plt.show()

# Constant variance test: plot predictions vs. residuals
plt.scatter(predictions, residuals)
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

# Final visualizations - plot out the results visually
df = pd.DataFrame({'Young_Fledged': Young_Fledged, 'Parasite_Young': Parasite_Young})
ax = df.plot(x='Parasite_Young', y='Young_Fledged', kind="scatter", edgecolor="k")
abline_plot(model_results=model, ax=ax, color="red")
plt.title("Parasite Young vs. Young Fledged")
plt.show()