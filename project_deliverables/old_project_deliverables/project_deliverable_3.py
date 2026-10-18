import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, MaxNLocator)
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.graphics.regressionplots import abline_plot
# from dataset_setup import (
#     cleaned_attempts_1,
#     attempts_with_parasite_count_2, 
#     attempts_with_parasite_count_3
# )

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
# Conclusion TODO: Kind of normal but there are fat tails, would central limit theorem still apply here?
# Perhaps I could use a different test more fitted for skewed data

# Independence test: plot residuals vs run/order
plt.scatter(df_1.index, residuals)
plt.xlabel('Run Order')
plt.ylabel('Residuals')
plt.title('Residuals vs Run Order')
plt.axhline(0, color='red', linestyle='-')
plt.savefig("Residuals_vs_RunOrder.svg")
plt.show()
# Conclusion TODO: Generally there's random scatter around the 0 line

# Equal variance test: plot residuals vs fitted values
plt.scatter(fitted, residuals)
plt.axhline(0, color='red', linestyle="--")
plt.title("Residuals vs Fitted")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.savefig("Residuals_vs_Fitted.svg")
plt.show()
# Conclusion TODO: Kind of no funnel shapes?

# Final visualizations # TODO
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

print(model.summary()) # r stat is 0.042

'''Verify assumptions for statistical test (multiple regression)'''
# Extract residuals and fitted values from the model
residuals = model.resid
predictions = model.predict(X)

# Normality test: q-q plot
sm.qqplot(residuals, line="s")
plt.title("Q-Q Plot of Residuals")
plt.show()
# Conclusion TODO: There are a ton of outliers

# Independence test: plot residuals vs run/order
plt.scatter(df_2.index, residuals)
plt.xlabel('Run Order')
plt.ylabel('Residuals')
plt.title('Run Order vs. Residuals')
plt.axhline(0, color='red', linestyle='-')
plt.savefig("Residuals_vs_RunOrder.svg")
plt.show()
# Conclusion TODO: Generally random

# Constant variance test: plot predictions vs. residuals
plt.scatter(predictions, residuals)
plt.title('Predicted Values vs. Residuals')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()
# Conclusion TODO: yeah not constant. very one sided
# probably not the best model to use

mean_parasite_eggs = [substrate['Parasite_Eggs'].mean() for _, substrate in attempts_with_parasite_count_2.groupby('Substrate')]

plt.scatter(attempts_with_parasite_count_2['Substrate'].unique(), mean_parasite_eggs)
plt.title('Substrate vs. Mean Parasite Eggs')
plt.xlabel('Substrate')
plt.ylabel('Mean Parasite Eggs')
plt.show()

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

# Plot out the results visually
df = pd.DataFrame({'Young_Fledged': Young_Fledged, 'Parasite_Young': Parasite_Young})
ax = df.plot(x='Parasite_Young', y='Young_Fledged', kind="scatter", edgecolor="k")
abline_plot(model_results=model, ax=ax, color="red")
plt.title("Parasite Young vs. Young Fledged")
plt.show()

# Verify assumptions (simple linear regression)
residuals = model.resid
predictions = model.predict(X)

# Normality test: q-q plot
sm.qqplot(residuals, line="s")
plt.title("Q-Q Plot of Residuals")
plt.show()
# Conclusion TODO: Pretty normal except for the outliers >?

# Independence test: plot residuals vs run/order
plt.scatter(df_3.index, residuals)
plt.xlabel('Run Order')
plt.ylabel('Residuals')
plt.title('Residuals vs Run Order')
plt.axhline(0, color='red', linestyle='-')
plt.savefig("Residuals_vs_RunOrder.svg")
plt.show()
# Conclusion TODO: kind of random

# Constant variance test: plot predictions vs. residuals
plt.scatter(predictions, residuals)
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()
# Conclusion TODO: There are lowkey funnel shapes

# Final visualizations TODO

# The last slope interval, .029 to .370, also excludes zero.