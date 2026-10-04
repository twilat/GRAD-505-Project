import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.graphics.regressionplots import abline_plot

# TODO: rerun all the code and modify the comments because forgot to include 2020

# Set up data first (using code from dataset_setup.py). The dataset names are:
# cleaned_attempts_1
# cleaned_attempts_2
# cleaned_attempts_3

'''
Research question 1: What is the effect of the nesting conditions on the outcome of the nest attempt?
H0: The means of young fledged are equal across each habitat.
(In other words, the type of habitat has no effect on how many young fledged).
Statistical tests to conduct:
- ANOVA test
'''
# Perform statistical test
cleaned_attempts_1 = cleaned_attempts_1.rename(columns={"Habitat 1m": "Habitat", "Proportion Fledged": "Proportion_Fledged"})
Habitat = cleaned_attempts_1['Habitat']
Proportion_Fledged = cleaned_attempts_1['Proportion_Fledged']

# Get dummies for the categorical variable Habitat
df_1 = pd.get_dummies(cleaned_attempts_1, columns=['Habitat'], drop_first=True) # drop one category to avoid multicollinearity
cols_1 = df_1.columns.unique()

# Convert dummy variables to 1 and 0
for habitat in cols_1[7:]:
    df_1[habitat] = df_1[habitat].astype(int)

# Perform ANOVA
model = ols('Proportion_Fledged ~ C(Habitat)', data=df_1).fit()
anova_results = anova_lm(model)
print(anova_results)

# Verify assumptions (ANOVA)
# Extract residuals and fitted values from our final model
residuals = model.resid
fitted = model.fittedvalues

# normality test
sm.qqplot(residuals, line="s")
plt.title("Q-Q Plot of ANOVA Residuals")
plt.show()
# Kind of normal but there are fat tails, would central limit theorem still apply here?
# Perhaps I could use a different test more fitted for skewed data

# independence test
# Residuals vs run/order (look for patterns/trends)
plt.scatter(df_1.index, residuals)
plt.xlabel('Run Order')
plt.ylabel('Residuals')
plt.title('Residuals vs Run Order')
plt.axhline(0, color='red', linestyle='-')
plt.savefig("Residuals_vs_RunOrder.svg")
plt.show()
# There are two outliers, but generally there's random scatter around the 0 line

# equal variance test
# Residuals vs fitted values (look for “funnel” shapes)
plt.scatter(fitted, residuals)
plt.axhline(0, color='red', linestyle="--")
plt.title("Residuals vs Fitted")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.savefig("Residuals_vs_Fitted.svg")
plt.show()
# One outlier but otherwise no funnel shapes

# Final visualizations # TODO

'''
Research question 2: Is there a relationship between the nesting substrates, the host species, and the presence of parasites in the nest?
H0: There is no effect of the nest substrate and host species on the number of parasite eggs in the nesting attempt.
Statistical tests to conduct:
- Multiple regression
'''
attempts_with_parasite_count_2 = attempts_with_parasite_count_2.rename(columns={"Species Code": "Species", "Brood Parasite Eggs Count": "Parasite_Eggs"})
Species = attempts_with_parasite_count_2['Species']

# Get dummies for the categorical variable Habitat
df_2 = pd.get_dummies(attempts_with_parasite_count_2, columns=['Species', 'Substrate'], drop_first=True) # drop one category to avoid multicollinearity
cols_2 = df_2.columns.unique()

# Convert dummy variables to 1 and 0
for c in cols_2[5:]:
    df_2[c] = df_2[c].astype(int)

# Perform regression
X = df_2[df_2.columns[5:]]  # Predictor variables
y = df_2['Parasite_Eggs']  # Response variable

X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

print(model.summary()) # r stat is 0.042. if just looking at species alone it is 0.028 TODO, and substrate alone is 0.01 TODO

# Verify assumptions (multiple regression)
# Extract residuals and fitted values from our final model
residuals = model.resid
predictions = model.predict(X)

# normality test
sm.qqplot(residuals, line="s")
plt.title("Q-Q Plot of Residuals")
plt.show()
# Pretty normal except for the outliers >?

# independence test
# Residuals vs run/order (look for patterns/trends)
plt.scatter(df_2.index, residuals)
plt.xlabel('Run Order')
plt.ylabel('Residuals')
# plt.ylim(-1, 1) # take out this limit after showing the outlier
plt.title('Residuals vs Run Order')
plt.axhline(0, color='red', linestyle='-')
plt.savefig("Residuals_vs_RunOrder.svg")
plt.show()
# Generally random

# constant variance test
# Scatter plot for residuals (for visual inspection)
plt.scatter(predictions, residuals)
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()
# yeah not constant. very one sided
# probably not the best model to use

# Final visualizations # TODO

'''
Research question 3: Does the presence of parasites in the nest have an effect on the final outcome of the nest attempt?
H0: There is no correlation between the presence of parasites and the amount of young fledged.
Statistical tests to conduct:
- Simple linear regression
'''
# Perform statistical test
df_3 = attempts_with_parasite_count_3.rename(columns={"Young Fledged": "Young_Fledged", "Brood Parasite Live Young Count": "Parasite_Young"})
Young_Fledged = df_3['Young_Fledged']
Parasite_Young = df_3['Parasite_Young']

# Define the response variable and predictors
X = df_3['Parasite_Young']   # Predictor variables
y = df_3['Young_Fledged']  # Response variable

# Add a constant to the predictors (intercept term)
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the summary (similar to R's summary(lm(...)))
print(model.summary())

# plot
df = pd.DataFrame({'Young_Fledged': Young_Fledged, 'Parasite_Young': Parasite_Young})
ax = df.plot(x='Parasite_Young', y='Young_Fledged', kind="scatter", edgecolor="k")
abline_plot(model_results=results, ax=ax, color="red")
plt.title("Parasite Young vs. Young Fledged")
plt.show()

# Verify assumptions (Simple linear regression)
residuals = model.resid
predictions = model.predict(X)

# normality test
sm.qqplot(residuals, line="s")
plt.title("Q-Q Plot of Residuals")
plt.show()
# Pretty normal except for the outliers >?

# independence test
# Residuals vs run/order (look for patterns/trends)
plt.scatter(df_3.index, residuals)
plt.xlabel('Run Order')
plt.ylabel('Residuals')
plt.title('Residuals vs Run Order')
plt.axhline(0, color='red', linestyle='-')
plt.savefig("Residuals_vs_RunOrder.svg")
plt.show()
# kind of random

# constant variance test
# Scatter plot for residuals (for visual inspection)
plt.scatter(predictions, residuals)
plt.title('Residuals vs Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()
# funnel shapes

# check for any outliers/overly influential points


# Final visualizations
