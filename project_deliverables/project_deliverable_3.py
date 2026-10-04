import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import statsmodels.api as sm

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
for habitat in cols_1[6:]:
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

# independence test
# Residuals vs run/order (look for patterns/trends)
plt.scatter(df_1.index, residuals)
plt.xlabel('Run Order')
plt.ylabel('Residuals')
plt.title('Residuals vs Run Order')
plt.axhline(0, color='red', linestyle='--')
plt.savefig("Residuals_vs_RunOrder.svg")
plt.show()

# equal variance test
# Residuals vs fitted values (look for “funnel” shapes)
plt.scatter(fitted, residuals)
plt.axhline(0, color='red', linestyle="--")
plt.title("Residuals vs Fitted")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.savefig("Residuals_vs_Fitted.svg")
plt.show()

# Final visualizations

'''
Research question 2: Is there a relationship between the nesting substrates, the host species, and the presence of parasites in the nest?
H0: There is no effect of the nest substrate and host species on the number of parasite eggs in the nesting attempt.
Statistical tests to conduct:
- Multiple regression
'''
# Verify assumptions (Multiple regression)
# normality (q-q plot of residuals)
# plot residuals vs. y
    # constant variance
    # mean of 0
    # independence
# check for any outliers/overly influential points

# Perform statistical test

# Final visualizations

'''
Research question 3: Does the presence of parasites in the nest have an effect on the final outcome of the nest attempt?
H0: There is no correlation between the presence of parasites and the amount of young fledged.
Statistical tests to conduct:
- Simple linear regression
'''
# Verify assumptions (Simple linear regression)
# normality (q-q plot of residuals)
# plot residuals vs. y
    # constant variance
    # mean of 0
    # independence
# check for any outliers/overly influential points

# Perform statistical test

# Final visualizations
