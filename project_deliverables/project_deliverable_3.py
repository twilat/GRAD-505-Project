import pandas as pd
import matplotlib.pyplot as plt

'''
Research question 1: What is the effect of the nesting conditions on the outcome of the nest attempt?
H0: The means of young fledged are equal across each habitat and substrate.
Statistical tests to conduct:
- two-way ANOVA test
- multiple regression (optional)
'''
# Verify assumptions (ANOVA)
# Each sample is random and independent from the others
# Each population is normally distributed (q-q plot)
# The standard deviations of the populations are similar

# Perform statistical test

# Final visualizations

'''
Research question 2: Is there a relationship between the nesting substrates, the host species, and the presence of parasites in the nest?
H0: The means of parasite egg counts are equal across each substrate and species.
Statistical tests to conduct:
- Regression line (with dummies) (multiple regression)
- two-way ANOVA test (optional)
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
H0: The presence of parasites and the amount of young fledged are correlated.
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
