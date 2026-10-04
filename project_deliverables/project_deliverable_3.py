import pandas as pd
import matplotlib.pyplot as plt

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
# Verify assumptions (ANOVA)
# Each sample is random and independent from the others
# Each population is normally distributed (q-q plot)
# The standard deviations of the populations are similar

# Perform statistical test

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
