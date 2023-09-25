import pandas as pd
import scipy.stats as stats
import statsmodels
import statsmodels.api 
import statsmodels.formula.api as smf

dice1 = pd.Series([2, 3, 5, 4, 4, 3])
dice2 = pd.Series([4, 2, 3, 5, 2, 3])
dice3 = pd.Series([3, 1, 4, 4, 3, 5])

#test all for normality with shapiro
print(stats.shapiro(dice1))
print(stats.shapiro(dice2))
print(stats.shapiro(dice3))

#modify data for anova
data = pd.DataFrame({'dice1': dice1, 'dice2': dice2, 'dice3': dice3})
modified_data = pd.melt(data)
modified_data.columns = ['dice', 'value']

#creaete model 
model = smf.ols('value ~ C(dice)', data=modified_data).fit()

#ANOVA
anova = statsmodels.api.stats.anova_lm(model, typ=1)
print(anova)

countries1 = pd.Series(['Fi','Sw','Fi','No','Sw','Fi'])
countries2 = pd.Series(['No','Sw','No','Fi','Fi','Fi'])
countries3 = pd.Series(['Sw','Fi','No','Sw','Sw','No'])

#create a dataframe of occurances
data = pd.DataFrame({'countries1': countries1, 'countries2': countries2, 'countries3': countries3})
data = data.apply(pd.value_counts)

#Pearson's chi-squared test on the data
print("P-Value:" ,stats.chi2_contingency(data)[1])