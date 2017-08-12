import numpy as np
import pandas as pd

titanic_df = pd.read_csv('./titanic-data.csv')

titanic_df.columns = ['passengerid', 'survived', 'se_class', 'name', 'sex', 'age', 'siblings_spouses_count', \
'parents_children_count', 'ticket', 'fare', 'cabin', 'embarcation_point']

titanic_by_survived = titanic_df.groupby('survived')
print '\ntitanic survival split'
print titanic_by_survived.size()

titanic_by_class = titanic_df.groupby('se_class')

print '\n titanic class split'
print titanic_by_class.size()

class_survival = titanic_by_class['survived'].mean()
print '\nsurvival by class'
print class_survival

titanic_by_gender = titanic_df.groupby('sex')

print '\ntitanic gender split'
print titanic_by_gender.size()

gender_survival = titanic_by_gender['survived'].mean()
print '\nsurvival by gender'
print gender_survival


# show distribution of ages. Because useful
# remove null values from age. Base is still 714
titanic_without_null_ages = titanic_df.dropna(subset=['age'])
print '\ntitanic without null ages base: '
print len(titanic_without_null_ages )
# print titanic_without_null_ages['age'].value_counts(normalize=True, sort=True)

# age_bins = [0, 5, ]
# age_bin_labels = ['0-5', '6-15', '16-21', '21-30', '31-45', '46-55', '55+']

# titanic_age_binned = pd.qcut(titanic_without_null_ages['age'], q=4, labels=['1', '2', '3', '4'])

titanic_age_binned = pd.cut(titanic_without_null_ages['age'], bins=5, labels=['first', 'second', 'third', 'fourth', 'fifth'])
print titanic_age_binned.value_counts()

# print '\nsurvival by age'
