import numpy as np
import pandas as pd

titanic_df = pd.read_csv('./titanic-data.csv')

# rename columns
titanic_df.columns = ['passengerid', 'survived', 'se_class', 'name', 'sex', 'age', 'siblings_spouses_count', \
'parents_children_count', 'ticket', 'fare', 'cabin', 'embarcation_point']

# show initial survival rates
print '\n survival rate overall'
titanic_by_survived = titanic_df.groupby('survived')
print '\ntitanic survival split'
print titanic_by_survived.size()


# se class survival basic analysis
titanic_by_class = titanic_df.groupby('se_class')
print '\n titanic class split'
print titanic_by_class.size()
class_survival = titanic_by_class['survived'].mean()
print '\nsurvival by class'
print class_survival

# gender survival basic analysis
titanic_by_gender = titanic_df.groupby('sex')
print '\ntitanic gender split'
print titanic_by_gender.size()
gender_survival = titanic_by_gender['survived'].mean()
print '\nsurvival by gender'
print gender_survival

# age survival basic analysis
# remove null values from age. Base is still 714
titanic_without_null_ages = titanic_df.dropna(subset=['age'])
print '\ntitanic without null ages base: '
print len(titanic_without_null_ages )

# used this to figure out a good split for bins. Those returned by qcut are [ 0.42,  19.  ,  25.  ,  31.8 ,  41.  ,  80.  ]
titanic_age_equal_bin_labels = ['first', 'second', 'third', 'fourth', 'fifth']
titanic_age_equal_bins = pd.qcut(titanic_without_null_ages['age'], q=5, retbins=True, labels=titanic_age_equal_bin_labels)
print '\nmost equal equal age bin sizes'
print titanic_age_equal_bin_labels
print titanic_age_equal_bins[1]
titanic_age_equal_bins = pd.qcut(titanic_without_null_ages['age'], q=5, labels=titanic_age_equal_bin_labels)
print '\nbin size counts - seem like reasonable bases'
print titanic_age_equal_bins.value_counts()

# split ages into sensible bins based on above
titanic_age_bins = [0, 18, 25, 30, 40, 80]
titanic_age_bin_labels = ['0-18', '18-25','25-30', '30-40', '40-80']
titanic_df['age_binned'] = pd.cut(titanic_without_null_ages['age'], titanic_age_bins, labels=titanic_age_bin_labels)

titanic_by_age = titanic_df.groupby('age_binned')
print '\ntitanic age split'
print titanic_by_age.size()
print '\nsurvival by age'
print titanic_by_age['survived'].mean()
