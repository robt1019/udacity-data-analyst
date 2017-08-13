import numpy as np
import pandas as pd

titanic_df = pd.read_csv('./titanic-data.csv')

# rename columns
titanic_df.columns = ['passengerid', 'survived', 'se_class', 'name', 'sex', 'age', 'siblings_spouses_count', \
'parents_children_count', 'ticket', 'fare', 'cabin', 'embarcation_point']

# show initial survival rates
titanic_by_survived = titanic_df.groupby('survived')
print '\nsurvival rate overall'
print titanic_by_survived.size()

def survival_rate_by(data_frame, dimension):
    data_frame_by_dimension = data_frame.groupby(dimension)
    print '\ntitanic ' + dimension + ' split'
    print data_frame_by_dimension.size() 
    print '\ntitanic survival by ' + dimension
    print data_frame_by_dimension['survived'].mean()
    return data_frame_by_dimension['survived'].mean()

# se class survival basic analysis
survival_rate_by(titanic_df, 'se_class')

# gender survival basic analysis
survival_rate_by(titanic_df, 'sex')

# age survival basic analysis
# remove null values from age. Base is still 714. Seems reasonable for analysis
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
survival_rate_by(titanic_df, 'age_binned')

# look at impact of being part of a family on survival
def part_of_family(index):
    if titanic_df.iloc[index]['siblings_spouses_count'] > 0 or titanic_df.iloc[index]['parents_children_count'] > 0:
        return True
    return False

titanic_by_part_of_family = titanic_df.groupby(part_of_family)
print '\ntitanic part of family split'
print titanic_by_part_of_family.size()
print '\ntitanic survival by part of family'
print titanic_by_part_of_family['survived'].mean()