import pandas as pd


#################################
# 1. Read DataFrame  #
#################################

# Original data source: https://www.encyclopedia-titanica.org/

df = pd.read_csv('titanic.csv')

##################################
# 2. Basic overview              #
##################################

df.head()           # First five rows
df.tail(3)          # Last rows
df.info()           # Summary of DataFrame
df.describe()       # Descriptive Statistics
df.shape            # Number of rows and columns
df.columns          # Column names
df.index            # Row labels


###################################
# 3. Subset rows and cols         #
###################################

###########################
# iloc accessor
###########################
# Selection based on integer location
# Start-stop-step logic (same as numpy arrays)
df.iloc[0]               # single row
df.iloc[0:3]             # rows
df.iloc[0:10:2, 0:4]     # rows and columns
df.iloc[:, 0:4]           # all rows


###########################
# loc accessor
###########################
# Selection based on labels
df['fullname'] = df['first_name'] + " " + df['family_name']
df = df.set_index('fullname')

df.loc["Thomas Andrews"]                            # single row
df.loc[:, "price"]                         # single column
df.loc[:, "price":"survived"]              # multiple columns
df.loc[:, ["price", "survived"]]           # multiple columns

# Selection based on boolean mask
df.loc[df.age > 70, ["age", "survived"]]

#################################
# Shorter alternative without loc
#################################

# ... if entire columns are selected
df["age"]                                        # returns Series
df.age                                           # returns Series
df[["age"]]                                      # returns DataFrame

# ... if entire rows are selected using boolean masks
df[df.age > 70]                                  # condition
df[~(df.age > 70)]                               # negation of this condition
df[(df.age > 70) & (df.price < 8)]      # both conditions must be met
df[(df.age > 70) | (df.price < 8)]      # at least one must be met
df[df.departure.isin(["Cherbourg", "Queenstown"])]

# ... using query method
df.query("age > 70")
df.query("age > 70 and price < 8")
df.query("departure in ['Cherbourg', 'Queenstown']")


###################################
# 4. Sorting
###################################

df[["price"]].sort_values("price")
df[["price"]].sort_values("price", ascending=False)
df[['pclass', 'price']].sort_values(
    ['pclass', 'price'], ascending=[True, False])

# Useful shortcuts
df.price.nlargest(5)
df.price.nsmallest(5)

# Modify DataFrame permanently
df.sort_values('price', inplace=True)

# Sort according to index
df.sort_index()               # sorted copy of DataFrame
df.sort_index(inplace=True)   # modify DataFrame inplace


###################################
# 5. Aggregate
###################################

# Standard aggregation functions
df.price.mean()              # skip missing values
df.price.mean(skipna=False)  # do not skip missing values
df.price.min()
df.price.sum()

# ... applied on entire DataFrame
df.mean()
df.mean(numeric_only=True)

###########################
# agg method
###########################

# Apply list of aggregation functions
df.age.agg(["min", "mean", "max"])

# Apply dictionary of aggregation functions
functions_dict = {"age": "mean", "price": ["mean", "sum"]}
df.agg(functions_dict)

# Apply custom function
df[["age", "price"]].agg(lambda x: x.max() - x.min())


###################################
# 6. Groupby method
###################################

# Group by single column
df.groupby('gender').size()         # number of passengers by gender
# number of non-missing age values by gender
df.groupby('gender').age.count()
df.groupby('gender').age.mean()     # average age per gender

# ... multiple columns
df.groupby(['gender', 'pclass']).size()

# ... according to condition
df.groupby(df.age > 70).size()

# Useful shortcuts
df.value_counts('gender')
df.value_counts('gender', normalize=True, sort=False)

# Store groupby object
df_gender = df.groupby('gender')
type(df_gender)
df_gender.groups.keys()
df_gender.get_group('Female').age
