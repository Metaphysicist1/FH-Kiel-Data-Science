import pandas as pd

###########################
# 1. Read Data
###########################

fruits = pd.read_csv('fact_fruits.csv')
vegetables = pd.read_csv('fact_vegetables.csv')
metadata = pd.read_csv('dim_products.csv')

###########################
# 2. Combine datasets
###########################

##########
# Concat
##########

groceries = pd.concat([fruits, vegetables], ignore_index=True)   # axis=0

##########
# Merge
##########

# 1. Outer Merge - with checks
check = pd.merge(groceries, metadata, on='product_id',
                 how='outer', indicator=True)

# Check number of rows before and after
print(groceries.shape)
print(metadata.shape)
print(check.shape)

# Check from which dataset each row comes from
check['_merge'].value_counts()
check[check['_merge'] == 'right_only']

# 2. Pick the merge type according to the requirements

df = pd.merge(groceries, metadata, on='product_id', how='left')
print(df.shape)


###########################
# 3. Reshape data
###########################

#####################
# Pivot and melt
#####################

# Pivot: turn into wide format
wide = df.pivot(index='name', columns='date', values='amount')
df

# Melt: turn into long format
wide.reset_index(inplace=True)
long = wide.melt(
    id_vars='name', var_name='date', value_name='amount')

#####################
# Unstack and stack
#####################

# Same as pivot and melt, but using indices explicitly
df_indexed = df.set_index(['name', 'date'])
wide2 = df_indexed['amount'].unstack()
long2 = wide2.stack()
long2.reset_index()


###########################
# 4. Manipulate columns
###########################

df['revenue'] = df['amount'] * df['price']                 # Add new column
df.drop(columns=['remarks'], inplace=True)                 # Drop columns
df.rename(columns={'name': 'product_name'}, inplace=True)  # Rename column

################
# Strings
################

df['is_organic'] = df['tags'].str.contains('organic')
df.product_name.str.len()

#################
# Dates and Times
#################
# Convert date and extract weekday
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['dayname'] = df['date'].dt.day_name()


###########################
# 5. Missing values
###########################


# Identify missing values ...
df.isna().sum(axis=0)  # per column
df.isna().sum(axis=1)  # per row

# Dropping missing values
df.dropna(subset=['amount', 'tags'])   # in one of the specified columns

# Fill missing amounts with 0
df['amount'] = df['amount'].fillna(0)


###########################
# 6. Complex transformations
###########################

#################
# Transform
#################

# Compute total revenue per product type, and append to each row
df['total_revenue_per_type'] = df.groupby(
    'product_type')['revenue'].transform('sum')


#################
# Map
#################

# Rename data values using a dictionary
df['product_type_code'] = df['product_type'].map(
    {'fruit': 'F', 'vegetable': 'V'})

# Carry out a function element-wise
df['origin_country'] = df['origin'].map(lambda x: x.split(',')[0].strip())


#################
# Apply
#################
# Apply custom function to each row (or column)

def classify(row):
    if row['price'] > 3 and row['is_organic']:
        return 'premium'
    else:
        return 'standard'
