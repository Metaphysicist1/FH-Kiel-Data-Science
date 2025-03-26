import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('titanic.csv')
df.head()
df.shape

########################
# 1. Seaborn
########################

################
# Scatterplot
################


# Fixed properties of the scatter points
sns.scatterplot(data=df,
                x='age',
                y='price',
                c='red',     # fixed color choice
                s=50,        # fixed size choice
                marker='x')  # fixed marker choice


# Mapping columns to visual properties
sns.scatterplot(data=df,
                x='age',
                y='price',
                hue='survived',  # color mapping
                size='pclass',   # size mapping
                style='gender')  # marker style mapping


################################
# Other axes-level functions
################################
# Histogram
sns.histplot(data=df, x='age', binwidth=10)
sns.histplot(data=df, x='age', bins=100)

# Barplot
data = df.groupby('pclass').survived.mean().reset_index()
sns.barplot(data=data, x='pclass', y='survived')

################################
# Figure-level functions
################################

sns.relplot(data=df, x='age', y='price', col='survived',  kind='scatter')
sns.jointplot(data=df, x='age', y='price', kind='scatter')


fig, ax = plt.subplots(nrows=2, ncols=2)
sns.scatterplot(df, x='age', y='price', ax=ax[0, 1])
fig.tight_layout()


##########################################################
# 2. Implicit Pyplot interface vs Explicit Axes interface
##########################################################

# Implicit Pyplot interface
plt.figure(figsize=(10, 5))
x = sns.scatterplot(data=df, x='age', y='price')
plt.title("Age vs price Scatterplot")
plt.xlabel("Age")
plt.ylabel("price")
plt.savefig('scatterplot.svg')

# Explict Axes interface
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=df, x='age', y='price', hue='survived', ax=ax)
ax.set_title("Age vs price Scatterplot")
ax.set_xlabel("Age")
ax.set_ylabel("price")
fig.savefig('scatterplot_oo.svg')


################################
# Arranging multiple plots
################################
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
fig.suptitle('Age vs price Scatterplot')
sns.scatterplot(data=df, x='age', y='price', c='red', ax=ax[0])
sns.histplot(data=df, x='age', ax=ax[1])
ax[1].set_xlabel("Age")
fig.savefig('myfirstfigure.png')


########################
# 3. Pandas
########################

# Histogram

df.age.plot(kind='hist')
df.age.plot.hist(bins=30)

# Bar plot
passengers_per_class = df.groupby("pclass").size()
passengers_per_class.plot.bar()

# Plotting on grouped data
df.groupby('gender').plot.scatter(x='age', y='price')

# Overlaying multiple plots
fig, ax = plt.subplots(figsize=(5, 5))
df[df.survived == 0].plot.scatter(x='age', y='price', c='red', ax=ax)
df[df.survived == 1].plot.scatter(x='age', y='price', c='blue', ax=ax)

# Arranging multiple plots
fig, ax = plt.subplots(nrows=1, ncols=2)
df[df.survived == 0].plot.scatter(x='age', y='price', c='red', ax=ax[0])
df[df.survived == 1].plot.scatter(x='age', y='price', c='blue', ax=ax[1])
