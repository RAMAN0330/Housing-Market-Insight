import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Load the dataset
train_file_path = "train.csv"
dataset_df = pd.read_csv(train_file_path)
print("Full train dataset shape is {}".format(dataset_df.shape))


# The data is composed of 81 columns and 1460 entries. We can see all 81 dimensions of our dataset by printing out the first 3 entries using the following code:
dataset_df.head(3)


# The dataset comprises 79 feature columns. Utilizing these features, your model is required to predict the house sale price, as denoted by the label column named `SalePrice`.
# We will drop the `Id` column as it is not necessary for model training.

dataset_df = dataset_df.drop('Id', axis=1)
dataset_df.head(3)

# We can inspect the types of feature columns using the following code:
dataset_df.info()


# House Price Distribution
# Now let us take a look at how the house prices are distributed.

print(dataset_df['SalePrice'].describe())
plt.figure(figsize=(9, 8))
sns.distplot(dataset_df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4});


# Numerical data distribution
# We will now take a look at how the numerical features are distributed. In order to do this, let us first list all the types of data from our dataset and select only the numerical ones.

list(set(dataset_df.dtypes.tolist()))

df_num = dataset_df.select_dtypes(include = ['float64', 'int64'])
df_num.head()


# Now let us plot the distribution for all the numerical features.
df_num.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)