import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


## Load the dataset
train_file_path = "train.csv"
dataset_df = pd.read_csv(train_file_path)
print("Full train dataset shape is {}".format(dataset_df.shape))

model = LinearRegression()

# Prepare the dataset
# To Implement a linear regression model to predict the prices of houses based on their square footage and the number of bedrooms and bathrooms.
Data = dataset_df[["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr", "SalePrice"]]
Data.head(3)

X_train, X_test, y_train, y_test = train_test_split(Data.drop("SalePrice", axis=1), Data.SalePrice, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

valid_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, valid_pred)
print("Mean-Squared-Error : ", mse)

# Prediction on Testing Data
test_ds = pd.read_csv("test.csv")

predictions = model.predict(test_ds[["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr"]].dropna())

print("Model Prediction on the Testing dataset :->", predictions)
