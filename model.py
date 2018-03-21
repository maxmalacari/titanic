import pandas as pd
import numpy as np

# Load the training and test data
train_data = pd.read_csv("~/.kaggle/competitions/titanic/train.csv")
test_data = pd.read_csv("~/.kaggle/competitions/titanic/test.csv")

# Connect the data together, we'll split it later
train_data = train_data.append(test_data)

# Let's look at the data
print(train_data.shape,"\n","-"*40)
print(train_data.columns,"\n","-"*40)
print(train_data.info(),"\n","-"*40)
print(train_data.head(),"\n","-"*40)
print(train_data.describe(),"\n","-"*40)

# Drop the name, ticket, embarked, and passenger id columns
train_data_cleaned = train_data.drop(["PassengerId","Name","Ticket","Embarked"], axis=1)
print(train_data_cleaned.columns,"\n","-"*40)

# Set the training target
#y = train_data_cleaned.Survived
#train_data_cleaned = train_data_cleaned.drop("Survived", axis=1)
#print(train_data_cleaned.columns,"\n","-"*40)

# One-hot encode the sex column
# columns wants a list, drop_first will set male = 0, female = 1, as opposed to having two distinct columns
train_data_cleaned = pd.get_dummies(train_data_cleaned, columns=['Sex'], drop_first=True)

# Clean up columns with NaN data
print("Missing data:\n", train_data_cleaned.isnull().sum())

# Cabin has missing values - let's try training with this column dropped first
train_data_cleaned_noCabin = train_data_cleaned.drop("Cabin", axis=1)

# There are a lot of NaN entries in the survived column, remove these rows
#for row in y_train:
#    print(row)
X = train_data_cleaned_noCabin
X = X.dropna(axis=0, subset=["Survived"])
y = X.Survived
X = X.drop("Survived", axis=1)

# First try an imputer on the missing age values (fill NaN with average age)
# we have to split the data into training and test samples first, otherwise the imputer will use the average
# of the combined dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)

from sklearn.preprocessing import Imputer
my_imputer = Imputer()
X_train = my_imputer.fit_transform(X_train)
X_test = my_imputer.fit_transform(X_test)

print("Final number of training data points:", y_train.size)
print("Are all entries finite?", np.all(np.isfinite(X_train)))
print("Are any entries NaN?", np.any(np.isnan(X_train)))

print("Final number of testing data points:", y_test.size)
print("Are all entries finite?", np.all(np.isfinite(X_test)))
print("Are any entries NaN?", np.any(np.isnan(X_test)))

# Now, let's create a model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)

#from xgboost import XGBRegressor
#model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
#model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)])

# Test the model
from sklearn.metrics import mean_absolute_error
prediction = model.predict(X_test)
#print(prediction,"\n","-"*40)
#print(y_test.values) # y_test.values converts
print(mean_absolute_error(y_test.values, prediction))