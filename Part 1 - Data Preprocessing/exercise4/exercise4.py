"""
1: Import necessary Python libraries: pandas, train_test_split from sklearn.model_selection, and StandardScaler
from sklearn.preprocessing.

2: Load the Iris dataset using Pandas read.csv. Dataset name is iris.csv.

3: Use train_test_split to split the dataset into an 80-20 training-test set.

4: Apply random_state with 42 value in train_test_split function for reproducible results.

5: Print X_train, X_test, Y_train, and Y_test to understand the dataset split.

6: Use StandardScaler to apply feature scaling on the training and test sets.

7: Print scaled training and test sets to verify feature scaling.
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
df = pd.read_csv("iris.csv")

# Separate features and target
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=42)
# Apply feature scaling on the training and test sets
# the training data is scaled so that each feature has a mean==0 and a std_dev == 1
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Print the scaled training and test sets
print(X_train)
print(X_test)
print(y_train)
print(y_test)