"""
1: Import required libraries - Pandas, Numpy, and required classes for this task - ColumnTransformer, OneHotEncoder, LabelEncoder.

2: Start by loading the Titanic dataset into a pandas data frame. This can be done using the pd.read_csv function. The dataset's name is 'titanic.csv'.

3: Identify the categorical features in your dataset that need to be encoded. You can store these feature names in a list for easy access later.

4: To apply OneHotEncoding to these categorical features, create an instance of the ColumnTransformer class. Make sure to pass the OneHotEncoder() as an argument along with the list of categorical features.

5: Use the fit_transform method on the instance of ColumnTransformer to apply the OneHotEncoding.

6: The output of the fit_transform method should be converted into a NumPy array for further use.

7: The 'Survived' column in your dataset is the dependent variable. This is a binary categorical variable that should be encoded using LabelEncoder.

8.  Print the updated matrix of features and the dependent variable vector
"""


# Importing the necessary libraries

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the dataset
df = pd.read_csv("titanic.csv")


# Identify the categorical data
categorical_features = ["Sex", "Embarked", "Pclass"]

# Implement an instance of the ColumnTransformer class
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)],
                       remainder="passthrough")

# Apply the fit_transform method on the instance of ColumnTransformer
X = ct.fit_transform(df)

# Convert the output into a NumPy array
X = np.array(X)

# Use LabelEncoder to encode binary categorical data

y = LabelEncoder().fit_transform(df["Survived"])

# Print the updated matrix of features and the dependent variable vector
print("Updated matrix of features: \n", X)
print("Updated dependent variable vector: \n", y)
