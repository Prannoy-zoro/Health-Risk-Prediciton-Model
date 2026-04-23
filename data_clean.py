import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
df.plot(x='Age',y='Height',style='o')
plt.show()

print(df.info())

print(df.describe())

le_dict ={}
for col in df.select_dtypes(include='object'):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
df = df.dropna() # Drop rows with missing values
print(df.head())

df["BMI"] = df["Weight"] / (df["Height"] ** 2)
print(df.head())


df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
print("Initial shape:", df.shape)

# Handle missing values
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicates
df = df.drop_duplicates()
print("After duplicates:", df.shape)

# Remove outliers (Age, Height, Weight)
outlier_cols = ["Age", "Height", "Weight"]

Q1 = df[outlier_cols].quantile(0.25)
Q3 = df[outlier_cols].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[outlier_cols] < (Q1 - 1.5 * IQR)) |
          (df[outlier_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

print("After outliers:", df.shape)

# Create BMI
df["BMI"] = df["Weight"] / (df["Height"] ** 2)

# Encode categorical columns
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Final check
print(df.info())
print(df.head())

# Save cleaned data
df.to_csv("cleaned_obesity_data.csv", index=False)

print("✅ Cleaned data saved")