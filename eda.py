import pandas as pd

# Dataset load karo
df = pd.read_csv("titanic2.csv")

# Pehli 5 rows dekho
print(df.head())

# Dataset ki information
print(df.info())

# Missing values
print(df.isnull().sum())

# Statistical summary
print(df.describe())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Correlation
print("\nCorrelation:")
print(df.corr(numeric_only=True))
# Summary Statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))
# Unique values in each column
print("\nUnique Values:")
print(df.nunique())

# Null value percentage
print("\nMissing Value Percentage:")
print((df.isnull().sum()/len(df))*100)
print("\nSurvival Count:")
print(df["Survived"].value_counts())

print("\nPassenger Class Count:")
print(df["Pclass"].value_counts())

print("\nGender Count:")
print(df["Sex"].value_counts())


import matplotlib.pyplot as plt
import seaborn as sns

# Survival Count Plot
plt.figure(figsize=(5,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Passenger Class Count Plot
plt.figure(figsize=(5,4))
sns.countplot(x="Pclass", data=df)
plt.title("Passenger Class Count")
plt.show()

# Gender Count Plot
plt.figure(figsize=(5,4))
sns.countplot(x="Sex", data=df)
plt.title("Gender Count")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()