import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("titanic2.csv")

# Style
sns.set_style("whitegrid")

# 1. Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df["Age"].dropna(), bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# 2. Fare Distribution
plt.figure(figsize=(6,4))
sns.histplot(df["Fare"], bins=20, color="green")
plt.title("Fare Distribution")
plt.show()

# 3. Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

# 4. Passenger Class vs Fare
plt.figure(figsize=(6,4))
sns.boxplot(x="Pclass", y="Fare", data=df)
plt.title("Passenger Class vs Fare")
plt.show()