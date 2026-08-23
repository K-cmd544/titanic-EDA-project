#Exploratory Data Analysis EBA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url)

#inspect
print(df.info())
print(df.describe())

#handle missing value
df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df=df.drop_duplicates()

#Filter:passenger in first class
first_class=df[df["Pclass"]==1]
print("First Class passengers: \n", first_class.head())

#Bar chart:survival rate by class

survival_by_class=df.groupby("Pclass")["Survived"].mean()
survival_by_class.plot(kind="bar",color="skyblue")
plt.title("Survival Rate by class")
plt.ylabel("Survival Rate")
plt.show()

#Histogram: Age distribution
sns.histplot(df["Age"], kde=True, bins=20, color="purple")
plt.title("Age distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

#Scatter plot: age vs fare
plt.scatter(df["Age"],df["Fare"], alpha=0.5, color="green")
plt.title("Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()