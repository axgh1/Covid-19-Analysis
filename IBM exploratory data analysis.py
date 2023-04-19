import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DA0321EN-SkillsNetwork/LargeData/m2_survey_data.csv")


# Plot the distribution curve
sns.histplot(data=df, x="ConvertedComp", kde=True)

# Plot the histogram
plt.hist(df["ConvertedComp"], bins=50)
plt.xlabel("Salary (USD)")
plt.ylabel("Count")
plt.show()

# Calculate the median of the ConvertedComp column
median = df['ConvertedComp'].median()

print("The median of the ConvertedComp column is:", median)

# Count the number of responders who identified themselves only as a Man
count = df[df['Gender'] == 'Man']['Gender'].count()

print("The number of responders who identified themselves only as a Man is:", count)

# Calculate the median ConvertedComp of responders identified themselves only as a Woman
median = df[df['Gender'] == 'Woman']['ConvertedComp'].median()

print("The median ConvertedComp of responders identified themselves only as a Woman is:", median)

# Calculate the five number summary for the Age column
summary = df['Age'].describe(percentiles=[0.25, 0.5, 0.75, 0.95])

print("The five number summary for the Age column is:")
print(summary)

# Plot a histogram of the Age column
plt.hist(df['Age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Plot a box plot of the ConvertedComp column
sns.boxplot(x=df['ConvertedComp'])
plt.show()

# Calculate the interquartile range of the ConvertedComp column
Q1 = df['ConvertedComp'].quantile(0.25)
Q3 = df['ConvertedComp'].quantile(0.75)
IQR = Q3 - Q1

print("The interquartile range for the ConvertedComp column is:", IQR)

upper_bound = Q3 + 1.5*IQR
lower_bound = Q1 - 1.5*IQR
num_outliers = len(df[(df['ConvertedComp'] < lower_bound) | (df['ConvertedComp'] > upper_bound)])
num_outliers

df_new = df[(df['ConvertedComp'] >= lower_bound) & (df['ConvertedComp'] <= upper_bound)]
df_new

corr_matrix = df.corr()
corr_age = corr_matrix['Age']
corr_age

