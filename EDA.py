import pandas as pd
import numpy as np
import seaborn as sns
from collections import Counter

# Load Data
import os
os.getcwd()
os.chdir('./Titanic - ML from Disaster/data')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_ID = test['PassengerId']


# Data Summary
train.columns.values
test.columns.values

train.head()
test.head()

train.info()
test.info()

train.describe()
train.describe(include=['O']) # capital o, not 0


# Outlier detection
def outlier_finder(df, n, features):

    outlier_idx = []

    # iterate over features(columns)
    for col in features:
        Q1 = np.nanpercentile(df[col], 25)      # 1st quartile (25%)
        Q3 = np.nanpercentile(df[col], 75)      # 3rd quartile (75%)
        IQR = Q3 - Q1                           # Inter-quartile range (IQR)
        outlier_step = 1.5 * IQR                # outlier step

        # Find index for outliers for each cols, and append it to the list
        outlier_list = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_idx.extend(outlier_list)

        # select obs with more than n outliers
        outlier_indices = Counter(outlier_idx)
        multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers


# detect outliers from Age, SibSp , Parch and Fare
outliers = outlier_finder(train, 2, ["Age", "SibSp", "Parch", "Fare"])
print(train[["Age", "SibSp", "Parch", "Fare"]].loc[outliers])
train = train.drop(outliers, axis = 0).reset_index(drop=True)


# Join train and test data
train_len = len(train)
df = pd.concat(objs=[train, test], axis=0).reset_index(drop=True)  # 0/’index’, 1/’columns


# Missing values
df = df.fillna(np.nan)
df.isnull().sum()
# Age and Cabin have 256 and 1007 missing value respectively, and Survived had 418, corresponding the test.

# Drop Cabin variable
df.drop(labels=['PassengerId', 'Ticket', 'Cabin'], axis=1, inplace=True)


df['Embarked'].value_counts()
df['Embarked'] = df['Embarked'].fillna('S')         # Fill with the most frequent value
df['Fare'] = df['Fare'].fillna(df['Fare'].median()) # Fill with median

# convert Sex into categorical value 0 for male and 1 for female
df["Sex"] = df["Sex"].map({"male": 0, "female":1})
g = sns.heatmap(df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch']].corr(),cmap="BrBG",annot=True)

# Impute missing 'Age' with the median age of similar rows according to 'Pclass', 'SibSp' and 'Parch'
missing_age_idx = list(df["Age"][df["Age"].isnull()].index)

for i in missing_age_idx:
    med = df["Age"].median()
    pred = df["Age"][((df['SibSp'] == df.iloc[i]["SibSp"]) & (df['Parch'] == df.iloc[i]["Parch"]) & (df['Pclass'] == df.iloc[i]["Pclass"]))].median()
    if not np.isnan(pred) :
        df['Age'].iloc[i] = pred
    else:
        df['Age'].iloc[i] = med

df.isnull().sum()


# Analyze by Pivoting Features
def surv_rate(var1, var2='Survived'):
    a = train[[var1, var2]].groupby([var1], as_index=False).mean().sort_values(by=var2, ascending=False)
    print(a)

vars = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
for i in vars:
    print(surv_rate(i))
    print('-'*20)


# Visualizing

# Age
g = sns.kdeplot(train['Age'][(train["Survived"] == 0) & (train["Age"].notnull())], color="Red", shade = True)
g = sns.kdeplot(train['Age'][(train["Survived"] == 1) & (train["Age"].notnull())], ax=g, color="Blue", shade= True)
g.set_xlabel('Age')
g.set_ylabel('Frequency')
g = g.legend(['Not Survived', 'Survived'])
# Approximately Gaussian dist, little tailed, child and young tend to have slightly high chance of survival

# Fare
g = sns.distplot(df["Fare"], color="m", label="Skewness : %.2f"%(df["Fare"].skew()))
g = g.legend(loc="best")
# Distribution is very skewed, use log to reduce skew problem

df["Fare"] = df["Fare"].map(lambda i: np.log(i) if i > 0 else 0)

g = sns.distplot(df["Fare"], color="b", label="Skewness : %.2f"%(df["Fare"].skew()))
g = g.legend(loc="best")


# Get Title from Name
title = [i.split(",")[1].split(".")[0].strip() for i in df["Name"]]
df["Title"] = pd.Series(title)
df["Title"].head()

df["Title"] = df["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df["Title"] = df["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
df["Title"] = df["Title"].astype(int)

g = sns.countplot(df["Title"])
g = g.set_xticklabels(["Master", "Miss/Ms/Mme/Mlle/Mrs", "Mr", "Rare"])

g = sns.factorplot(x="Title", y="Survived", data=df, kind="bar")
g = g.set_xticklabels(["Master","Miss-Mrs","Mr","Rare"])
g = g.set_ylabels("survival probability")

# Drop Name variable
df.drop(labels=["Name"], axis=1, inplace=True)


from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
one_hot = encoder.fit_transform(df['Embarked'])
encoder.classes_
df2 = pd.DataFrame(one_hot)
df2.columns = ['Embarked_C', 'Embarked_Q', 'Embarked_S']
df = pd.concat(objs=[df, df2], axis=1)
df.drop(labels=['Embarked'], axis=1, inplace=True)
df.head()



# Separate train and test
train = df[:train_len]
test = df[train_len:]
test.drop(labels=['Survived'], axis=1, inplace=True)

train.head()
test.head()


pd.DataFrame(train).to_csv('clean_train.csv', index=False)
pd.DataFrame(test).to_csv('clean_test.csv', index=False)