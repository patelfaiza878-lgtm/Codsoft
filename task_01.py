# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# loading Dataset
df = pd.read_csv("train.csv")
print("DATASET loaded successfully!")
print(df.head())

# checking missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# filling the missing values
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)

df.drop(column=['Cabin'],inplace=True)
# Data visualization
plt.figure(figsize=(12,6))

# overall survival count
plt.subplot(2,3,2)
sns.countplot(x="sex",hue="survived",data=df,palette="set1")
plt.title("survived by Gender")

# converting categorical text into numbers
df['sex']=df['sex'].map({'male':0,'female':1})
df[Embarked].map({'s':0,'c':1,'q':2})

# selecting features and target
X =df[['Pclass','Sex','Age','Sibsp','Parch','Fare','Embarked']]
y = df['Survived']

# train_test split
X_train,X_test,y_tain,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# prediction and accuracy 
y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"\nModel accuracy:{accuracy*100:.2f}")

# predict for sample passenger 
import numpy as np 

test_passenger=np.array([[3,0,22,2,0,7.25,0]])
prediction = model.predict(test_passenger)
print("\nprediction for test passenger(0= Not survived, 1= survived):",prediction[0])

