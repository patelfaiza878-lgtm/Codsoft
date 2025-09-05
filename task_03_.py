import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# loading dataset
data pd.read_csv("IRIS.csv")
print ("sample Data:\n", data.head())

X = data.drop(columns=["Id", "Species"])
Y = data["Species"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size = 0.2, random_state=1, stratify = Y
)

# training decision tree
model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, Y_train)

# prediction
Y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:" accuracy_score(Y_test, Y_pred))
print("\nClassification Report:\n", classification_report(Y_test, Y_pred))