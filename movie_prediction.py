# importing libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

# loading dataset
df=pd.read("IMDb India Movies.csv")

# selecting columns
df = df[['Genre', 'Director', 'Actor1', 'Actor2', 'Actor3','Rating' ]]

# Dropping missing values
df = df.dropna()
X = df[['Genre', 'Director', 'Actor1', 'Actor2', 'Actor3' ]]
Y = df['Rating']

X_train, X_test, Y_train, Y_test = tain_test_split(X, Y, test_size=0.2, random_state=42)
preprocessor = ColumnTransformer('Cat', OneHotEncoder(handle_unknown='ignore'),['Genre', 'Director', 'Actor1', 'Actor2', 'Actor3'
])

linreg_model = Pipeline([('preprocessor', preprocess), ('regressor', LinearRegression())])

linreg_model.fit(X_train,Y_train)   
Y_pred_lr = linreg_model.predict(X_test)

print ("==LINEAR REGRESSION ==")
print ("MSE:",mean_squared_erorr(Y_test, Y_pred_lr))
print("R2 score:", r2_score(Y_test, Y_pred_lr))

rf_model = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=300, random_state=42))])
rf_model.fit(X_train, Y_train)
Y_pred_rf = rf_model.predict(X_test)

print ("\n===Random forest regression")
print ("MSE:", mean_squared_error(Y_test, Y_pred_rf))
print ("R2 score:", r2_score(Y_test, Y_pred_rf))



