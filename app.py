import streamlit as st
import pandas as pd
import numpy as np
st.title('ML Model for Selling Price of Cars')
df = pd.read_csv('Processed_Car_dataset.csv')
df.head()
col_drop=['name','selling_price']
X = df.drop(col_drop,axis=1)
y = df['selling_price']
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
def reg_eval_metrics(y_train, ypred): 
    mae = mean_absolute_error(y_train, ypred)
    mse = mean_squared_error(y_train, ypred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, ypred)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)
def train_test_scr(model):
    print('Training Score',model.score(X_train,y_train))  
    print('Testing Score',model.score(X_test,y_test))  
regressors = {
    'Linear Regression': LinearRegression(),
    'Lasso Regression': Lasso(),
    'Ridge Regression': Ridge(),
    'KNN Regression': KNeighborsRegressor(n_neighbors=11),
    'Decision Tree Regression': DecisionTreeRegressor(max_depth=5),
    'RandomForest Regression': RandomForestRegressor(n_estimators=100, random_state=42),
    'Bagging_DT Regression': BaggingRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=20, random_state=42),
    'Bagging_LR Regression': BaggingRegressor(LinearRegression(), n_estimators=20, random_state=42),
    'AdaBoost_DT Regression': AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=50, random_state=42)
}
model=[]
results = {}
for reg_name, reg in regressors.items():
    m=reg.fit(X_train, y_train)
    print("Training :",reg_name)
    ypred = reg.predict(X_test)
    print("Predicting using :",reg_name)
    reg_eval_metrics(y_test,ypred) 
    train_test_scr(m)
    results[reg_name] = m.score(X_test, y_test)
    model.append(m)

st.sidebar.header("Select the ML model you want to use")
Drop_options = ["Random Forest Regression", "Bagging with Decision Tree Regressor", "KNN Regression", "Bagging with Linear Regressor"]
param2 = st.sidebar.selectbox("Drop_options", options=Drop_options)

Year = st.slider("Model Purchase Year", min_value=1992, max_value=2020, step=1)
st.text(f"Year value: {Year}")
Km_Driven = st.slider("km driven", min_value=1, max_value=806599, step=1)
st.text(f"Km Driven value: {Km_Driven}")
Fuel = st.slider("Fuel type", min_value=0, max_value=5,step=1)
st.text(f"Fuel type value: {Fuel}")
Seller_Type = st.slider("Seller Type", min_value=0, max_value=2,step=1)
st.text(f"Seller_Type value: {Seller_Type}")
Transmission = st.slider("Transmission", min_value=0, max_value=1,step=1)
st.text(f"Transmission value: {Transmission}")
Owner = st.slider("Owner", min_value=0, max_value=4,step=1)
st.text(f"Owner value: {Owner}")
Model_Name = st.slider("Model_Name", min_value=0, max_value=28,step=1)
st.text(f"Model_Name value: {Model_Name}")
input={'year':Year,
       'km_driven':Km_Driven,
       'fuel':Fuel,
       'seller_type':Seller_Type,
       'transmission':Transmission,
       'owner':Owner,
       'Model_Name':Model_Name}
input_X=pd.DataFrame(input, index=['value'])
st.text("Input value of features:")
input_X.T
scaled_data={}
for i_name, i in input.items():
    mean_value = X[i_name].mean()
    std_value = X[i_name].std()
    S_data = (i - mean_value) / std_value
    scaled_data[i_name]=S_data

X1_scaled=pd.DataFrame(scaled_data, index=['value'])
if param2=="Random Forest Regression":
    ypred=model[5].predict(X1_scaled)
    st.text(f"Selling price of Car is: {ypred}")
elif param2=="Bagging with Decision Tree Regressor":  
    ypred=model[6].predict(X1_scaled)
    st.text(f"Selling price of Car is: {ypred}")
elif param2=="Bagging with Linear Regressor":
    ypred=model[7].predict(X1_scaled)
    st.text(f"Selling price of Car is: {ypred}")
else :
    ypred=model[3].predict(X1_scaled)
    st.text(f"Selling price of Car is: {ypred}")



