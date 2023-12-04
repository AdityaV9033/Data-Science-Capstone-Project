import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('Processed_Car_dataset.csv')
df.head()
col_drop=['name','selling_price']
X = df.drop(col_drop,axis=1)
y = df['selling_price']
print(X.shape)
print(y.shape)
print(type(X))
print(type(y))
!pip install -U scikit-learn
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
results = {}
for reg_name, reg in regressors.items():
    m=reg.fit(X_train, y_train)
    print("Training :",reg_name)
    ypred = reg.predict(X_test)
    print("Predicting using :",reg_name)
    reg_eval_metrics(y_test,ypred) 
    train_test_scr(m)
    results[reg_name] = m.score(X_test, y_test)
    print("")
res = pd.DataFrame(results,index=['R2_Score'])
res.T 
best_model = max(results, key=results.get)
print(f"The best model is: {best_model} with R2 score: {results[best_model]}")
