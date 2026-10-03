# main.py

###################################
# IMPORTS
###################################
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import lightgbm as lgb
import time
import joblib

###################################
# LOAD DATA & FEATURE ENGINEERING
###################################
data = pd.read_csv("./dataset/train-test.csv").drop(columns=['load_id'])
print("\nData Loaded.\n\n")

# Cull the extreme 1% outliers 
threshold = data['posted_rate'].quantile(0.99)
data = data[data['posted_rate'] <= threshold].copy()

# Add engineered math features
def haversine(lat1, lon1, lat2, lon2):
    R = 3959.0 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

data['haversine_dist'] = haversine(data['pickup_lat'], data['pickup_lon'], 
                                   data['delivery_lat'], data['delivery_lon'])

data['expected_base_rate'] = data['distance'] * data['quote_signal']

data['market_pressure'] = data['market_index'] * data['quote_signal']

# Temporal Feature Engineering
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['dayofweek'] = data['date'].dt.dayofweek
data['dayofyear'] = data['date'].dt.dayofyear

# Set native category types (LightGBM can deal with categorical features)
categorical_cols = ['pickup', 'delivery', 'equipment']
data[categorical_cols] = data[categorical_cols].astype('category')

# Drop unused columns
data = data.drop(columns=['date', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon'])

###################################
# SPLIT DATA
###################################
X = data.drop(columns=['posted_rate'])
y = data['posted_rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data Ready for Training.\n\n")

###################################
# TRAIN & EVALUATE
###################################
y_train_log = np.log1p(y_train)

model = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=500, num_leaves=31, random_state=42, n_jobs=-1)

start = time.time()
model.fit(X_train, y_train_log)
end = time.time()
print("Model Finished Training.\n\n")

preds = np.expm1(model.predict(X_test))

print(
    f"""
############################
EVALUATION
############################
\n
MAE: {mean_absolute_error(y_test, preds):.2f}
\n
RMSE: {root_mean_squared_error(y_test, preds):.2f}
\n
R2: {r2_score(y_test, preds):.2f}    
\n
Time: {(end-start):.2f}
\n\n
"""
)

###################################
# SAVE THE MODEL
###################################
joblib.dump(model, "./models/freight_rate_model_lgb.pkl")
print("Model saved to models/freight_rate_model_lgb.pkl")