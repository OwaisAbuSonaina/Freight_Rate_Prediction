# generate_predictions.py
import pandas as pd
import numpy as np
import joblib

def haversine(lat1, lon1, lat2, lon2):
    R = 3959.0 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1)/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def preprocess_for_inference(df, train_ref):
    df_clean = df.copy()
    
    # 1. Rescue missing features for the December dataset
    if 'pickup_lat' not in df_clean.columns:
        locs = train_ref[['pickup', 'pickup_lat', 'pickup_lon']].drop_duplicates('pickup')
        dests = train_ref[['delivery', 'delivery_lat', 'delivery_lon']].drop_duplicates('delivery')
        df_clean = df_clean.merge(locs, on='pickup', how='left').merge(dests, on='delivery', how='left')
        
    if 'market_index' not in df_clean.columns:
        df_clean['market_index'] = 1.0  # Assume neutral market
        df_clean['quote_signal'] = train_ref['quote_signal'].mean() # Use global average quote

    # 2. Apply the exact engineering from main.py
    df_clean['haversine_dist'] = haversine(df_clean['pickup_lat'], df_clean['pickup_lon'], 
                                           df_clean['delivery_lat'], df_clean['delivery_lon'])
    df_clean['expected_base_rate'] = df_clean['distance'] * df_clean['quote_signal']
    df_clean['market_pressure'] = df_clean['market_index'] * df_clean['quote_signal']
    
    # Temporal Feature Engineering
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    df_clean['month'] = df_clean['date'].dt.month
    df_clean['day'] = df_clean['date'].dt.day
    df_clean['dayofweek'] = df_clean['date'].dt.dayofweek
    df_clean['dayofyear'] = df_clean['date'].dt.dayofyear

    cat_cols = ['pickup', 'delivery', 'equipment']
    df_clean[cat_cols] = df_clean[cat_cols].astype('category')
    
    # 3. Drop columns not used in the model
    drop_cols = ['load_id', 'date', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 'predicted_rate']
    return df_clean.drop(columns=[col for col in drop_cols if col in df_clean.columns])

if __name__ == "__main__":
    # Load model and training data (used as a reference book for missing coordinates)
    model = joblib.load("./models/freight_rate_model_lgb.pkl")
    train_data = pd.read_csv("./dataset/train-test.csv")

    # Predict Validation Set
    val_df = pd.read_csv("./dataset/validation.csv")
    X_val = preprocess_for_inference(val_df, train_data)
    
    # Inverse the log transform (expm1) matching main.py
    val_preds = np.expm1(model.predict(X_val)) 
    
    val_template = pd.read_csv("./dataset/validation-predictions-template.csv")
    val_template['predicted_rate'] = val_preds
    val_template.to_csv("validation_predictions.csv", index=False)
    print("Saved validation_predictions.csv")

    # Predict December Set
    dec_df = pd.read_csv("./dataset/december-chart-inputs.csv")
    X_dec = preprocess_for_inference(dec_df, train_data)
    
    dec_df['predicted_rate'] = np.expm1(model.predict(X_dec))
    dec_df.to_csv("./dataset/december-chart-inputs.csv", index=False)
    print("Updated december-chart-inputs.csv")