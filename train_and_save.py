
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

# Load cleaned data
df = pd.read_csv('dataset_clean.csv')
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Drop columns (same as Phase 3)
drop_cols = [
    'destination_interest', 'qualified_flag', 'form_submissions',
    'number_of_pageviews', 'interaction_depth', 'deal_progress',
    'nurture_score', 'readiness_score',
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Encode categoricals and save encoders
cat_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Prepare features and targets
X = df.drop(columns=['record_id', 'target_conv_prob', 'target_profit'])
y_conv = df['target_conv_prob']
y_profit = df['target_profit']

# Train on ALL data for deployment
print("Training XGBoost (conversion)...")
xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
xgb.fit(X, y_conv)

print("Training Random Forest (profit)...")
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X, y_profit)

# Save
joblib.dump(xgb, 'model_conv.pkl')
joblib.dump(rf, 'model_profit.pkl')
joblib.dump(le_dict, 'encoders.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')
