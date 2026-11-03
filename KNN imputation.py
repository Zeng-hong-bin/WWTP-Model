import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
file_path = r".csv"
save_path = r".csv"
df = pd.read_csv(file_path, encoding="utf-8-sig")
target_cols = ['CODinf', 'BOD5inf', 'SSinf', 'NH3-Ninf', 'TNinf', 'TPinf','MLSS-Ⅰ', 'MLSS-Ⅱ', 'SV30-Ⅰ', 'SV30-Ⅱ' ] # fill in as needed
knn_data = df[target_cols].copy()
scaler = StandardScaler()
knn_scaled = pd.DataFrame(
    scaler.fit_transform(knn_data),
    columns=knn_data.columns
)
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
knn_imputed_scaled = knn_imputer.fit_transform(knn_scaled)

knn_imputed = pd.DataFrame(
    scaler.inverse_transform(knn_imputed_scaled),
    columns=knn_data.columns
)

for col in target_cols:
    df[col] = knn_imputed[col]

df.to_csv(save_path, index=False, encoding="utf-8-sig")

print("✅KNN Done")
print(f"✅：{save_path}")