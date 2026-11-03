import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

file_path = r"input.csv"
save_dir  = r"output_split"

target_cols  = ['CODinf', 'BOD5inf', 'SSinf', 'NH3-Ninf', 'TNinf', 'TPinf','MLSS-Ⅰ',
                'MLSS-Ⅱ', 'SV30-Ⅰ', 'SV30-Ⅱ']   # fill in as needed
feature_cols = ['Influent flowrate', 'pHinf', 'CODinf', 'BOD5inf', 'SSinf', 'NH3-Ninf', 'TNinf', 'TPinf',
               'MLSS-Ⅰ', 'MLSS-Ⅱ', 'SV30-Ⅰ', 'SV30-Ⅱ', 'Aeration volume']

random_state = 42
max_iter = 10
tol = 0.01
# ===================================================
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(file_path, encoding="utf-8-sig")
model_cols = feature_cols + target_cols

rng = np.random.RandomState(random_state)

row_has_missing = df[model_cols].isnull().any(axis=1)
complete_idx = df.index[~row_has_missing].tolist()
incomplete_idx = df.index[row_has_missing].tolist()

n = len(df)
n_test = int(round(n * 0.2))

test_idx = complete_idx.copy()
if len(test_idx) < n_test:
    need = n_test - len(test_idx)
    extra = rng.choice(incomplete_idx, size=need, replace=False).tolist()
    test_idx += extra
    incomplete_idx = [i for i in incomplete_idx if i not in set(extra)]

train_idx, val_idx = train_test_split(
    incomplete_idx, test_size=0.125, random_state=random_state
)

train_df = df.loc[train_idx].copy()
val_df   = df.loc[val_idx].copy()
test_df  = df.loc[test_idx].copy()

print(f"Total samples: {n}")
print(f"Complete rows: {len(complete_idx)} (all assigned to test set)")
print(f"Train: {len(train_df)} ({len(train_df)/n:.1%})")
print(f"Val:   {len(val_df)} ({len(val_df)/n:.1%})")
print(f"Test:  {len(test_df)} ({len(test_df)/n:.1%})")


# MissForest (fitted on the training set only)
def missforest_fit(train_df, max_iter=10, tol=0.01, random_state=42):
    """Iterative imputation on the training set.
    Returns: imputed train, final RF model per column, column means."""
    imputed = train_df.copy()
    cols_with_missing = [c for c in imputed.columns if imputed[c].isnull().any()]

    # initial fill: column means
    col_means = {}
    for col in cols_with_missing:
        col_means[col] = imputed[col].mean()
        imputed[col] = imputed[col].fillna(col_means[col])

    rf_models = {}
    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}")
        max_diff = 0.0

        for col in cols_with_missing:
            missing_mask = train_df[col].isnull()
            if missing_mask.sum() == 0:
                continue

            old_vals = imputed.loc[missing_mask, col].copy()

            feat_cols = [c for c in imputed.columns if c != col]
            X_train = imputed.loc[~missing_mask, feat_cols]
            y_train = imputed.loc[~missing_mask, col]
            X_pred  = imputed.loc[missing_mask, feat_cols]

            rf = RandomForestRegressor(
                n_estimators=100, max_depth=10,
                random_state=random_state, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            imputed.loc[missing_mask, col] = rf.predict(X_pred)
            rf_models[col] = rf  # keep final model for val/test prediction

            max_diff = max(max_diff, np.abs(old_vals - imputed.loc[missing_mask, col]).mean())

        if max_diff < tol:
            print(f"Converged at iteration {iteration + 1}, stopping early")
            break

    # fallback means: columns not missing in train but possibly missing elsewhere
    for col in imputed.columns:
        if col not in col_means:
            col_means[col] = imputed[col].mean()

    return imputed, rf_models, col_means


def missforest_transform(df_part, imputed_train, rf_models, col_means):
    """Fill missing values in val/test using models fitted on train."""
    out = df_part.copy()
    train_means = imputed_train.mean()  # column means of imputed train (fallback)

    for col in out.columns:
        mask = out[col].isnull()
        if mask.sum() == 0:
            continue
        if col in rf_models:
            feat_cols = [c for c in out.columns if c != col]
            # features may also be missing: temp-fill with train means, then predict
            X_pred = out.loc[mask, feat_cols].fillna(train_means[feat_cols])
            out.loc[mask, col] = rf_models[col].predict(X_pred)
        else:
            out[col] = out[col].fillna(train_means[col])
    return out


print("\nStarting MissForest imputation (training set only)")
imputed_train, rf_models, col_means = missforest_fit(
    train_df, max_iter=max_iter, tol=tol, random_state=random_state
)

print("Imputing val / test with models fitted on train")
imputed_val  = missforest_transform(val_df,  imputed_train, rf_models, col_means)
imputed_test = missforest_transform(test_df, imputed_train, rf_models, col_means)


train_out = imputed_train.copy(); train_out.insert(0, 'split', 'train')
val_out   = imputed_val.copy();   val_out.insert(0, 'split', 'val')
test_out  = imputed_test.copy();  test_out.insert(0, 'split', 'test')

train_out.to_csv(os.path.join(save_dir, "train.csv"), index=False, encoding="utf-8-sig")
val_out.to_csv(os.path.join(save_dir, "val.csv"),     index=False, encoding="utf-8-sig")
test_out.to_csv(os.path.join(save_dir, "test.csv"),   index=False, encoding="utf-8-sig")
pd.concat([train_out, val_out, test_out]).to_csv(
    os.path.join(save_dir, "all_with_split.csv"), index=False, encoding="utf-8-sig"
)

print("\nDone")
print(f"Saved to: {save_dir}")
print("  - train.csv / val.csv / test.csv (imputed, with split column)")
print("  - all_with_split.csv (combined file)")