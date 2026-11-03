import numpy as np
import pandas as pd

def mad_outlier_replacement(df: pd.DataFrame, feature_list: list, threshold: float = 3.0) -> tuple[pd.DataFrame, dict]:

    df_clean = df.copy()
    outlier_summary = {}

    for col in feature_list:
        x = df_clean[col].values
        med = np.median(x)
        mad = np.median(np.abs(x - med))

        if mad == 0:
            robust_z = np.zeros_like(x)
        else:
            robust_z = 0.6745 * (x - med) / mad

        outlier_mask = np.abs(robust_z) > threshold
        n_out = int(np.sum(outlier_mask))
        outlier_summary[col] = n_out

        df_clean.loc[outlier_mask, col] = med

    return df_clean, outlier_summary


if __name__ == "__main__":

    target_features = ["CODinf"]
    df_result, outlier_info = mad_outlier_replacement(target_features, threshold=3.0)

    print("===== Outlier Count Summary (MAD method) =====")
    for ft, cnt in outlier_info.items():
        print(f"{ft}: {cnt} outliers replaced")

    print("\nOriginal data first 10 rows:")
    print("\nData after MAD outlier replacement first 10 rows:")
    print(df_result.head())
