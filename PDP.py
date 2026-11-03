import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

# define font settings
font_en = FontProperties(family="Times New Roman", weight="bold")

def set_plot_font():
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False

def plot_and_save_pdp(model, X_train, save_root=r""):
    set_plot_font()
    rcParams.update({
        'axes.labelweight': 'bold',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'font.size': 12,
    })

    explainer = shap.Explainer(model)
    sample = X_train.sample(n=min(3000, len(X_train)), random_state=30)
    shap_values = explainer(sample)
    shap_importance = pd.Series(abs(shap_values.values).mean(axis=0), index=sample.columns).sort_values(ascending=False)
    top_features = shap_importance.head(15).index.tolist()

    pdp_results = []

    for feat in top_features:
        x_min = X_train[feat].min()
        x_max = X_train[feat].max()
        x_grid = np.linspace(x_min, x_max, 50)

        X_copy = sample.copy()
        avg_preds = []
        for val in x_grid:
            X_copy[feat] = val
            preds = model.predict(X_copy)
            avg_preds.append(np.mean(preds))

        plt.figure(figsize=(8, 5))
        plt.plot(x_grid, avg_preds, linewidth=3, color='#2c3e50')
        plt.grid(alpha=0.3, linestyle='--')

        plt.xlabel(feat, fontproperties=font_en, fontsize=12)
        plt.ylabel('Partial Dependence ', fontproperties=font_en, fontsize=12)
        plt.title(f'PDP - {feat}', fontproperties=font_en, fontsize=14)
        plt.tight_layout()

        img_path = f"{save_root}/TN_PDP_{feat}.png"
        plt.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close()

        temp = pd.DataFrame({
            'Feature': feat,
            'Feature_Value': x_grid,
            'Partial_Dependence': avg_preds
        })
        pdp_results.append(temp)

    pdp_df = pd.concat(pdp_results, ignore_index=True)
    csv_path = f"{save_root}/TEC_PDP_analysis_results_XGB.csv"
    pdp_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print("PDP analysis finished! Figures and csv file have been saved.")
