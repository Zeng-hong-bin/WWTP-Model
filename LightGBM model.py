import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import shap
import os
import chardet
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from scipy.stats import zscore

# ---------------------- 1. Load data with automatic encoding detection ----------------------
def load_data(filepath):
    with open(filepath, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']
    print(f"Detected file encoding: {encoding}")
    return pd.read_csv(filepath, encoding=encoding)

# ---------------------- 2. Custom Rolling Origin CV, compatible with GridSearchCV ----------------------
class RollingOriginCV:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            val_start = train_end
            val_end = train_end + fold_size
            if val_end > n:
                val_end = n
            train_idx = np.arange(0, train_end)
            val_idx = np.arange(val_start, val_end)
            yield train_idx, val_idx

# ---------------------- 3. LightGBM optimization (GridSearch with Rolling-origin CV) ----------------------
def train_lgbm_with_optimization(X_train, y_train):
    param_grid = {
        'num_leaves': [30, 50, 70, 100],
        'max_depth': [3, 10, 20, 30],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500]
    }

    model = LGBMRegressor(random_state=30, n_jobs=-1)
    rolling_cv = RollingOriginCV(n_splits=5)
    grid = GridSearchCV(model, param_grid, cv=rolling_cv, n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print("Best hyperparameters:", grid.best_params_)
    return grid.best_estimator_

# ---------------------- 4. Font settings ----------------------
def set_plot_font():
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 5. Prediction scatter plot ----------------------
def plot_fit(y_true, y_pred, set_name):
    set_plot_font()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    font = FontProperties(family='Times New Roman', size=18)

    rcParams.update({'axes.labelweight': 'bold', 'axes.titlesize': 20, 'axes.labelsize': 20,
                     'font.size': 14, 'font.family': 'serif', 'font.serif': ['Times New Roman']})

    plt.figure(figsize=(10, 6.5))
    plt.scatter(y_true, y_pred, s=150, alpha=0.5, edgecolors='w')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)

    plt.xlabel('True value (mg/L)')
    plt.ylabel('Predicted value (mg/L)')

    plt.text(0.75, 0.15,
             f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR2: {r2:.4f}',
             transform=plt.gca().transAxes, fontsize=18,
             fontproperties=font,
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    plt.tight_layout()
    path = f"TN_LGBM_prediction_{set_name}.png"
    plt.savefig(path)
    plt.close()
    print(f"{set_name} prediction plot saved: {path}")

# ---------------------- 6. Save predictions to CSV ----------------------
def save_predictions(y_true, y_pred, X_data, full_data, set_name):
    save_path = f"TN_LGBM_predictions_{set_name}.csv"

    if 'DateTime' in full_data.columns:
        time_col = full_data.loc[y_true.index, 'DateTime']
    else:
        time_col = y_true.index

    df = pd.DataFrame({
        'DateTime': time_col,
        'True': y_true,
        'Predicted': y_pred
    })

    for col in X_data.columns:
        df[col] = X_data[col].values

    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"{set_name} predictions saved: {save_path}")

# ---------------------- 7. SHAP analysis ----------------------
def plot_shap(model, X_train):
    set_plot_font()

    explainer = shap.Explainer(model)
    sample = X_train.sample(n=min(5000, len(X_train)), random_state=30)
    shap_values = explainer(sample)

    rcParams.update({
        'axes.labelweight': 'bold',
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'font.size': 14,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold',
        'mathtext.default': 'regular'
    })

    plt.figure(figsize=(10, 6.5))
    shap.summary_plot(shap_values.values, sample, plot_type="dot", show=False)

    ax = plt.gca()
    new_labels = []
    for label in ax.get_yticklabels():
        text = label.get_text()
        text = text.replace("NH3-N", r"NH$_3$-N")
        new_labels.append(text)

    ax.set_yticklabels(
        new_labels,
        fontsize=14,
        fontweight='bold',
        fontfamily='serif',
        fontname='Times New Roman'
    )
    # y-axis tick labels
    for label in ax.get_yticklabels():
        label.set_fontsize(14)
        label.set_fontweight('bold')
        label.set_fontname('Times New Roman')

    # x-axis tick labels
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
        label.set_fontweight('bold')
        label.set_fontname('Times New Roman')

    # axis titles
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.xaxis.label.set_fontsize(14)
    ax.yaxis.label.set_fontsize(14)
    ax.xaxis.label.set_fontname('Times New Roman')
    ax.yaxis.label.set_fontname('Times New Roman')

    # color bar text (High/Low)
    if len(plt.gcf().axes) > 1:
        cbar_ax = plt.gcf().axes[-1]
        for lab in cbar_ax.get_xticklabels() + cbar_ax.get_yticklabels():
            lab.set_fontweight('bold')
            lab.set_fontname('Times New Roman')
            lab.set_fontsize(14)
        cbar_ax.yaxis.label.set_fontweight('bold')
        cbar_ax.yaxis.label.set_fontname('Times New Roman')
        cbar_ax.yaxis.label.set_fontsize(14)
        for text in cbar_ax.texts:
            text.set_fontweight('bold')
            text.set_fontname('Times New Roman')
            text.set_fontsize(14)

    plt.tight_layout()
    save_path = "TN_SHAP_LGBM.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"SHAP plot saved: {save_path}")

def save_shap_importance(model, X_train):
    explainer = shap.Explainer(model)
    sample = X_train.sample(n=min(8000, len(X_train)), random_state=30)
    shap_values = explainer(sample)

    importance = np.abs(shap_values.values).mean(axis=0)
    df = pd.DataFrame({"Feature": X_train.columns, "SHAP_Importance": importance})
    df = df.sort_values("SHAP_Importance", ascending=False)

    save_path = "TN_SHAP_feature_importance_LGBM.csv"
    df.to_csv(save_path, index=False, encoding='utf-8-sig')

    print(f"SHAP feature importance saved: {save_path}")

# ---------------------- 8. Main pipeline ----------------------
def process_site_data(filepath):
    data = load_data(filepath)

    features = ['Influent flowrate', 'pHinf', 'CODinf', 'BOD5inf', 'SSinf', 'NH3-Ninf', 'TNinf', 'TPinf',
                'MLSS-I', 'MLSS-II', 'SV30-I', 'SV30-II', 'Aeration volume']
    target = 'TNeff'

    # 7:1:2 random split train:val:test, shuffle=True, random_state=30
    train_val, test_data = train_test_split(data, test_size=0.2, random_state=30, shuffle=True)
    train_data, val_data = train_test_split(train_val, test_size=0.125, random_state=30, shuffle=True)

    X_train, y_train = train_data[features], train_data[target]
    X_val, y_val = val_data[features], val_data[target]
    X_test, y_test = test_data[features], test_data[target]

    # train model (Rolling-origin CV inside GridSearch)
    model = train_lgbm_with_optimization(X_train, y_train)

    # train set: plot + save
    train_pred = model.predict(X_train)
    plot_fit(y_train, train_pred, "Train")
    save_predictions(y_train, train_pred, X_train, train_data, "Train")

    # validation set: plot + save
    val_pred = model.predict(X_val)
    plot_fit(y_val, val_pred, "Val")
    save_predictions(y_val, val_pred, X_val, val_data, "Val")

    # test set: plot + save
    test_pred = model.predict(X_test)
    plot_fit(y_test, test_pred, "Test")
    save_predictions(y_test, test_pred, X_test, test_data, "Test")

    # SHAP analysis
    plot_shap(model, X_train)
    save_shap_importance(model, X_train)

def main():
    file_path = ''
    process_site_data(file_path)

if __name__ == "__main__":
    main()