import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression

# Try to import CLR (Requirement: pip install scikit-bio)
try:
    from skbio.stats.composition import clr
except ImportError:
    print("\n[ERROR] The 'scikit-bio' library is not installed.")
    print("Please run: pip install scikit-bio")
    exit()

# 1. File Input
path = input("Enter the file path (.csv or .txt): ").strip().replace("'", "").replace('"', "")

try:
    # Try reading with tab separator first, then fallback to comma
    df = pd.read_csv(path, sep='\t', decimal=',')
    if df.shape[1] < 2:
        df = pd.read_csv(path, sep=',', decimal='.')
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# 2. Data Preparation
# The LAST column is the dependent variable (Y), previous ones are anatomical features (X)
X_raw = df.iloc[:, :-1].copy()
Y_raw = df.iloc[:, -1:].copy()
Y_name = Y_raw.columns[0]

print(f"\n[INFO] Analyzing anatomical influence on: {Y_name}")

# --- AUTOMATIC PROPORTION DETECTION (Suffix _%) ---
# Identifies columns ending in _% to apply CLR transformation
cols_prop = [c for c in X_raw.columns if str(c).endswith('_%')]

if cols_prop:
    print(f"[INFO] Applying CLR to {len(cols_prop)} proportion variables.")
    # Replace zeros with a tiny value to avoid logarithm errors
    matriz_prop = X_raw[cols_prop].replace(0, 1e-6)
    X_raw[cols_prop] = clr(matriz_prop)
else:
    print("[WARNING] No columns with '_%' found for CLR transformation.")

# --- STANDARDIZATION (StandardScaler) ---
# Essential for comparing microns, counts, and CLR values on the same scale
scaler = StandardScaler()
X_sc = scaler.fit_transform(X_raw)
Y_sc = scaler.fit_transform(Y_raw)

# 3. Statistical Calculation
reg = LinearRegression()
reg.fit(X_sc, Y_sc)

r2_real = reg.score(X_sc, Y_sc)
f_stat, p_values = f_regression(X_sc, Y_sc.flatten())

# 4. Generate Results Report
results_file_name = f"results_{Y_name}.txt"
with open(results_file_name, "w") as f:
    f.write(f"=== ANATOMICAL INFLUENCE ANALYSIS: {Y_name} ===\n\n")
    f.write(f"Dependent Variable (Response): {Y_name}\n")
    f.write(f"R² (Model Explanatory Power): {r2_real:.4f}\n\n")
    f.write(f"{'Feature':<20} {'p-value':<10} {'Significant?':<15}\n")
    for i, col in enumerate(X_raw.columns):
        sig = "Yes (*)" if p_values[i] < 0.05 else "No"
        f.write(f"{col:<20} {p_values[i]:<10.4f} {sig:<15}\n")

# 5. Graphics (Biplot)
importances = reg.coef_[0]
# Pearson Correlation for the horizontal axis (direct relationship strength)
corrs = np.array([np.corrcoef(X_sc[:, i], Y_sc.flatten())[0, 1] for i in range(X_sc.shape[1])])

plt.figure(figsize=(14, 8))
sns.set_theme(style="white")
# Varied color palette to distinguish variables
colors = sns.color_palette("husl", len(X_raw.columns))
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '8', 'X']

# Central axes
plt.axhline(0, color='gray', lw=1, ls='--', alpha=0.3)
plt.axvline(0, color='gray', lw=1, ls='--', alpha=0.3)

for i, col in enumerate(X_raw.columns):
    xi, yi = corrs[i], importances[i]
    # Light gray arrow for explanatory variables
    plt.arrow(0, 0, xi*0.9, yi*0.9, color='gray', alpha=0.2, head_width=0.02)
    # Point with specific marker and color
    plt.scatter(xi, yi, color=colors[i], marker=markers[i % len(markers)], 
                s=250, edgecolor='black', label=f"{col} (p={p_values[i]:.3f})", zorder=5)

# Red Highlight Arrow for Dependent Variable (Y)
plt.arrow(0, 0, 1.1, 0, color='red', width=0.005, head_width=0.04, zorder=6)
plt.text(1.15, 0, Y_name, color='red', fontweight='bold', fontsize=14, va='center')

plt.title(f"RDA: Anatomical Influence on {Y_name}\n[R² = {r2_real:.2f}]", fontsize=15)
plt.xlabel(f"Correlation with {Y_name}", fontsize=12)
plt.ylabel("Model Weight (Relative Importance)", fontsize=12)

# Place legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Anatomical Variables")
plt.tight_layout()

graph_name = f"rda_{Y_name}.png"
plt.savefig(graph_name, dpi=300)
plt.show()

print(f"\nCompleted for {Y_name}!")
print(f"- Report generated: {results_file_name}")
print(f"- Graph saved: {graph_name}")