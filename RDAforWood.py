import sys
import traceback

try:
    import subprocess

    # --- AUTOMATIC PACKAGE INSTALLER ---
    def check_and_install(package, import_name=None):
        import_name = import_name or package
        try:
            __import__(import_name)
        except ImportError:
            print(f"\n[INFO] The '{package}' library is missing. Installing automatically via pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    print("[INFO] Checking system dependencies...")
    check_and_install('pandas')
    check_and_install('numpy')
    check_and_install('matplotlib', 'matplotlib')
    check_and_install('seaborn')
    check_and_install('scikit-learn', 'sklearn')
    check_and_install('statsmodels')
    check_and_install('scikit-bio', 'skbio')

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.feature_selection import f_regression
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from skbio.stats.composition import clr

    # 1. File Input
    print("\nHINT: Ensure you type the file name WITH the extension (e.g., my_data.csv or my_data.txt)\n")
    path = input("Enter the file name or path: ").strip().replace("'", "").replace('"', "")

    # --- UNIVERSAL FILE READER ---
    print("\n[INFO] Reading file...")
    try:
        df = pd.read_csv(path, sep=';', decimal=',')
        if df.shape[1] < 2:
            df = pd.read_csv(path, sep='\t', decimal=',')
        if df.shape[1] < 2:
            df = pd.read_csv(path, sep=',', decimal='.')
        if df.shape[1] < 2:
            raise ValueError("Data separator not recognized. Ensure it is a valid CSV or TXT.")
    except Exception as e:
        raise ValueError(f"Could not read the file. Check if the name is correct. Detail: {e}")

    # 2. Data Preparation
    df = df.dropna(axis=1, how='all')
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype == object:
             df[col] = df[col].astype(str).str.replace(',', '.')

    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='any')

    print(f"[INFO] Valid samples (rows) found for analysis: {len(df)}")

    X_raw = df.iloc[:, :-1].copy()
    Y_raw = df.iloc[:, -1:].copy()
    Y_name = str(Y_raw.columns[0]).strip()

    if Y_name.startswith('Unnamed') or Y_name == '':
        Y_name = "Target_Variable"
        Y_raw.columns = [Y_name]

    print(f"\n[INFO] Target variable detected: '{Y_name}'")

    # --- SUB-COMPOSITIONAL CLR TRANSFORMATION ---
    vessel_cols = [c for c in X_raw.columns if c in ['SVP_%', 'MVP_%']]
    tissue_cols = [c for c in X_raw.columns if c in ['VLuP_%', 'VWP_%', 'FLuP_%', 'FWP_%', 'RP_%', 'APP_%']]

    if vessel_cols:
        print(f"[INFO] Applying CLR on Vessel sub-composition.")
        matriz_vessels = X_raw[vessel_cols].replace(0, 1e-6)
        X_raw[vessel_cols] = clr(matriz_vessels)

    if tissue_cols:
        print(f"[INFO] Applying CLR on Tissue sub-composition.")
        matriz_tissues = X_raw[tissue_cols].replace(0, 1e-6)
        X_raw[tissue_cols] = clr(matriz_tissues)

    # --- CONTROLLED BASELINE REMOVAL ---
    cols_to_drop = [c for c in ['SVP_%', 'VWP_%'] if c in X_raw.columns]
    if cols_to_drop:
        X_raw = X_raw.drop(columns=cols_to_drop)
        print(f"[INFO] Intentionally removed {cols_to_drop} AFTER CLR to act as statistical baselines.")

    # --- STANDARDIZATION ---
    scaler = StandardScaler()
    X_sc_df = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)
    Y_sc = scaler.fit_transform(Y_raw)

    # --- 3. VIF (SIMULAÇÃO E ESCOLHA DO USUÁRIO) ---
    print("\n" + "="*50)
    print(" --- FILTRO DE MULTICOLINEARIDADE (VIF) ---")
    print("="*50)

    vif_input = input("\nDigite o limite do VIF (ex: 5 para rigoroso ou 10 para padrão): ").strip()
    try:
        vif_threshold = float(vif_input)
    except ValueError:
        print("[AVISO] Entrada inválida. Usando limite padrão: 10.0")
        vif_threshold = 10.0

    print(f"\n[INFO] Simulando cortes do VIF...")
    
    X_sim = X_sc_df.copy()
    sim_removed = []
    while True:
        vifs = []
        for i in range(X_sim.shape[1]):
            try:
                vif_val = variance_inflation_factor(X_sim.values, i)
            except:
                vif_val = np.inf
            vifs.append((X_sim.columns[i], vif_val))
            
        vifs.sort(key=lambda x: x[1], reverse=True)
        if vifs and (vifs[0][1] > vif_threshold or np.isinf(vifs[0][1])):
            sim_removed.append(vifs[0])
            X_sim = X_sim.drop(columns=[vifs[0][0]])
        else:
            break

    protected_vars = []
    
    if not sim_removed:
        print("\n[RESULTADO] Ótima notícia: Nenhuma variável ultrapassou o limite do VIF.")
    else:
        print("\n" + "-"*50)
        print(f" ATENÇÃO: O VIF automático EXCLUIRIA as seguintes {len(sim_removed)} variáveis:")
        for var, val in sim_removed:
            print(f"  - {var} (VIF: {val:.2f})")
        print("-" * 50)
        
        print("\nVocê deseja SALVAR alguma destas variáveis do corte automático?")
        protect_input = input("Digite o nome exato delas separadas por vírgula (ex: APP_%, FWP_%) ou pressione ENTER para aceitar a exclusão: ").strip()
        
        if protect_input:
            protected_vars = [v.strip() for v in protect_input.split(',')]
            print(f"[INFO] Variáveis salvas por decisão da especialista: {protected_vars}")

    # Aplicação do VIF definitivo
    removed_vars = []
    while True:
        vifs = []
        for i in range(X_sc_df.shape[1]):
            try:
                vif_val = variance_inflation_factor(X_sc_df.values, i)
            except:
                vif_val = np.inf
            vifs.append((X_sc_df.columns[i], vif_val))
            
        vifs.sort(key=lambda x: x[1], reverse=True)
        removed_in_this_step = False
        
        for feature, vif_val in vifs:
            if vif_val > vif_threshold or np.isinf(vif_val):
                if feature not in protected_vars:
                    removed_vars.append((feature, vif_val))
                    X_sc_df = X_sc_df.drop(columns=[feature])
                    removed_in_this_step = True
                    break 
                else:
                    continue 
            else:
                break 
                
        if not removed_in_this_step:
            break 

    print(f"\n[INFO] Filtro concluído. {len(removed_vars)} variáveis descartadas definitivamente.")
    
    X_sc_final = X_sc_df.values
    final_cols = X_sc_df.columns

    # --- 4. STATISTICAL CALCULATION & PARTIÇÃO DA VARIÂNCIA (EIGENVALUES) ---
    def calc_pseudo_F(X, Y):
        reg = LinearRegression().fit(X, Y)
        Y_pred = reg.predict(X)
        tot_var = np.sum(np.var(Y, axis=0))
        const_var = np.sum(np.var(Y_pred, axis=0))
        unconst_var = tot_var - const_var
        n, k = X.shape
        df_model = k
        df_res = n - k - 1
        if df_res <= 0 or unconst_var == 0:
            return np.inf, const_var / tot_var
        f_stat = (const_var / df_model) / (unconst_var / df_res)
        return f_stat, const_var / tot_var, const_var, unconst_var, tot_var

    f_obs, r2_real, const_var, unconst_var, tot_var = calc_pseudo_F(X_sc_final, Y_sc)
    
    n_perms = 999
    count_greater = 0
    np.random.seed(42)
    for _ in range(n_perms):
        Y_perm = np.random.permutation(Y_sc)
        f_perm, _, _, _, _ = calc_pseudo_F(X_sc_final, Y_perm)
        if f_perm >= f_obs:
            count_greater += 1
    p_val_model = (count_greater + 1) / (n_perms + 1)

    reg_final = LinearRegression()
    reg_final.fit(X_sc_final, Y_sc)
    f_stat_ind, p_values_ind = f_regression(X_sc_final, Y_sc.flatten())

    # --- 5. GENERATE RESULTS REPORT ---
    results_file_name = f"results_RDA_{Y_name}.txt"
    with open(results_file_name, "w", encoding="utf-8") as f:
        f.write(f"=== ANATOMICAL INFLUENCE ANALYSIS (RDA): {Y_name} ===\n\n")
        f.write(f"Dependent Variable (Response): {Y_name}\n")
        f.write(f"VIF Threshold Applied: {vif_threshold}\n")
        if protected_vars:
            f.write(f"Protected Features (exempt from VIF filtering): {', '.join(protected_vars)}\n")
        f.write(f"Variables Excluded due to Multicollinearity: {len(removed_vars)}\n")
        for var, v in removed_vars:
            f.write(f"  - {var} (VIF at removal: {v:.2f})\n")
        
        f.write(f"\n--- RDA GLOBAL SIGNIFICANCE (ANOVA) ---\n")
        f.write(f"Number of Permutations: {n_perms}\n")
        f.write(f"F-statistic: {f_obs:.4f}\n")
        
        p_val_model_str = f"{p_val_model:.2e}" if p_val_model < 0.001 else f"{p_val_model:.3f}"
        f.write(f"Model p-value: {p_val_model_str}\n")
        f.write(f"R² (Model Explanatory Power): {r2_real:.4f}\n\n")
        
        # NOVA SEÇÃO PARA MATERIAL SUPLEMENTAR
        f.write(f"--- SUPPLEMENTARY MATERIAL DATA (VARIANCE PARTITIONING) ---\n")
        f.write(f"Total Variance: {tot_var:.4f}\n")
        f.write(f"Constrained Variance (Eigenvalue Axis 1): {const_var:.4f}\n")
        f.write(f"Unconstrained Variance (Residual): {unconst_var:.4f}\n")
        f.write(f"Proportion Explained (R²): {r2_real:.4f}\n\n")

        f.write(f"--- INDIVIDUAL FEATURE SIGNIFICANCE ---\n")
        f.write(f"{'Feature':<20} {'p-value':<10} {'Significant?':<15}\n")
        for i, col in enumerate(final_cols):
            sig = "Yes (*)" if p_values_ind[i] < 0.05 else "No"
            p_val_str = f"{p_values_ind[i]:.2e}" if p_values_ind[i] < 0.001 else f"{p_values_ind[i]:.4f}"
            f.write(f"{col:<20} {p_val_str:<10} {sig:<15}\n")

    # --- 6. GRAPHICS (RDA BIPLOT) ---
    importances = reg_final.coef_[0]
    corrs = np.array([np.corrcoef(X_sc_final[:, i], Y_sc.flatten())[0, 1] for i in range(X_sc_final.shape[1])])

    plt.figure(figsize=(14, 8))
    sns.set_theme(style="white")
    colors = sns.color_palette("husl", len(final_cols))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '8', 'X']

    plt.axhline(0, color='gray', lw=1, ls='--', alpha=0.3)
    plt.axvline(0, color='gray', lw=1, ls='--', alpha=0.3)

    for i, col in enumerate(final_cols):
        xi, yi = corrs[i], importances[i]
        plt.arrow(0, 0, xi*0.9, yi*0.9, color='gray', alpha=0.2, head_width=0.02)
        
        p_val = p_values_ind[i]
        p_str_legend = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.3f}"
        
        plt.scatter(xi, yi, color=colors[i], marker=markers[i % len(markers)], 
                    s=250, edgecolor='black', label=f"{col} (p={p_str_legend})", zorder=5)

    plt.arrow(0, 0, 1.1, 0, color='red', width=0.005, head_width=0.04, zorder=6)
    plt.text(1.15, 0, Y_name, color='red', fontweight='bold', fontsize=14, va='center')

    plt.title(f"RDA: Anatomical Influence on {Y_name}\n[R² = {r2_real:.2f} | F = {f_obs:.2f} | p = {p_val_model_str}]", fontsize=15)
    plt.xlabel(f"Correlation with {Y_name}", fontsize=12)
    plt.ylabel("Model Weight (Relative Importance)", fontsize=12)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Anatomical Variables\n(Post-VIF)",
               markerscale=0.5, labelspacing=1.0, handletextpad=0.5, borderpad=1.0)

    plt.tight_layout()
    graph_name = f"rda_biplot_{Y_name}.png"
    plt.savefig(graph_name, dpi=300)
    plt.close()

    print(f"\n[SUCCESS] Finalizado para {Y_name}!")
    print(f"  -> RDA Biplot salvo: {graph_name}")
    print(f"  -> Relatório texto salvo: {results_file_name}")
    input("\nPressione ENTER para fechar o programa...")

except Exception as e:
    print("\n" + "="*50)
    print(" 🚨 CRITICAL ERROR ENCOUNTERED 🚨 ")
    print("="*50)
    traceback.print_exc()
    print("="*50)
    input("\nThe program paused. Take a photo or copy the text above, then press ENTER to exit...")