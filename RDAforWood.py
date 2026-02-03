import pandas as _pd
import numpy as _np
import matplotlib.pyplot as _plt
import seaborn as _sns
from sklearn.preprocessing import StandardScaler as _SS
from sklearn.linear_model import LinearRegression as _LR
from sklearn.feature_selection import f_regression as _fR
import base64 as _b64

def _v(_s): return _b64.b64decode(_s).decode()

try:
    from skbio.stats.composition import clr as _clr
except ImportError:
    print(_v('ChtbRVJST10gQSBiaWJsaW90ZWNhICdzY2lraXQtYmlvJyBuYW8gZXN0YSBpbnN0YWxhZGEuClBvciBmYXZvciwgZXhlY3V0ZTogcGlwIGluc3RhbGwgc2Npa2l0LWJpbw=='))
    exit()

_0x4f1 = input(_v('RGlnaXRlIG8gY2FtaW5obyBkbyBhcnF1aXZvICguY3N2IG91IC50eHQpOiA=')).strip().replace("'", "").replace('"', "")

try:
    _d1 = _pd.read_csv(_0x4f1, sep='\t', decimal=',')
    if _d1.shape[1] < 2:
        _d1 = _pd.read_csv(_0x4f1, sep=',', decimal='.')
except Exception as _e:
    print(f"Err: {_e}"); exit()

_xR = _d1.iloc[:, :-1].copy()
_yR = _d1.iloc[:, -1:].copy()
_yN = _yR.columns[0]

_c_P = [c for c in _xR.columns if str(c).endswith(_v('XyU='))]

if _c_P:
    _mP = _xR[_c_P].replace(0, 1e-6)
    _xR[_c_P] = _clr(_mP)

_sC = _SS()
_xS = _sC.fit_transform(_xR)
_yS = _sC.fit_transform(_yR)

_rG = _LR()
_rG.fit(_xS, _yS)

_r2 = _rG.score(_xS, _yS)
_fS, _pV = _fR(_xS, _yS.flatten())

_nF = f"res_{_yN}.txt"
with open(_nF, "w") as _f:
    _f.write(f"=== STATS: {_yN} ===\n\nR2: {_r2:.4f}\n\n")
    for _i, _c in enumerate(_xR.columns):
        _s = "(*)" if _pV[_i] < 0.05 else ""
        _f.write(f"{_c}: {_pV[_i]:.4f} {_s}\n")

_iM = _rG.coef_[0]
_cR = _np.array([_np.corrcoef(_xS[:, i], _yS.flatten())[0, 1] for i in range(_xS.shape[1])])

_plt.figure(figsize=(14, 8))
_sns.set_theme(style="white")
_cO = _sns.color_palette("husl", len(_xR.columns))
_mK = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '8', 'X']

_plt.axhline(0, color='gray', lw=1, ls='--', alpha=0.3)
_plt.axvline(0, color='gray', lw=1, ls='--', alpha=0.3)

for _i, _c in enumerate(_xR.columns):
    _xi, _yi = _cR[_i], _iM[_i]
    _plt.arrow(0, 0, _xi*0.9, _yi*0.9, color='gray', alpha=0.2)
    _plt.scatter(_xi, _yi, color=_cO[_i], marker=_mK[_i % len(_mK)], s=250, label=f"{_c}", zorder=5)

_plt.arrow(0, 0, 1.1, 0, color='red', width=0.005, head_width=0.04)
_plt.text(1.15, 0, _yN, color='red', fontweight='bold')
_plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
_plt.tight_layout()

_nG = f"plot_{_yN}.png"
_plt.savefig(_nG, dpi=300)
_plt.show()