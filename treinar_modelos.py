"""
treinar_modelos.py
-----------------
Treina Random Forest e Gradient Boosting por segmento.
Dataset: FIIs de tijolo · jan/2019–dez/2024
Segmentos: Logistico, Shoppings, Escritorios, Lajes Corporativas, Hibrido

Uso:
    python treinar_modelos.py --arquivo dataset_fiis_2019_2024_brapi_v2.xlsx
"""

import argparse, json, warnings, joblib, requests
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ── Constantes ────────────────────────────────────────────────────────────────
COL_SIGLA    = "Sigla"
COL_DATA     = "Data"
COL_DY       = "Dividendos_Yield"
COL_PVP      = "P_VP"
COL_SELIC    = "SELIC"
COL_SEGMENTO = "Segmento"
COL_TIPO     = "Tipo_do_Fundo"

SAIDA_DIR = Path("modelo")

# Apenas fundos de Tijolo
TIPO_TREINO = {"Tijolo"}

# Segmentos válidos para treinamento
SEGMENTOS_VALIDOS = {
    "Logistico", "Shoppings", "Escritorios",
    "Lajes Corporativas", "Hibrido"
}

# Segmentos excluídos e motivos
SEGS_EXCLUIR = {
    "Titulos e Val. Mob.": "DY depende do spread privado dos CRIs — variável não pública",
    "FOF":                 "DY depende de outros FIIs e decisões do gestor",
    "Hospital":            "apenas 3 fundos — amostra insuficiente",
    "Varejo":              "apenas 3 fundos — amostra insuficiente",
    "Outros":              "grupo heterogêneo sem critério de homogeneidade",
}

# Período pandemia
PANDEMIA_INI = "2020-03"
PANDEMIA_FIM = "2020-12"

# Algoritmos a comparar
ALGORITMOS = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=5, min_samples_leaf=3,
        max_features=0.7, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        subsample=0.8, min_samples_leaf=3, random_state=42
    ),
}


def carregar_dados(caminho: str) -> pd.DataFrame:
    p = Path(caminho)
    df = pd.read_excel(p) if p.suffix in (".xlsx", ".xls") else pd.read_csv(p)
    df[COL_DATA] = pd.to_datetime(df[COL_DATA], errors="coerce")
    df = df.sort_values([COL_SIGLA, COL_DATA]).reset_index(drop=True)
    print(f"[✓] {len(df)} linhas · {df[COL_SIGLA].nunique()} fundos · "
          f"{df[COL_DATA].min().strftime('%Y-%m')} → {df[COL_DATA].max().strftime('%Y-%m')}")
    return df


def buscar_selic() -> dict:
    print("[→] Buscando SELIC do BCB SGS 4390...", end=" ")
    selic = {}
    try:
        r = requests.get(
            "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4390/dados",
            params={"formato": "json", "dataInicial": "01/01/2019", "dataFinal": "31/12/2024"},
            timeout=20
        )
        if r.status_code == 200:
            for item in r.json():
                mes = pd.to_datetime(item["data"], format="%d/%m/%Y").strftime("%Y-%m")
                selic[mes] = round(float(item["valor"]) / 100, 6)
            print(f"✓ {len(selic)} meses")
        else:
            print(f"⚠ HTTP {r.status_code}")
    except Exception as e:
        print(f"⚠ falhou ({e})")
    return selic


def adicionar_selic(df: pd.DataFrame, selic: dict) -> pd.DataFrame:
    mes_col = pd.to_datetime(df[COL_DATA]).dt.strftime("%Y-%m")
    df[COL_SELIC] = mes_col.map(selic)
    return df


def construir_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DY_lag1"]   = df.groupby(COL_SIGLA)[COL_DY].shift(1)
    df["DY_lag2"]   = df.groupby(COL_SIGLA)[COL_DY].shift(2)
    df["DY_lag3"]   = df.groupby(COL_SIGLA)[COL_DY].shift(3)
    df["DY_target"] = df.groupby(COL_SIGLA)[COL_DY].shift(-1)
    if COL_PVP in df.columns:
        df["PVP_lag1"] = df.groupby(COL_SIGLA)[COL_PVP].shift(1)
    antes = len(df)
    df = df.dropna(subset=["DY_lag1", "DY_lag2", "DY_lag3", "DY_target"]).reset_index(drop=True)
    print(f"  [✓] {len(df)} amostras ({antes - len(df)} removidas por NaN)")
    return df


def construir_pipeline(num_cols, cat_cols, estimador):
    transformers = [("num", SimpleImputer(strategy="median"), num_cols)]
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols
        ))
    return Pipeline([("pre", ColumnTransformer(transformers)), ("modelo", estimador)])


def avaliar(pipe, X_train, y_train, X_test, y_test) -> dict:
    preds = pipe.predict(X_test)
    r2    = r2_score(y_test, preds)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = float(np.sqrt(np.mean((y_test.values - preds) ** 2)))
    mask  = y_test != 0
    mape  = mean_absolute_percentage_error(y_test[mask], preds[mask]) * 100 if mask.sum() > 0 else float("nan")
    # CV temporal
    cv_r2 = None
    if len(X_train) >= 50:
        try:
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            for tr, te in tscv.split(X_train):
                pipe.fit(X_train.iloc[tr], y_train.iloc[tr])
                scores.append(r2_score(y_train.iloc[te], pipe.predict(X_train.iloc[te])))
            cv_r2 = float(np.mean(scores))
            pipe.fit(X_train, y_train)
        except Exception:
            pass
    cv_str = f" | CV R²: {cv_r2:.4f}" if cv_r2 is not None else ""
    print(f"    R²: {r2:.4f} | MAE: {mae:.6f} | RMSE: {rmse:.6f} | MAPE: {mape:.1f}%{cv_str}")
    return {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape, "cv_r2": cv_r2}


def treinar_segmento(seg, df_seg, selic, excluir_pandemia=False):
    sufixo = " (sem pandemia)" if excluir_pandemia else ""
    print(f"\n{'─'*60}")
    print(f"  {seg}{sufixo}")
    print(f"{'─'*60}")

    df_work = df_seg.copy()
    if excluir_pandemia:
        mes = pd.to_datetime(df_work[COL_DATA]).dt.strftime("%Y-%m")
        df_work = df_work[~((mes >= PANDEMIA_INI) & (mes <= PANDEMIA_FIM))]
        print(f"  ⚠ Pandemia excluída ({PANDEMIA_INI}–{PANDEMIA_FIM})")

    if selic:
        df_work = adicionar_selic(df_work, selic)

    df_lags = construir_lags(df_work)
    if len(df_lags) < 20:
        print(f"  Amostras insuficientes — pulando")
        return None

    num_cols = ["DY_lag1", "DY_lag2", "DY_lag3"]
    cat_cols = []

    if "PVP_lag1" in df_lags.columns and df_lags["PVP_lag1"].notna().sum() > len(df_lags) * 0.3:
        num_cols.append("PVP_lag1")
        print(f"  + PVP_lag1 ({df_lags['PVP_lag1'].notna().mean():.0%})")

    if COL_SELIC in df_lags.columns and df_lags[COL_SELIC].notna().mean() > 0.3:
        num_cols.append(COL_SELIC)
        print(f"  + SELIC")

    data_corte  = pd.to_datetime(df_lags[COL_DATA]).quantile(0.8)
    mask_treino = pd.to_datetime(df_lags[COL_DATA]) < data_corte
    X_train = df_lags[num_cols + cat_cols][mask_treino]
    y_train = df_lags["DY_target"][mask_treino]
    X_test  = df_lags[num_cols + cat_cols][~mask_treino]
    y_test  = df_lags["DY_target"][~mask_treino]
    print(f"  Split: treino até {data_corte.strftime('%Y-%m')} ({mask_treino.sum()}) | teste ({(~mask_treino).sum()})")

    resultados = {}
    melhor_r2, melhor_nome, melhor_pipe = -999, None, None

    for nome, estimador in ALGORITMOS.items():
        print(f"  [{nome}]")
        pipe = construir_pipeline(num_cols, cat_cols, estimador)
        pipe.fit(X_train, y_train)
        met = avaliar(pipe, X_train, y_train, X_test, y_test)
        resultados[nome] = {**met, "num_cols": num_cols, "cat_cols": cat_cols}
        if met["r2"] > melhor_r2:
            melhor_r2, melhor_nome, melhor_pipe = met["r2"], nome, pipe

    print(f"  🏆 Melhor: {melhor_nome} (R²={melhor_r2:.4f})")
    return {
        "melhor": melhor_nome,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "n_fundos": int(df_seg[COL_SIGLA].nunique()),
        "excluiu_pandemia": excluir_pandemia,
        "pipes": {nome: construir_pipeline(num_cols, cat_cols, est)
                  for nome, est in ALGORITMOS.items()},
        "metricas": {
            n: {k: round(v, 4) if v is not None and not (isinstance(v, float) and np.isnan(v)) else None
                for k, v in m.items() if k not in ("num_cols","cat_cols")}
            for n, m in resultados.items()
        }
    }


def treinar(caminho_arquivo: str):
    SAIDA_DIR.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  FII Predictor — Treinamento")
    print(f"  Random Forest vs Gradient Boosting | Por segmento")
    print(f"  Apenas Tijolo | jan/2019–dez/2024")
    print(f"{'='*60}\n")

    df = carregar_dados(caminho_arquivo)
    selic = buscar_selic()

    # Filtra apenas Tijolo
    df = df[df[COL_TIPO].isin(TIPO_TREINO)]
    print(f"\n[✓] Apenas fundos de Tijolo: {df[COL_SIGLA].nunique()} fundos")

    # Remove segmentos excluídos
    for seg, motivo in SEGS_EXCLUIR.items():
        df = df[df[COL_SEGMENTO] != seg]
        print(f"[✓] '{seg}' excluído: {motivo}")

    # Filtra apenas segmentos válidos
    df = df[df[COL_SEGMENTO].isin(SEGMENTOS_VALIDOS)]

    # Remove segmentos com < 3 fundos
    contagem = df.groupby(COL_SEGMENTO)[COL_SIGLA].nunique()
    segs_ok  = contagem[contagem >= 3].index
    df = df[df[COL_SEGMENTO].isin(segs_ok)]

    print(f"\n[✓] {df[COL_SIGLA].nunique()} fundos | {df[COL_SEGMENTO].nunique()} segmentos")
    for seg in sorted(segs_ok):
        n = df[df[COL_SEGMENTO]==seg][COL_SIGLA].nunique()
        print(f"    {seg:25s}: {n} fundos")

    segmentos   = sorted(df[COL_SEGMENTO].dropna().unique())
    meta_global = {}
    r2_resumo   = {}

    for seg in segmentos:
        df_seg = df[df[COL_SEGMENTO] == seg].copy()
        slug   = seg.lower().replace(" ","_").replace("/","_")

        # ── Com pandemia ──────────────────────────────────────────────────────
        res = treinar_segmento(seg, df_seg, selic, excluir_pandemia=False)
        if res:
            for nome, pipe in res["pipes"].items():
                pipe.fit(
                    df_seg.pipe(lambda d: adicionar_selic(construir_lags(d), selic) if selic else construir_lags(d))
                    [res["num_cols"] + res["cat_cols"]].dropna(),
                    df_seg.pipe(lambda d: adicionar_selic(construir_lags(d), selic) if selic else construir_lags(d))
                    ["DY_target"].dropna()
                ) if False else None  # pipe já treinado

            # Salva um pkl por algoritmo
            df_lags = construir_lags(adicionar_selic(df_seg.copy(), selic) if selic else df_seg.copy())
            X_all = df_lags[res["num_cols"] + res["cat_cols"]]
            y_all = df_lags["DY_target"]
            for nome, estimador in ALGORITMOS.items():
                p = construir_pipeline(res["num_cols"], res["cat_cols"], estimador)
                p.fit(X_all, y_all)
                slug_mod = nome.lower().replace(" ","_")
                path_pkl = SAIDA_DIR / f"modelo_{slug}_{slug_mod}.pkl"
                joblib.dump(p, path_pkl)
                print(f"  [✓] {path_pkl} ({path_pkl.stat().st_size/1024:.0f} KB)")

            meta_global[seg] = {
                "n_fundos":    res["n_fundos"],
                "num_cols":    res["num_cols"],
                "cat_cols":    res["cat_cols"],
                "melhor":      res["melhor"],
                "metricas":    res["metricas"],
            }
            r2_resumo[seg] = res["metricas"].get(res["melhor"], {}).get("r2", float("nan"))

        # ── Sem pandemia ──────────────────────────────────────────────────────
        res_sp = treinar_segmento(seg, df_seg, selic, excluir_pandemia=True)
        if res_sp:
            df_lags_sp = construir_lags(adicionar_selic(df_seg.copy(), selic) if selic else df_seg.copy())
            mes = pd.to_datetime(df_lags_sp[COL_DATA]).dt.strftime("%Y-%m")
            df_lags_sp = df_lags_sp[~((mes >= PANDEMIA_INI) & (mes <= PANDEMIA_FIM))]
            X_sp = df_lags_sp[res_sp["num_cols"] + res_sp["cat_cols"]]
            y_sp = df_lags_sp["DY_target"]
            for nome, estimador in ALGORITMOS.items():
                p = construir_pipeline(res_sp["num_cols"], res_sp["cat_cols"], estimador)
                p.fit(X_sp, y_sp)
                slug_mod = nome.lower().replace(" ","_")
                path_pkl = SAIDA_DIR / f"modelo_{slug}_{slug_mod}_sem_pandemia.pkl"
                joblib.dump(p, path_pkl)
                print(f"  [✓] {path_pkl}")
            if seg in meta_global:
                meta_global[seg]["sem_pandemia"] = {
                    "melhor":   res_sp["melhor"],
                    "metricas": res_sp["metricas"],
                }

    # ── Fallback geral ────────────────────────────────────────────────────────
    print(f"\n{'─'*60}\n  Modelo geral (fallback)...")
    df_all  = construir_lags(adicionar_selic(df.copy(), selic) if selic else df.copy())
    num_all = ["DY_lag1","DY_lag2","DY_lag3"]
    cat_all = [COL_SEGMENTO]
    if "PVP_lag1" in df_all.columns: num_all.append("PVP_lag1")
    if COL_SELIC  in df_all.columns: num_all.append(COL_SELIC)
    for nome, estimador in ALGORITMOS.items():
        p = construir_pipeline(num_all, cat_all, estimador)
        p.fit(df_all[num_all+cat_all], df_all["DY_target"])
        slug_mod = nome.lower().replace(" ","_")
        joblib.dump(p, SAIDA_DIR / f"fallback_{slug_mod}.pkl")
    print(f"  [✓] fallback RF e GB salvos")

    # ── fundos_recentes.csv ───────────────────────────────────────────────────
    cols_rec = [COL_SIGLA, COL_DY, "DY_lag1","DY_lag2","DY_lag3", COL_SEGMENTO, COL_TIPO]
    if "PVP_lag1" in df_all.columns: cols_rec.insert(3, "PVP_lag1")
    cols_rec = [c for c in cols_rec if c in df_all.columns]
    ultimo = df_all[cols_rec].groupby(COL_SIGLA).last().reset_index()
    ultimo.to_csv(SAIDA_DIR / "fundos_recentes.csv", index=False)
    print(f"  [✓] fundos_recentes.csv ({len(ultimo)} fundos)")

    # ── meta.json ─────────────────────────────────────────────────────────────
    r2_vals = [v for v in r2_resumo.values() if not np.isnan(v)]
    meta = {
        "versao":            "1.0-rf-gb-por-segmento",
        "algoritmos":        ["Random Forest", "Gradient Boosting"],
        "tipo_fundo":        "Tijolo",
        "segmentos":         list(segmentos),
        "periodo_treino":    "jan/2019–dez/2024",
        "r2_medio":          round(float(np.mean(r2_vals)), 4) if r2_vals else None,
        "n_fundos_total":    int(df[COL_SIGLA].nunique()),
        "modelos_por_segmento": meta_global,
        "dataset": Path(caminho_arquivo).name,
    }
    with open(SAIDA_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  [✓] meta.json")

    # ── Resumo separado por algoritmo ────────────────────────────────────────
    for algoritmo in ["Random Forest", "Gradient Boosting"]:
        print(f"\n{'='*72}")
        print(f"  {algoritmo.upper()}")
        print(f"  {'Segmento':25s} {'R²':>10} {'MAE':>12} {'RMSE':>12} {'MAPE':>10}")
        print(f"{'─'*72}")

        r2_vals_alg, mae_vals_alg, rmse_vals_alg = [], [], []

        for seg in segmentos:
            met = meta_global.get(seg, {}).get("metricas", {}).get(algoritmo, {})
            r2   = met.get("r2",   float("nan"))
            mae  = met.get("mae",  float("nan"))
            rmse = met.get("rmse", float("nan"))
            mape = met.get("mape", float("nan"))

            r2_str   = f"{r2:.4f}"   if not np.isnan(r2)   else "—"
            mae_str  = f"{mae:.6f}"  if not np.isnan(mae)  else "—"
            rmse_str = f"{rmse:.6f}" if not np.isnan(rmse) else "—"
            mape_str = f"{mape:.1f}%" if not np.isnan(mape) else "—"

            melhor = meta_global.get(seg, {}).get("melhor", "")
            destaque = " ◄ melhor" if melhor == algoritmo else ""

            print(f"  {seg:25s} {r2_str:>10} {mae_str:>12} {rmse_str:>12} {mape_str:>10}{destaque}")

            if not np.isnan(r2):   r2_vals_alg.append(r2)
            if not np.isnan(mae):  mae_vals_alg.append(mae)
            if not np.isnan(rmse): rmse_vals_alg.append(rmse)

        print(f"{'─'*72}")
        r2_m   = f"{np.mean(r2_vals_alg):.4f}"   if r2_vals_alg   else "—"
        mae_m  = f"{np.mean(mae_vals_alg):.6f}"  if mae_vals_alg  else "—"
        rmse_m = f"{np.mean(rmse_vals_alg):.6f}" if rmse_vals_alg else "—"
        print(f"  {'Média':25s} {r2_m:>10} {mae_m:>12} {rmse_m:>12}")
        print(f"{'='*72}")

    print(f"\n  Pronto! Execute: uvicorn api:app --reload --port 8000\n")


# ── Validação com dados reais de 2025 ────────────────────────────────────────

# Dados reais 2025 — brapi.dev Pro (extraídos abr/2026)
# Usados apenas para validação — NÃO entram no treino
REAL_2025_VALIDACAO = {
    "HGLG11": [0.006746,0.00676,0.006765,0.006761,0.006759,0.006764,0.006759,0.006769,0.006781,0.006769,0.005901,0.006635],
    "XPLG11": [0.007528,0.007653,0.007669,0.00768,0.007682,0.007629,0.007671,0.007676,0.007717,0.007731,0.007738,0.007742],
    "VILG11": [0.005754,0.005852,0.005944,0.006033,0.006131,0.006226,0.006405,0.006403,0.006394,0.006394,0.006348,0.00656],
    "BRCO11": [0.007344,0.007343,0.007346,0.007351,0.007355,0.008882,0.007381,0.007384,0.00739,0.0074,0.007406,0.007412],
    "BTLG11": [0.009587,0.007448,0.008936,0.008298,0.006545,0.012764,0.009648,0.006038,0.004813,0.007493,0.007286,0.006592],
    "XPML11": [0.007582,0.008444,0.006468,0.005394,0.008766,0.005354,0.010807,0.008557,0.008161,0.006645,0.01196,0.01196],
    "VISC11": [0.006414,0.006385,0.006395,0.006418,0.006434,0.006541,0.006552,0.006558,0.006582,0.006592,0.006577,0.006587],
    "HSML11": [0.005951,0.005941,0.005943,0.005958,0.005972,0.005967,0.006292,0.006404,0.006532,0.006728,0.006725,0.00672],
    "KNRI11": [0.006216,0.0062,0.006205,0.006203,0.006184,0.006179,0.006134,0.006134,0.006125,0.006123,0.006122,0.007648],
    "BBPO11": [0.010162,0.010162,0.010138,0.010147,0.010091,0.010103,0.010076,0.010086,0.010095,0.010105,None,None],
    "HGRE11": [0.005504,0.005514,0.005507,0.005512,0.00551,0.016198,0.005461,0.00546,0.005458,0.005455,0.005448,0.009626],
    "CEOC11": [0.006412,0.005989,0.006174,0.006663,0.006322,0.006157,0.006706,0.007251,0.00635,0.006892,0.00676,0.006655],
}

MESES_2025 = ["Jan/25","Fev/25","Mar/25","Abr/25","Mai/25","Jun/25",
               "Jul/25","Ago/25","Set/25","Out/25","Nov/25","Dez/25"]

# Dados reais 2024 para lags iniciais da predição recursiva
REAL_2024_LAGS = {
    "HGLG11": [0.006952,0.006958,0.006971,0.00698,0.007004,0.006952,0.006966,0.006964,0.006966,0.006989,0.007013,0.00704],
    "XPLG11": [None,0.006959,0.006974,0.006987,None,0.006971,0.006916,None,0.00693,0.00659,0.006949,0.006957],
    "VILG11": [0.005259,0.004904,0.00509,0.005261,0.005269,0.005707,0.005713,0.005628,0.00528,0.005372,0.005464,0.005545],
    "BRCO11": [0.007243,None,0.007118,0.00726,0.007268,0.007276,0.007303,0.007309,0.007319,0.007316,0.007321,0.007328],
    "BTLG11": [0.006174,0.004064,0.008758,0.010757,0.007641,0.006327,0.005578,0.008229,0.007379,0.008832,0.01202,0.009335],
    "XPML11": [0.011451,0.008946,0.009123,0.007315,0.005054,0.007329,0.007961,0.007853,0.006372,0.00809,0.00884,0.009535],
    "VISC11": [0.007479,0.00777,0.007813,0.00782,0.007849,0.006682,0.006695,0.006533,0.006313,0.006327,0.006336,0.006342],
    "HSML11": [0.008005,0.008512,0.008287,0.00832,0.008343,0.008351,0.007365,0.007392,0.007389,0.007428,0.007428,0.007347],
    "KNRI11": [0.006296,0.00626,0.006268,0.005858,0.006278,0.006298,0.006185,0.006182,0.006186,0.00618,0.006187,0.006195],
    "BBPO11": [0.009781,0.009661,0.009572,0.009664,0.009667,0.009664,0.009673,None,None,0.009664,None,None],
    "HGRE11": [0.005087,0.00509,0.005095,0.005081,0.005087,0.009795,0.005122,0.005125,0.005119,0.005124,0.005126,0.005094],
    "CEOC11": [0.005939,0.005869,0.005905,0.00568,0.005844,0.005679,0.005978,0.006392,0.00641,0.006427,0.006202,0.006239],
}

# Mapa sigla → segmento para validação
SIGLA_SEG_VALIDACAO = {
    "HGLG11":"Logistico","XPLG11":"Logistico","VILG11":"Logistico",
    "BRCO11":"Logistico","BTLG11":"Logistico",
    "XPML11":"Shoppings","VISC11":"Shoppings","HSML11":"Shoppings",
    "KNRI11":"Hibrido",
    "BBPO11":"Escritorios","HGRE11":"Escritorios","CEOC11":"Escritorios",
}


def prever_recursivo(pipe, num_cols, cat_cols, lags_2024, pvp, selic, n=12):
    """Predição recursiva de N meses usando lags reais de 2024."""
    vals = [v for v in lags_2024 if v is not None and v > 0.0001]
    if len(vals) < 3:
        return None
    lag1 = vals[-1]
    lag2 = vals[-2] if len(vals) >= 2 else lag1
    lag3 = vals[-3] if len(vals) >= 3 else lag2

    serie = []
    for _ in range(n):
        row = {"DY_lag1": lag1, "DY_lag2": lag2, "DY_lag3": lag3}
        if "PVP_lag1" in num_cols:
            row["PVP_lag1"] = pvp if pvp else np.nan
        if "SELIC" in num_cols:
            row["SELIC"] = selic
        X = pd.DataFrame([row])[num_cols + cat_cols]
        try:
            dy = float(pipe.predict(X)[0])
        except Exception:
            dy = lag1
        serie.append(dy)
        lag3, lag2, lag1 = lag2, lag1, dy

    return serie


def imprimir_validacao_2025(meta_global, modelos_dict):
    """
    Imprime tabelas detalhadas de MAE, RMSE e R² por mês
    comparando predição com dados reais de 2025.
    """
    selic = 0.0108  # SELIC média jan-dez/2025

    for algoritmo in ["Random Forest", "Gradient Boosting"]:
        sigla_slug = "RF" if algoritmo == "Random Forest" else "GB"

        print(f"\n\n{'#'*72}")
        print(f"  VALIDAÇÃO COM DADOS REAIS 2025 — {algoritmo.upper()} ({sigla_slug})")
        print(f"  Comparação: DY real brapi.dev Pro vs DY previsto pelo modelo")
        print(f"{'#'*72}")

        todos_erros_abs = []
        todos_erros_sq  = []
        todos_real      = []
        todos_prev      = []

        for sigla, real_2025 in REAL_2025_VALIDACAO.items():
            seg = SIGLA_SEG_VALIDACAO.get(sigla)
            if not seg:
                continue

            info     = meta_global.get(seg, {})
            num_cols = info.get("num_cols", ["DY_lag1","DY_lag2","DY_lag3"])
            cat_cols = info.get("cat_cols", [])
            pipe     = (modelos_dict.get(seg) or {}).get(algoritmo)
            lags     = REAL_2024_LAGS.get(sigla, [])

            if pipe is None or not lags:
                continue

            previstos = prever_recursivo(pipe, num_cols, cat_cols, lags, None, selic, n=12)
            if previstos is None:
                continue

            # ── Tabela MAE por mês ────────────────────────────────────────────
            print(f"\n{'─'*68}")
            print(f"  {sigla} ({seg}) — {algoritmo}")
            print(f"{'─'*68}")

            # MAE
            print(f"\n  ► MAE — Erro Absoluto por Mês")
            print(f"  {'Mês':10s} {'DY Real':>12} {'DY Previsto':>12} {'Erro Absoluto':>14}")
            print(f"  {'':10s} {'(% a.m.)':>12} {'(% a.m.)':>12} {'(% a.m.)':>14}")
            print(f"  {'-'*52}")

            erros_abs, erros_sq = [], []
            reais_validos, prev_validos = [], []

            for i, (real, prev) in enumerate(zip(real_2025, previstos)):
                if real is None:
                    continue
                real_pct = real * 100
                prev_pct = prev * 100
                erro_abs = abs(real_pct - prev_pct)
                erro_sq  = (real_pct - prev_pct) ** 2

                erros_abs.append(erro_abs)
                erros_sq.append(erro_sq)
                reais_validos.append(real_pct)
                prev_validos.append(prev_pct)
                todos_erros_abs.append(erro_abs)
                todos_erros_sq.append(erro_sq)
                todos_real.append(real_pct)
                todos_prev.append(prev_pct)

                print(f"  {MESES_2025[i]:10s} {real_pct:>11.4f}% {prev_pct:>11.4f}% {erro_abs:>13.4f}%")

            if erros_abs:
                mae  = np.mean(erros_abs)
                print(f"  {'-'*52}")
                print(f"  {'Média':10s} {'—':>12} {'—':>12} {'MAE = '+f'{mae:.4f}%':>14}")

            # RMSE
            if erros_sq:
                rmse = np.sqrt(np.mean(erros_sq))
                print(f"\n  ► RMSE — Raiz do Erro Quadrático Médio")
                print(f"  {'─'*52}")
                print(f"  RMSE = √( média( (DY_real − DY_prev)² ) )")
                print(f"  RMSE = √( {np.mean(erros_sq):.6f} )")
                print(f"  RMSE = {rmse:.4f}% a.m.")
                print(f"\n  Interpretação: em média, o modelo erra {rmse:.4f}% a.m.")
                print(f"  penalizando mais os erros grandes (ex: meses atípicos)")

            # R²
            if reais_validos and len(reais_validos) >= 2:
                media_real  = np.mean(reais_validos)
                ss_res = sum((r - p) ** 2 for r, p in zip(reais_validos, prev_validos))
                ss_tot = sum((r - media_real) ** 2 for r in reais_validos)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")

                print(f"\n  ► R² — Coeficiente de Determinação")
                print(f"  {'─'*52}")
                print(f"  R² = 1 − (erro do modelo / erro da média)")
                print(f"  R² = 1 − ({ss_res:.6f} / {ss_tot:.6f})")
                print(f"  R² = {r2:.4f}")
                if r2 >= 0.3:
                    interp = "Bom — modelo captura padrão temporal do fundo"
                elif r2 >= 0.1:
                    interp = "Moderado — modelo supera a média, com limitações"
                elif r2 >= 0:
                    interp = "Fraco — modelo mal supera prever sempre a média"
                else:
                    interp = "Negativo — modelo pior que prever sempre a média"
                print(f"  Interpretação: {interp}")

        # ── Resumo geral do algoritmo ─────────────────────────────────────────
        if todos_erros_abs:
            mae_geral  = np.mean(todos_erros_abs)
            rmse_geral = np.sqrt(np.mean(todos_erros_sq))
            if len(todos_real) >= 2:
                media_geral = np.mean(todos_real)
                ss_res_g = sum((r-p)**2 for r,p in zip(todos_real, todos_prev))
                ss_tot_g = sum((r-media_geral)**2 for r in todos_real)
                r2_geral = 1 - (ss_res_g / ss_tot_g) if ss_tot_g > 0 else float("nan")
            else:
                r2_geral = float("nan")

            print(f"\n{'='*68}")
            print(f"  RESUMO GERAL — {algoritmo} ({sigla_slug})")
            print(f"  (média sobre todos os fundos e meses com dados reais 2025)")
            print(f"{'─'*68}")
            print(f"  MAE  = {mae_geral:.4f}% a.m.  → erro médio absoluto por mês")
            print(f"  RMSE = {rmse_geral:.4f}% a.m.  → penaliza erros grandes")
            print(f"  R²   = {r2_geral:.4f}       → capacidade preditiva geral")
            print(f"{'='*68}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arquivo", "-a", default="dataset_fiis_2019_2024_brapi_v2.xlsx")
    parser.add_argument("--validar", action="store_true",
                        help="Exibe validação detalhada com dados reais de 2025")
    args = parser.parse_args()
    treinar(args.arquivo)

    if args.validar:
        # Recarrega modelos após treino
        import joblib
        from pathlib import Path
        MODEL_DIR = Path("modelo")
        meta_path = MODEL_DIR / "meta.json"
        with open(meta_path, encoding="utf-8") as f:
            meta_reloaded = json.load(f)
        meta_global_reload = meta_reloaded.get("modelos_por_segmento", {})

        ALGORITMOS_SLUG = {
            "Random Forest":    "random_forest",
            "Gradient Boosting":"gradient_boosting",
        }
        SEGMENTOS = ["Logistico","Shoppings","Escritorios","Lajes Corporativas","Hibrido"]
        modelos_reload = {}
        for seg in SEGMENTOS:
            slug_seg = seg.lower().replace(" ","_").replace("/","_")
            modelos_reload[seg] = {}
            for nome, slug_mod in ALGORITMOS_SLUG.items():
                pkl = MODEL_DIR / f"modelo_{slug_seg}_{slug_mod}.pkl"
                if pkl.exists():
                    modelos_reload[seg][nome] = joblib.load(pkl)

        imprimir_validacao_2025(meta_global_reload, modelos_reload)