"""
api.py
------
API FastAPI — FII Predictor v2
Random Forest vs Gradient Boosting | Por segmento | Apenas Tijolo

Uso local:
    uvicorn api:app --reload --port 8000
"""

import json, os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()
MODEL_DIR = Path("modelo")

app = FastAPI(
    title="FII Predictor API",
    description="Predição de DY por segmento — Random Forest vs Gradient Boosting | Apenas Tijolo",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ALGORITMOS_SLUG = {
    "Random Forest":    "random_forest",
    "Gradient Boosting":"gradient_boosting",
}

SEGMENTOS_DISPONIVEIS = [
    "Logistico", "Shoppings", "Escritorios",
    "Lajes Corporativas", "Hibrido"
]

SEGMENTOS_EXCLUIDOS = {
    "Titulos e Val. Mob.": "DY depende do spread privado dos CRIs",
    "FOF":                 "DY depende de outros FIIs e decisões do gestor",
    "Hospital":            "amostra insuficiente (3 fundos)",
    "Varejo":              "amostra insuficiente (3 fundos)",
    "Outros":              "grupo heterogêneo",
}

# ── Carregamento ──────────────────────────────────────────────────────────────

def carregar_artefatos():
    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        raise RuntimeError("modelo/meta.json não encontrado. Execute treinar_modelo.py primeiro.")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    # Carrega todos os modelos disponíveis
    modelos = {}
    for seg in SEGMENTOS_DISPONIVEIS:
        slug_seg = seg.lower().replace(" ","_").replace("/","_")
        modelos[seg] = {}
        for nome, slug_mod in ALGORITMOS_SLUG.items():
            for sufixo in ["", "_sem_pandemia"]:
                pkl = MODEL_DIR / f"modelo_{slug_seg}_{slug_mod}{sufixo}.pkl"
                if pkl.exists():
                    chave = f"{nome}{sufixo}"
                    modelos[seg][chave] = joblib.load(pkl)
                    print(f"[✓] {seg} | {chave}")

    fundos_path = MODEL_DIR / "fundos_recentes.csv"
    fundos_df   = pd.read_csv(fundos_path) if fundos_path.exists() else None

    return meta, modelos, fundos_df


try:
    META, MODELOS, FUNDOS_DF = carregar_artefatos()
except Exception as e:
    print(f"[!] Erro: {e}")
    META, MODELOS, FUNDOS_DF = {}, {}, None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        v = float(val)
        return None if np.isnan(v) else round(v, 6)
    except Exception:
        return None

def _safe_str(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return str(val).strip() or None

def _get_meta_seg(seg: str) -> dict:
    return (META.get("modelos_por_segmento") or {}).get(seg, {})

def _get_num_cat_cols(seg: str):
    info = _get_meta_seg(seg)
    return info.get("num_cols", ["DY_lag1","DY_lag2","DY_lag3"]), info.get("cat_cols", [])

def _get_selic_atual() -> float:
    if FUNDOS_DF is not None and "SELIC" in FUNDOS_DF.columns:
        vals = FUNDOS_DF["SELIC"].dropna()
        if not vals.empty:
            return float(vals.iloc[-1])
    return 0.0104


# ── Schemas ───────────────────────────────────────────────────────────────────

class EntradaSerie(BaseModel):
    sigla:            str            = Field(..., example="HGLG11")
    segmento:         str            = Field(..., example="Logistico")
    dy_lag1:          float          = Field(..., example=0.00704)
    dy_lag2:          float          = Field(..., example=0.007013)
    dy_lag3:          float          = Field(..., example=0.006989)
    pvp:              Optional[float]= Field(None, example=0.8054)
    modelo:           str            = Field("Random Forest",
                                             description="'Random Forest' ou 'Gradient Boosting'")
    n_meses:          int            = Field(12, ge=1, le=12)
    excluir_pandemia: bool           = Field(False)


class InfoFundo(BaseModel):
    sigla:         str
    dy_recente:    Optional[float] = None
    dy_lag1:       Optional[float] = None
    dy_lag2:       Optional[float] = None
    dy_lag3:       Optional[float] = None
    pvp:           Optional[float] = None
    selic:         Optional[float] = None
    segmento:      Optional[str]   = None
    tipo_do_fundo: Optional[str]   = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Status"])
def raiz():
    return {
        "status":                "online",
        "versao":                "2.0.0",
        "algoritmos":            list(ALGORITMOS_SLUG.keys()),
        "segmentos_disponiveis": SEGMENTOS_DISPONIVEIS,
        "segmentos_excluidos":   SEGMENTOS_EXCLUIDOS,
        "tipo_fundo":            "Tijolo",
        "periodo_treino":        "jan/2019–dez/2024",
    }


@app.get("/health", tags=["Status"])
def health():
    mps = (META.get("modelos_por_segmento") or {})
    resumo = {}
    for seg, info in mps.items():
        melhor = info.get("melhor", "")
        sp     = info.get("sem_pandemia", {})
        resumo[seg] = {
            "n_fundos":    info.get("n_fundos"),
            "melhor":      melhor,
            "metricas":    info.get("metricas", {}),
            "sem_pandemia": sp.get("metricas", {}) if sp else None,
        }
    return {
        "status":         "ok",
        "versao_modelo":  META.get("versao"),
        "n_fundos_total": META.get("n_fundos_total"),
        "r2_medio":       META.get("r2_medio"),
        "segmentos":      resumo,
    }


@app.get("/fundos", tags=["Dados"], response_model=list[InfoFundo])
def listar_fundos(segmento: Optional[str] = None):
    if FUNDOS_DF is None:
        return []
    df = FUNDOS_DF.copy()
    if segmento and "Segmento" in df.columns:
        df = df[df["Segmento"] == segmento]
    resultado = []
    for _, row in df.iterrows():
        resultado.append(InfoFundo(
            sigla         = str(row.get("Sigla", "")),
            dy_recente    = _safe_float(row.get("Dividendos_Yield")),
            dy_lag1       = _safe_float(row.get("DY_lag1")),
            dy_lag2       = _safe_float(row.get("DY_lag2")),
            dy_lag3       = _safe_float(row.get("DY_lag3")),
            pvp           = _safe_float(row.get("PVP_lag1")),
            selic         = _safe_float(row.get("SELIC")),
            segmento      = _safe_str(row.get("Segmento")),
            tipo_do_fundo = _safe_str(row.get("Tipo_do_Fundo")),
        ))
    return resultado


@app.get("/segmentos", tags=["Dados"])
def listar_segmentos():
    mps = META.get("modelos_por_segmento") or {}
    disponiveis = []
    for seg in SEGMENTOS_DISPONIVEIS:
        info   = mps.get(seg, {})
        melhor = info.get("melhor", "")
        met_rf = info.get("metricas", {}).get("Random Forest", {})
        met_gb = info.get("metricas", {}).get("Gradient Boosting", {})
        disponiveis.append({
            "segmento":          seg,
            "n_fundos":          info.get("n_fundos"),
            "melhor_algoritmo":  melhor,
            "r2_rf":             met_rf.get("r2"),
            "r2_gb":             met_gb.get("r2"),
            "mape_rf":           met_rf.get("mape"),
            "mape_gb":           met_gb.get("mape"),
        })
    return {
        "disponiveis": disponiveis,
        "excluidos":   [{"segmento": s, "motivo": m} for s, m in SEGMENTOS_EXCLUIDOS.items()],
    }


@app.post("/predict/serie", tags=["Previsão"])
def prever_serie(entrada: EntradaSerie):
    """
    Predição recursiva de N meses.
    Cada mês usa o DY previsto do mês anterior como lag.
    Retorna as séries de ambos os algoritmos para comparação.
    """
    seg = entrada.segmento
    if seg in SEGMENTOS_EXCLUIDOS:
        raise HTTPException(400, detail=f"Segmento '{seg}' excluído: {SEGMENTOS_EXCLUIDOS[seg]}")

    num_cols, cat_cols = _get_num_cat_cols(seg)
    selic              = _get_selic_atual()
    sufixo_sp          = "_sem_pandemia" if entrada.excluir_pandemia else ""
    info               = _get_meta_seg(seg)

    # Seleciona o pipe do algoritmo solicitado
    slug_mod   = ALGORITMOS_SLUG.get(entrada.modelo, "random_forest")
    chave_pipe = f"{entrada.modelo}{sufixo_sp}"
    pipe       = (MODELOS.get(seg) or {}).get(chave_pipe)

    if pipe is None:
        # Fallback: tenta sem sufixo
        pipe = (MODELOS.get(seg) or {}).get(entrada.modelo)
    if pipe is None:
        raise HTTPException(503, detail=f"Modelo '{entrada.modelo}' não disponível para {seg}.")

    met_key = "sem_pandemia" if entrada.excluir_pandemia else "metricas"
    if entrada.excluir_pandemia:
        met = (info.get("sem_pandemia") or {}).get("metricas", {}).get(entrada.modelo, {})
    else:
        met = info.get("metricas", {}).get(entrada.modelo, {})

    # Loop recursivo
    lag1, lag2, lag3 = entrada.dy_lag1, entrada.dy_lag2, entrada.dy_lag3
    pvp = entrada.pvp
    serie = []
    ano, mes_num = 2025, 1

    for i in range(entrada.n_meses):
        row = {"DY_lag1": lag1, "DY_lag2": lag2, "DY_lag3": lag3}
        if "PVP_lag1" in num_cols:
            row["PVP_lag1"] = pvp if pvp is not None else np.nan
        if "SELIC" in num_cols:
            row["SELIC"] = selic

        X = pd.DataFrame([row])[num_cols + cat_cols]
        try:
            dy_pred = float(pipe.predict(X)[0])
        except Exception:
            dy_pred = lag1

        mes_str = f"{ano}-{str(mes_num).zfill(2)}"
        serie.append({
            "mes":             mes_str,
            "dy_previsto":     round(dy_pred, 6),
            "dy_previsto_pct": round(dy_pred * 100, 4),
            "dy_previsto_aa":  round(dy_pred * 12 * 100, 2),
        })

        lag3 = lag2
        lag2 = lag1
        lag1 = dy_pred

        mes_num += 1
        if mes_num > 12:
            mes_num = 1
            ano += 1

    return {
        "sigla":    entrada.sigla.upper(),
        "segmento": seg,
        "modelo":   entrada.modelo,
        "n_meses":  entrada.n_meses,
        "serie":    serie,
        "r2":       met.get("r2"),
        "mape":     met.get("mape"),
        "mae":      met.get("mae"),
        "cv_r2":    met.get("cv_r2"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
