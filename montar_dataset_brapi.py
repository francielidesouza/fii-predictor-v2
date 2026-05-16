"""
montar_dataset_brapi.py
-----------------------
Coleta dados históricos de FIIs da brapi.dev Pro e consolida em um único
dataset Excel cobrindo o período de janeiro/2019 a dezembro/2025.

Variáveis coletadas por fundo e por mês:
    - Dividendos_Yield  (dividendYield1m)
    - P_VP              (priceToNav)

Fonte macro:
    - SELIC mensal      (BCB SGS série 4390)

Uso:
    1. Configure o token no arquivo .env:
       BRAPI_TOKEN=seu_token_aqui

    2. Execute:
       python montar_dataset_brapi.py

    3. Saída:
       dataset_fiis_2019_2025.xlsx
"""

import os, time, requests, json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TOKEN   = os.getenv("BRAPI_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
BASE    = "https://brapi.dev/api/v2/fii"

DATA_INI = "2019-01-01"
DATA_FIM = "2025-12-31"
LOTE     = 20          # máximo de símbolos por requisição
SAIDA    = "dataset_fiis_2019_2025.xlsx"
PAUSE    = 0.4         # segundos entre requisições


# ── Metadados dos segmentos (fonte: investidor10.com.br) ─────────────────────
# Fundos Multicategoria remapeados manualmente para seus segmentos corretos
SEGMENTO_MANUAL = {
    "KNRI11": "Hibrido",    "ALZR11": "Hibrido",
    "BRCR11": "Hibrido",    "RBRD11": "Hibrido",
    "PATB11": "Hibrido",    "VERE11": "Hibrido",
    "BRRI11": "Hibrido",    "MXRF11": "Titulos e Val. Mob.",
    "KNCR11": "Titulos e Val. Mob.", "KNHY11": "Titulos e Val. Mob.",
    "KNIP11": "Titulos e Val. Mob.", "HGCR11": "Titulos e Val. Mob.",
    "IRDM11": "Titulos e Val. Mob.", "CPTS11": "Titulos e Val. Mob.",
}


def buscar_lista_fiis() -> list:
    """Retorna lista de siglas de todos os FIIs disponíveis na brapi."""
    print("[→] Buscando lista de FIIs...", end=" ")
    r = requests.get(f"{BASE}/list", headers=HEADERS, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Erro ao buscar lista: HTTP {r.status_code}")
    siglas = [f["stock"] for f in r.json().get("indexes", [])]
    print(f"✓ {len(siglas)} fundos encontrados")
    return siglas


def buscar_historico_lote(siglas: list) -> list:
    """
    Busca histórico mensal de DY e P/VP para um lote de até 20 siglas.
    Retorna lista de dicts com colunas padronizadas.
    """
    simbolos = ",".join(siglas)
    params = {
        "symbols":   simbolos,
        "startDate": DATA_INI,
        "endDate":   DATA_FIM,
        "sortOrder": "asc",
    }
    r = requests.get(
        f"{BASE}/indicators/history",
        headers=HEADERS,
        params=params,
        timeout=60,
    )
    if r.status_code != 200:
        print(f"  ⚠ HTTP {r.status_code} para lote {siglas[:3]}...")
        return []

    dados = r.json()
    linhas = []

    for item in dados.get("results", []):
        sigla   = item.get("stock", "")
        seg     = item.get("segment") or SEGMENTO_MANUAL.get(sigla, "Outros")
        tipo    = item.get("fundType") or "Tijolo"
        history = item.get("history", [])

        for mes in history:
            data_ref = mes.get("referenceDate", "")[:7]  # YYYY-MM
            dy       = mes.get("dividendYield1m")
            pvp      = mes.get("priceToNav")

            if not data_ref or dy is None:
                continue

            linhas.append({
                "Data":             pd.to_datetime(data_ref + "-01"),
                "Sigla":            sigla,
                "Segmento":         seg,
                "Tipo_do_Fundo":    tipo,
                "Dividendos_Yield": round(float(dy), 6),
                "P_VP":             round(float(pvp), 4) if pvp else None,
            })

    return linhas


def buscar_selic() -> dict:
    """Busca SELIC mensal do BCB SGS série 4390 (2019–2025)."""
    print("[→] Buscando SELIC do BCB SGS 4390...", end=" ")
    selic = {}
    try:
        r = requests.get(
            "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4390/dados",
            params={
                "formato":      "json",
                "dataInicial":  "01/01/2019",
                "dataFinal":    "31/12/2025",
            },
            timeout=20,
        )
        if r.status_code == 200:
            for item in r.json():
                mes = pd.to_datetime(
                    item["data"], format="%d/%m/%Y"
                ).strftime("%Y-%m")
                selic[mes] = round(float(item["valor"]) / 100, 6)
            print(f"✓ {len(selic)} meses")
        else:
            print(f"⚠ HTTP {r.status_code}")
    except Exception as e:
        print(f"⚠ falhou ({e})")
    return selic


def montar_dataset():
    print("\n" + "="*60)
    print("  FII Dataset Builder — brapi.dev Pro")
    print(f"  Período: {DATA_INI} → {DATA_FIM}")
    print("="*60 + "\n")

    # 1. Lista de FIIs
    siglas = buscar_lista_fiis()

    # 2. Histórico em lotes
    print(f"[→] Coletando histórico em lotes de {LOTE}...")
    todas_linhas = []
    total_lotes  = (len(siglas) + LOTE - 1) // LOTE

    for i in range(0, len(siglas), LOTE):
        lote  = siglas[i:i+LOTE]
        num   = i // LOTE + 1
        print(f"  Lote {num:3d}/{total_lotes} — {lote[0]}...{lote[-1]}", end=" ")
        linhas = buscar_historico_lote(lote)
        todas_linhas.extend(linhas)
        print(f"→ {len(linhas)} registros")
        time.sleep(PAUSE)

    if not todas_linhas:
        print("[!] Nenhum dado coletado. Verifique o token BRAPI_TOKEN.")
        return

    # 3. Monta DataFrame
    df = pd.DataFrame(todas_linhas)
    df = df.sort_values(["Sigla", "Data"]).reset_index(drop=True)
    print(f"\n[✓] {len(df)} linhas · {df['Sigla'].nunique()} fundos coletados")

    # 4. Adiciona SELIC
    selic = buscar_selic()
    mes_col      = pd.to_datetime(df["Data"]).dt.strftime("%Y-%m")
    df["SELIC"]  = mes_col.map(selic)

    # 5. Corrige ETFs conhecidos
    etfs = ["GLDN11", "SCOO11"]
    df   = df[~df["Sigla"].isin(etfs)]
    print(f"[✓] ETFs removidos: {etfs}")

    # 6. Resumo por segmento
    print(f"\n[✓] Distribuição final por segmento:")
    for seg, grp in df.groupby("Segmento"):
        n     = grp["Sigla"].nunique()
        anos  = f"{grp['Data'].min().strftime('%Y-%m')} → {grp['Data'].max().strftime('%Y-%m')}"
        print(f"    {seg:30s}: {n:3d} fundos | {anos}")

    # 7. Salva Excel
    df.to_excel(SAIDA, index=False)
    print(f"\n[✓] Salvo: {SAIDA}")
    print(f"    {len(df)} linhas · {df['Sigla'].nunique()} fundos · "
          f"{df['Data'].min().strftime('%Y-%m')} → {df['Data'].max().strftime('%Y-%m')}")
    print("\n" + "="*60)


if __name__ == "__main__":
    montar_dataset()