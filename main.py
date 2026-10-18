# ==============================================================================
# JANUS ENGINE — INSTITUTIONAL GRADE (CORREGIDO)
# ==============================================================================

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import sqlite3
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# CONFIGURACIÓN SEGURA PARA GITHUB ACTIONS Y LOCAL
FRED_API_KEY = os.environ.get('FRED_API_KEY', '37016f0e7a9489663c246fe275a8f8b4')
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '8045615537:AAHljJ57Y_-kn-j7A3SLHfbaRqj61y3Tntg')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '1816896290')
DB_NAME = '/content/market_intelligence.db'

FEATURES = ['vix', 'hy_spread', 'ratio_xlf_spy', 'curve_slope', 'sofr']
WEIGHTS = np.array([1.0, 3.0, 3.0, 1.0, 1.0])
LOOKBACK = 20
THRESHOLD = 6.5

class InstitutionalEngine:
    def __init__(self):
        self.weights_rep = np.tile(WEIGHTS, LOOKBACK)

    def ingest(self):
        """Ingesta robusta con corrección para la nueva API de Yahoo."""
        print("📡 Sincronizando fuentes...")

        # 1. FRED Data
        f_df = pd.DataFrame()
        try:
            fred = Fred(api_key=FRED_API_KEY)
            series_map = {
                'vix': 'VIXCLS', 'hy_spread': 'BAMLH0A0HYM2',
                'dgs2': 'DGS2', 'dgs10': 'DGS10', 'sofr': 'SOFR', 'diesel': 'GASDESLSW'
            }
            frames = []
            for name, code in series_map.items():
                s = fred.get_series(code)
                if not s.empty:
                    s.name = name
                    frames.append(s)
            if frames:
                f_df = pd.concat(frames, axis=1)
                print(f"✅ FRED: {len(f_df)} registros cargados.")
        except Exception as e:
            print(f"❌ Error en FRED: {e}")

        # 2. Yahoo Data
        close = pd.DataFrame()
        try:
            df_yf = yf.download(['XLF', 'KBE', 'SPY'], start="1990-01-01", progress=False)

            if isinstance(df_yf.columns, pd.MultiIndex):
                if 'Adj Close' in df_yf.columns.get_level_values('Price'):
                    close = df_yf.xs('Adj Close', level='Price', axis=1)
                else:
                    close = df_yf.xs('Close', level='Price', axis=1)
            else:
                close = df_yf['Adj Close'] if 'Adj Close' in df_yf.columns else df_yf['Close']

            close.columns = [c.lower() for c in close.columns]
            print(f"✅ Yahoo: {len(close)} registros cargados.")
        except Exception as e:
            print(f"❌ Error en Yahoo: {e}")
            import traceback
            traceback.print_exc()

        # 3. Merge
        if f_df.empty and close.empty:
            raise ValueError("No se pudieron cargar datos de ninguna fuente.")

        df = pd.concat([f_df, close], axis=1)

        # Feature Engineering
        if 'xlf' in df.columns and 'spy' in df.columns:
            df['ratio_xlf_spy'] = df['xlf'] / df['spy']
            print("✅ Ratio XLF/SPY calculado.")
        else:
            print("⚠️ No se pudo calcular el ratio (faltan XLF o SPY).")

        if 'dgs10' in df.columns and 'dgs2' in df.columns:
            df['curve_slope'] = df['dgs10'] - df['dgs2']

        df.fillna(method='ffill', inplace=True)
        df.sort_index(inplace=True)
        df.index.name = 'date'

        conn = sqlite3.connect(DB_NAME)
        df.to_sql('market_synced_data', conn, if_exists='replace', index=True)
        conn.close()
        return df

    def analyze_live(self, df):
        """Análisis en vivo."""
        available_features = [f for f in FEATURES if f in df.columns]
        if len(available_features) < 2:
            raise ValueError("Insuficientes datos para analizar.")

        changes = df[available_features].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
        mean_h = changes.mean(); std_h = changes.std(); std_h[std_h==0]=1

        curr_win = changes.iloc[-LOOKBACK:]
        curr_vec = ((curr_win - mean_h) / std_h).values.flatten()

        h_states, h_dates = [], []
        for i in range(len(changes) - LOOKBACK):
            w = changes.iloc[i:i+LOOKBACK]
            h_states.append(((w - mean_h) / std_h).values.flatten())
            h_dates.append(changes.index[i+LOOKBACK-1])

        scaler = StandardScaler()
        h_scaled = scaler.fit_transform(h_states)
        c_scaled = scaler.transform([curr_vec])

        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(h_scaled)
        dist, idx = knn.kneighbors(c_scaled)

        twins = [{'date': h_dates[idx[0][k]], 'dist': dist[0][k]} for k in range(5)]
        return twins, dist[0][0]

    def alert(self, dist):
        status = "🟢 NORMAL" if dist > THRESHOLD else "🔴 ALERTA SISTÉMICA"
        msg = f"🤖 *Janus Engine*\n📅 {pd.Timestamp.now().strftime('%Y-%m-%d')}\n📏 Dist: {dist:.2f}\n{status}"
        try:
            requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={msg}&parse_mode=Markdown")
            print("✅ Alerta enviada a Telegram.")
        except Exception as e:
            print(f"⚠️ Fallo en Telegram: {e}")

    def run(self):
        conn = sqlite3.connect(DB_NAME)
        try:
            df = pd.read_sql_query("SELECT * FROM market_synced_data", conn, index_col='date', parse_dates=['date'])
            if df.empty: raise Exception("DB vacía")
        except:
            df = self.ingest()
        conn.close()

        twins, dist = self.analyze_live(df)
        self.alert(dist)

        # Dashboard
        if 'spy' in df.columns:
            fig = make_subplots(rows=2, cols=1, vertical_spacing=0.05, subplot_titles=("SPY vs Gemelos", "Ratio XLF/SPY"))
            fig.add_trace(go.Scatter(x=df.index, y=df['spy'], name="SPY", line=dict(color="gold")), row=1, col=1)

            colors = ['red', 'orange', 'gray', 'lightgray', 'cyan']
            for i, t in enumerate(twins):
                if t['date'] in df.index:
                    t_idx = df.index.get_loc(t['date'])
                    if t_idx + 60 < len(df):
                        t_series = df['spy'].iloc[t_idx:t_idx+60]
                        norm_t = t_series * (df['spy'].iloc[-1] / t_series.iloc[0])
                        fig.add_trace(go.Scatter(x=t_series.index, y=norm_t, name=f"Gemelo {i+1}", line=dict(dash="dash", color=colors[i])), row=1, col=1)

            fig.add_trace(go.Scatter(x=df.index, y=df['ratio_xlf_spy'], name="Ratio Financiero", line=dict(color="cyan")), row=2, col=1)
            fig.update_layout(height=700, title_text=f"JANUS ENGINE | Dist: {dist:.2f} ({'RISK' if dist<THRESHOLD else 'SAFE'})", template="plotly_dark")
            fig.show()
        else:
            print("No hay datos de SPY para graficar.")

# EJECUCIÓN
engine = InstitutionalEngine()
engine.run()
