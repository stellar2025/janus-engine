import os
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import sqlite3
import requests
import warnings
warnings.filterwarnings('ignore')

# CONFIGURACIÓN DE CREDENCIALES
FRED_API_KEY = os.environ.get('FRED_API_KEY', '37016f0e7a9489663c246fe275a8f8b4')
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '8045615537:AAHljJ57Y_-kn-j7A3SLHfbaRqj61y3Tntg')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '1816896290')

DB_NAME = '/tmp/market_intelligence.db' # /tmp es writable en GitHub Actions

FEATURES = ['vix', 'hy_spread', 'ratio_xlf_spy', 'curve_slope', 'sofr']
WEIGHTS = np.array([1.0, 3.0, 3.0, 1.0, 1.0]) 
LOOKBACK = 20
THRESHOLD = 6.5 

class InstitutionalEngine:
    def __init__(self):
        self.weights_rep = np.tile(WEIGHTS, LOOKBACK)

    def ingest(self):
        print("📡 Sincronizando fuentes...")
        
        # 1. FRED Data
        f_df = pd.DataFrame()
        try:
            fred = Fred(api_key=FRED_API_KEY.strip())
            series_map = {'vix': 'VIXCLS', 'hy_spread': 'BAMLH0A0HYM2', 
                         'dgs2': 'DGS2', 'dgs10': 'DGS10', 'sofr': 'SOFR'}
            frames = []
            for name, code in series_map.items():
                s = fred.get_series(code)
                if not s.empty:
                    s.name = name
                    frames.append(s)
            if frames:
                f_df = pd.concat(frames, axis=1)
                print(f"✅ FRED: {len(f_df)} registros.")
        except Exception as e:
            print(f"⚠️ FRED Error: {str(e)[:60]}")

        # 2. Yahoo Data (con manejo de bloqueo de DB)
        close = pd.DataFrame()
        try:
            yf_data = yf.download(['XLF', 'KBE', 'SPY'], start="2000-01-01", progress=False)
            if isinstance(yf_data.columns, pd.MultiIndex):
                try: close = yf_data.xs('Adj Close', level='Price', axis=1)
                except: close = yf_data.xs('Close', level='Price', axis=1)
            else:
                close = yf_data['Close'] if 'Close' in yf_data.columns else yf_data
            
            close.columns = [c.lower() for c in close.columns]
            print(f"✅ Yahoo: {len(close)} registros.")
        except Exception as e:
            print(f"⚠️ Yahoo Error: {str(e)[:60]}")

        if f_df.empty and close.empty:
            raise ValueError("Sin datos de ninguna fuente.")

        df = pd.concat([f_df, close], axis=1)
        
        # Feature Engineering seguro
        if 'xlf' in df.columns and 'spy' in df.columns:
            df['ratio_xlf_spy'] = df['xlf'] / df['spy']
        if 'dgs10' in df.columns and 'dgs2' in df.columns:
            df['curve_slope'] = df['dgs10'] - df['dgs2']
            
        # CORRECCIÓN 1: Sintaxis compatible con Pandas >= 2.0
        df = df.ffill() 
        df.sort_index(inplace=True)
        df.index.name = 'date'
        
        # Guardar en DB (usando WAL mode para evitar locks)
        conn = sqlite3.connect(DB_NAME)
        conn.execute("PRAGMA journal_mode=WAL;")
        df.to_sql('market_synced_data', conn, if_exists='replace', index=True)
        conn.close()
        print(f"💾 BD guardada: {len(df)} filas.")
        return df

    def analyze_live(self, df):
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
            
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors import NearestNeighbors
        
        scaler = StandardScaler()
        h_scaled = scaler.fit_transform(h_states)
        c_scaled = scaler.transform([curr_vec])
        
        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(h_scaled)
        dist, idx = knn.kneighbors(c_scaled)
        
        twins = [{'date': h_dates[idx[0][k]], 'dist': dist[0][k]} for k in range(5)]
        return twins, dist[0][0]

    def send_alert(self, dist, has_macro):
        if has_macro:
            status = "🟢 NORMAL" if dist > THRESHOLD else "🔴 ALERTA SISTÉMICA"
            msg = f"🤖 *Janus Engine*\n📅 {pd.Timestamp.now().strftime('%Y-%m-%d')}\n Dist: {dist:.2f}\n{status}\n📊 Macro: ✅"
        else:
            msg = f"🤖 *Janus Engine*\n📅 {pd.Timestamp.now().strftime('%Y-%m-%d')}\n⚠️ Solo precios\n📊 FRED: ⏳\n Alertas limitadas"
            
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={msg}&parse_mode=Markdown"
            requests.get(url)
            print("✅ Telegram enviado.")
        except Exception as e:
            print(f"⚠️ Telegram Error: {str(e)[:50]}")

    def run(self):
        # CORRECCIÓN 2: Manejo robusto de DB vacía/corrupta
        conn = sqlite3.connect(DB_NAME)
        conn.execute("PRAGMA journal_mode=WAL;")
        try: 
            df = pd.read_sql_query("SELECT * FROM market_synced_data", conn, index_col='date', parse_dates=['date'])
            if df.empty: raise Exception("DB vacía")
        except: 
            print("⚠️ BD no encontrada o vacía. Generando datos frescos...")
            df = self.ingest()
        conn.close()
        
        has_macro = all(f in df.columns for f in ['vix', 'hy_spread', 'sofr'])
        twins, dist = self.analyze_live(df)
        self.send_alert(dist, has_macro)
        print(f"🏁 Janus Engine completado. Distancia: {dist:.2f}")

if __name__ == "__main__":
    engine = InstitutionalEngine()
    engine.run()
