import os
import time  # NECESARIO PARA LA LÓGICA DE REINTENTOS
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import sqlite3
import requests
import warnings
warnings.filterwarnings('ignore')

# CONFIGURACIÓN SEGURA PARA GITHUB ACTIONS Y COLAB
FRED_API_KEY = os.environ.get('FRED_API_KEY', '37016f0e7a9489663c246fe275a8f8b4')
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '8045615537:AAHljJ57Y_-kn-j7A3SLHfbaRqj61y3Tntg')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '1816896290')

DB_NAME = '/tmp/market_intelligence.db' # /tmp es writable en GitHub Actions
FEATURES = ['vix', 'hy_spread', 'ratio_xlf_spy', 'curve_slope', 'sofr']
BASE_THRESHOLD = 6.5 
LOOKBACK = 20

class InstitutionalEngine:
    def __init__(self):
        pass

    def ingest(self):
        print("📡 Sincronizando fuentes...")
        
        # 1. FRED Data CON RETRY LOGIC PARA EVITAR BLOQUEOS DE IP
        f_df = pd.DataFrame()
        max_retries = 3
        
        for attempt in range(max_retries):
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
                    print(f"✅ FRED: {len(f_df)} registros cargados.")
                    break  # Éxito, salir del bucle de reintentos
                    
            except Exception as e:
                wait_time = 2 ** (attempt + 1)  # Espera exponencial: 2s, 4s, 8s
                print(f"️ FRED Intento {attempt+1}/{max_retries} fallido: {str(e)[:60]}")
                if attempt < max_retries - 1:
                    print(f"   ⏳ Esperando {wait_time}s antes de reintentar...")
                    time.sleep(wait_time)
        else:
            # Este bloque solo se ejecuta si el bucle NO se rompió (todos los intentos fallaron)
            print("❌ FRED agotó todos los intentos. Continuando solo con Yahoo.")

        # 2. Yahoo Data
        close = pd.DataFrame()
        try:
            yf_data = yf.download(['XLF', 'KBE', 'SPY'], start="2000-01-01", progress=False)
            if isinstance(yf_data.columns, pd.MultiIndex):
                try: close = yf_data.xs('Adj Close', level='Price', axis=1)
                except: close = yf_data.xs('Close', level='Price', axis=1)
            else:
                close = yf_data['Close'] if 'Close' in yf_data.columns else yf_data
            
            close.columns = [c.lower() for c in close.columns]
            print(f"✅ Yahoo: {len(close)} registros cargados.")
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
            
        # Sintaxis compatible con Pandas >= 2.0
        df = df.ffill() 
        df.sort_index(inplace=True)
        df.index.name = 'date'
        
        # Guardar en DB con WAL mode para evitar locks en CI/CD
        conn = sqlite3.connect(DB_NAME)
        conn.execute("PRAGMA journal_mode=WAL;")
        df.to_sql('market_synced_data', conn, if_exists='replace', index=True)
        conn.close()
        print(f"💾 BD guardada: {len(df)} filas.")
        return df

    def analyze_live(self, df):
        """Análisis adaptativo con normalización por dimensionalidad."""
        
        # Filtrado estricto: Solo columnas existentes Y con varianza real (>0)
        available_features = [f for f in FEATURES 
                             if f in df.columns and df[f].std() > 0]
        
        n_features = len(available_features)
        if n_features == 0:
            raise ValueError("Sin datos con varianza para analizar.")
            
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
        
        # CORRECCIÓN MATEMÁTICA: Normalizar distancia por sqrt(n)
        raw_dist = dist[0][0]
        normalized_dist = raw_dist / np.sqrt(n_features) 
        
        # Umbral dinámico: escala proporcionalmente para mantener significancia
        dynamic_threshold = BASE_THRESHOLD / np.sqrt(len(FEATURES)) * np.sqrt(n_features)
        
        twins = [{'date': h_dates[idx[0][k]], 'dist': normalized_dist} for k in range(5)]
        return twins, normalized_dist, dynamic_threshold, n_features

    def send_alert(self, dist, threshold, n_feat, execution_time, data_date):
        has_full_data = (n_feat == len(FEATURES))
        
        if has_full_data:
            status = "🟢 NORMAL" if dist > threshold else "🔴 ALERTA SISTÉMICA"
            msg = f"🤖 *Janus Engine*\n⏱️ Ejecutado: {execution_time}\n📅 Datos hasta: {data_date}\n Dist: {dist:.2f} (umbral: {threshold:.2f})\n{status}\n📊 Macro: ✅ ({n_feat}/5 vars)"
        else:
            status = " OBSERVACIÓN" if dist > threshold else "️ SEGUIMIENTO"
            msg = f"🤖 *Janus Engine*\n️ Ejecutado: {execution_time}\n📅 Datos hasta: {data_date}\n Dist: {dist:.2f} (ajustado: {threshold:.2f})\n{status}\n📊 Parcial: {n_feat}/5 vars\n⏳ FRED pendiente"
            
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={msg}&parse_mode=Markdown"
        response = requests.get(url, timeout=10)
        
        # FORZAR ERROR VISIBLE SI TELEGRAM RECHAZA EL MENSAJE
        if response.status_code != 200:
            raise Exception(f"TELEGRAM FALLÓ: Código {response.status_code} - {response.text}")
        
        print("✅ Telegram enviado correctamente.")

    def run(self):
        # FORZAR ACTUALIZACIÓN DIARIA DE DATOS FRESCOS
        print("🔄 Actualizando datos frescos antes del análisis...")
        df = self.ingest()
        
        twins, dist, dyn_thresh, n_feat = self.analyze_live(df)
        
        # Timestamps reales para el mensaje
        exec_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')
        data_date = df.index[-1].strftime('%Y-%m-%d')
        
        self.send_alert(dist, dyn_thresh, n_feat, exec_time, data_date)
        print(f"🏁 Janus Engine completado. Distancia normalizada: {dist:.2f} | Variables activas: {n_feat}")

if __name__ == "__main__":
    engine = InstitutionalEngine()
    engine.run()
