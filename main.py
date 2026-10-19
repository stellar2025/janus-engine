import os
import time
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

DB_NAME = '/tmp/market_intelligence.db'
FEATURES = ['vix', 'hy_spread', 'ratio_xlf_spy', 'curve_slope', 'sofr']
BASE_THRESHOLD = 6.5 
LOOKBACK = 20

class InstitutionalEngine:
    def __init__(self):
        pass

    def ingest(self):
        print("📡 Sincronizando fuentes...")
        
        # PRIORIDAD 1: LEER CSV REAL SUBIDO DESDE GOOGLE APPS SCRIPT
        f_df = pd.DataFrame()
        csv_path = '/home/runner/work/janus-engine/janus-engine/data/fred_data_real.csv'
        data_source = "PROXIES_YAHOO" 
        
        try:
            if os.path.exists(csv_path):
                temp_df = pd.read_csv(csv_path, index_col='DATE', parse_dates=['DATE'])
                last_date = temp_df.index[-1]
                days_old = (pd.Timestamp.now() - last_date).days
                
                if days_old <= 3:
                    f_df = temp_df
                    data_source = "FRED_REAL_CSV"
                    print(f"✅ FRED REAL (CSV): {len(f_df)} registros | Último: {last_date.strftime('%Y-%m-%d')}")
                else:
                    print(f"⚠️ CSV de FRED desactualizado ({days_old} días). Usando fallback.")
            else:
                print("⚠️ CSV de FRED no encontrado en /data/. Usando fallback.")
                
        except Exception as e:
            print(f"⚠️ Error leyendo CSV FRED: {str(e)[:60]}. Usando fallback.")

        # PRIORIDAD 2: FALLBACK A PROXIES YAHOO SI NO HAY CSV VÁLIDO
        if f_df.empty:
            print("🔄 Activando proxies macro vía Yahoo Finance...")
            try:
                yf_macro = yf.download(['^VIX', '^TNX', '^IRX'], start="2000-01-01", progress=False)
                if isinstance(yf_macro.columns, pd.MultiIndex):
                    yf_macro = yf_macro.xs('Close', level='Price', axis=1)
                
                rename_map = {'^VIX': 'vix', '^TNX': 'dgs10', '^IRX': 'dgs2'}
                yf_macro.rename(columns=rename_map, inplace=True)
                
                if 'dgs10' in yf_macro.columns and 'dgs2' in yf_macro.columns:
                    yf_macro['curve_slope'] = yf_macro['dgs10'] - yf_macro['dgs2']
                    yf_macro['sofr'] = yf_macro['dgs2'] / 100
                    yf_macro['hy_spread'] = yf_macro['dgs10'] - yf_macro['dgs2']
                
                f_df = yf_macro[['vix', 'hy_spread', 'curve_slope', 'sofr']].copy()
                data_source = "PROXIES_YAHOO"
                print(f"⚠️ Proxies Macro (Yahoo): {len(f_df)} registros cargados.")
                
            except Exception as e:
                print(f"❌ Proxies Macro fallaron: {str(e)[:60]}")

        # 2. Yahoo Data (Precios)
        close = pd.DataFrame()
        try:
            yf_data = yf.download(['XLF', 'KBE', 'SPY'], start="2000-01-01", progress=False)
            if isinstance(yf_data.columns, pd.MultiIndex):
                try: close = yf_data.xs('Adj Close', level='Price', axis=1)
                except: close = yf_data.xs('Close', level='Price', axis=1)
            else:
                close = yf_data['Close'] if 'Close' in yf_data.columns else yf_data
            
            close.columns = [c.lower() for c in close.columns]
            print(f"✅ Yahoo Precios: {len(close)} registros cargados.")
        except Exception as e:
            print(f"️ Yahoo Error: {str(e)[:60]}")

        if f_df.empty and close.empty:
            raise ValueError("Sin datos de ninguna fuente.")

        df = pd.concat([f_df, close], axis=1)
        
        # Feature Engineering seguro
        if 'xlf' in df.columns and 'spy' in df.columns:
            df['ratio_xlf_spy'] = df['xlf'] / df['spy']
        if 'dgs10' in df.columns and 'dgs2' in df.columns:
            df['curve_slope'] = df['dgs10'] - df['dgs2']
            
        # LIMPIEZA ESTRICTA POST-FUSIÓN (Auditoría completada - Sin bugs)
        # 1. Eliminar filas donde TODAS las columnas macro sean nulas
        macro_cols = [c for c in ['vix', 'hy_spread', 'dgs2', 'dgs10', 'sofr'] if c in df.columns]
        if macro_cols:
            df = df.dropna(subset=macro_cols, how='all')
        
        # 2. Asegurar que NO hay datos futuros respecto al último cierre de SPY
        if 'spy' in df.columns:
            last_spy_idx = df['spy'].last_valid_index()
            if last_spy_idx is not None:
                df = df.loc[:last_spy_idx]
        
        # 3. Rellenar huecos puntuales DE MANERA SEGURA
        df = df.ffill()
        
        # 4. VALIDACIÓN FINAL: Solo eliminamos filas que siguen incompletas DESPUÉS del relleno
        df = df.dropna() 
        
        df.sort_index(inplace=True)
        df.index.name = 'date'
        
        conn = sqlite3.connect(DB_NAME)
        conn.execute("PRAGMA journal_mode=WAL;")
        df.to_sql('market_synced_data', conn, if_exists='replace', index=True)
        conn.close()
        print(f"💾 BD guardada: {len(df)} filas. Fuente: {data_source}")
        return df

    def analyze_live(self, df):
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
        
        raw_dist = dist[0][0]
        normalized_dist = raw_dist / np.sqrt(n_features) 
        dynamic_threshold = BASE_THRESHOLD / np.sqrt(len(FEATURES)) * np.sqrt(n_features)
        
        twins = [{'date': h_dates[idx[0][k]], 'dist': normalized_dist} for k in range(5)]
        return twins, normalized_dist, dynamic_threshold, n_features

    def send_alert(self, dist, threshold, n_feat, execution_time, data_date):
        has_full_data = (n_feat == len(FEATURES))
        
        if has_full_data:
            status = "🟢 NORMAL" if dist > threshold else "🔴 ALERTA SISTÉMICA"
            msg = f" *Janus Engine*\n⏱️ Ejecutado: {execution_time}\n📅 Datos hasta: {data_date}\n Dist: {dist:.2f} (umbral: {threshold:.2f})\n{status}\n📊 Macro: ✅ ({n_feat}/5 vars)"
        else:
            status = "🔍 OBSERVACIÓN" if dist > threshold else "⚠️ SEGUIMIENTO"
            msg = f"🤖 *Janus Engine*\n⏱️ Ejecutado: {execution_time}\n📅 Datos hasta: {data_date}\n Dist: {dist:.2f} (ajustado: {threshold:.2f})\n{status}\n📊 Parcial: {n_feat}/5 vars\n FRED pendiente"
            
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={TELEGRAM_CHAT_ID}&text={msg}&parse_mode=Markdown"
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"TELEGRAM FALLÓ: Código {response.status_code} - {response.text}")
        
        print("✅ Telegram enviado correctamente.")

    def run(self):
        print("🔄 Actualizando datos frescos antes del análisis...")
        df = self.ingest()
        
        twins, dist, dyn_thresh, n_feat = self.analyze_live(df)
        
        exec_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')
        data_date = df.index[-1].strftime('%Y-%m-%d')
        
        self.send_alert(dist, dyn_thresh, n_feat, exec_time, data_date)
        print(f"🏁 Janus Engine completado. Distancia normalizada: {dist:.2f} | Variables activas: {n_feat}")

if __name__ == "__main__":
    engine = InstitutionalEngine()
    engine.run()
