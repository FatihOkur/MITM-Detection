# App.py - YAKUT Mobile App Backend with Real Model Integration
from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json
import time
import random
from datetime import datetime
import threading
import os

app = Flask(__name__)
CORS(app)

# Global değişkenler
is_security_active = False
activity_log = []
model = None
scaler = None
top_features = None
csv_data = None
current_row_index = 0

# Model ve veri yükleme
def load_model_and_data():
    global model, scaler, top_features, csv_data
    try:
        # Model yükle
        model_path = 'Models/yakut_xgboost_model.joblib'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ Model yüklendi: {model_path}")
        else:
            print(f"❌ Model bulunamadı: {model_path}")
            return False
        
        # Scaler yükle (eğer varsa)
        scaler_path = 'Models/yakut_scaler.joblib'
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"✅ Scaler yüklendi: {scaler_path}")
        
        # Top features yükle
        features_path = 'Models/top_10_features.json'
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                top_features = json.load(f)
            print(f"✅ Top features yüklendi: {top_features}")
        else:
            # Eğer top features dosyası yoksa, varsayılan feature isimleri kullan
            print("⚠️ Top features dosyası bulunamadı, tüm özellikler kullanılacak")
            top_features = None
        
        # CSV verisini yükle
        csv_path = 'Data/All_Labelled.csv'
        if os.path.exists(csv_path):
            csv_data = pd.read_csv(csv_path)
            print(f"✅ CSV verisi yüklendi: {len(csv_data)} satır")
            
            # Kategorik sütunları encode et (model eğitimindeki gibi)
            df_encoded = csv_data.copy()
            cat_cols = df_encoded.select_dtypes(include='object').columns
            
            # Label sütununu çıkar (eğer varsa)
            if 'Label' in cat_cols:
                cat_cols = cat_cols.drop('Label')
            
            # NaN'leri "unknown" ile doldur ve label encode et
            from sklearn.preprocessing import LabelEncoder
            for col in cat_cols:
                df_encoded[col] = df_encoded[col].fillna("unknown")
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
            
            csv_data = df_encoded
            print("✅ Veri ön işleme tamamlandı")
        else:
            print(f"❌ CSV dosyası bulunamadı: {csv_path}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Model/veri yükleme hatası: {e}")
        return False

# CSV'den satır okuma ve tahmin yapma
def get_next_prediction():
    global current_row_index, csv_data
    
    if csv_data is None or len(csv_data) == 0:
        return None, None, None
    
    # Döngüsel olarak veriyi oku
    current_row_index = current_row_index % len(csv_data)
    row = csv_data.iloc[current_row_index]
    
    # Label varsa çıkar
    true_label = None
    if 'Label' in row:
        true_label = row['Label']
        row = row.drop('Label')
    
    try:
        # Özellikleri hazırla
        if top_features:
            # Sadece top 10 feature kullan
            features = row[top_features].values.reshape(1, -1)
        else:
            # Tüm özellikleri kullan
            features = row.values.reshape(1, -1)
        
        # Scaler varsa uygula
        if scaler:
            features = scaler.transform(features)
        
        # Tahmin yap
        prediction = model.predict(features)[0]
        
        # Olasılık tahmini (eğer destekleniyorsa)
        try:
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities))
        except:
            confidence = 0.85 if prediction == 1 else 0.75
        
        current_row_index += 1
        
        return int(prediction), float(confidence), true_label
        
    except Exception as e:
        print(f"❌ Tahmin hatası: {e}")
        current_row_index += 1
        return None, None, None

# Arka plan izleme thread'i
def background_monitoring():
    """Arka planda sürekli veri analizi yap"""
    global is_security_active, activity_log
    
    while True:
        if is_security_active and model is not None:
            # Model tahmini yap
            prediction, confidence, true_label = get_next_prediction()
            
            if prediction is not None:
                # Ağ verisi simülasyonu (görselleştirme için)
                network_data = {
                    'source_ip': f"192.168.1.{random.randint(1, 255)}",
                    'dest_ip': f"10.0.0.{random.randint(1, 255)}",
                    'source_port': random.choice([80, 443, 8080, 3000, 5000]),
                    'dest_port': random.choice([80, 443, 8080, 3000, 5000]),
                    'protocol': random.choice(['TCP', 'UDP']),
                    'packet_size': random.randint(100, 1500),
                    'row_index': current_row_index - 1
                }
                
                # Log kaydı oluştur
                is_attack = prediction == 1
                log_entry = {
                    'id': int(time.time() * 1000),
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'type': 'attack' if is_attack else 'normal',
                    'confidence': confidence,
                    'prediction': prediction,
                    'true_label': true_label,
                    'details': f"{'MITM Saldırısı Tespit Edildi!' if is_attack else 'Normal Trafik'} - Güven: %{confidence*100:.1f}",
                    'network_data': network_data
                }
                
                activity_log.append(log_entry)
                
                # Son 100 kaydı tut
                if len(activity_log) > 100:
                    activity_log = activity_log[-100:]
                
                # Saldırı varsa konsola yazdır
                if is_attack:
                    print(f"⚠️  MITM SALDIRISI TESPİT EDİLDİ! Güven: %{confidence*100:.1f} | Satır: {current_row_index-1}")
                else:
                    print(f"✅ Normal trafik | Güven: %{confidence*100:.1f} | Satır: {current_row_index-1}")
        
        # 10 saniye bekle
        time.sleep(10)

# API Endpoints
@app.route('/api/security/status', methods=['GET'])
def get_security_status():
    """Güvenlik durumunu getir"""
    attack_count = len([log for log in activity_log if log.get('prediction') == 1])
    normal_count = len([log for log in activity_log if log.get('prediction') == 0])
    
    return jsonify({
        'is_active': is_security_active,
        'model_loaded': model is not None,
        'total_analyzed': len(activity_log),
        'attacks_detected': attack_count,
        'normal_traffic': normal_count,
        'current_row': current_row_index,
        'total_rows': len(csv_data) if csv_data is not None else 0
    })

@app.route('/api/security/toggle', methods=['POST'])
def toggle_security():
    """Güvenliği aç/kapat"""
    global is_security_active
    
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model yüklenmemiş! Lütfen modeli kontrol edin.'
        }), 400
    
    is_security_active = not is_security_active
    
    # Sistem log'u ekle
    log_entry = {
        'id': int(time.time() * 1000),
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'type': 'system',
        'details': f"Güvenlik sistemi {'aktif edildi' if is_security_active else 'devre dışı bırakıldı'}"
    }
    activity_log.append(log_entry)
    
    return jsonify({
        'success': True,
        'is_active': is_security_active,
        'message': f"Güvenlik {'aktif' if is_security_active else 'pasif'}"
    })

@app.route('/api/activity/logs', methods=['GET'])
def get_activity_logs():
    """Aktivite loglarını getir"""
    # Son 50 kaydı ters sırada döndür
    return jsonify({
        'logs': activity_log[-50:][::-1]
    })

@app.route('/api/activity/stats', methods=['GET'])
def get_activity_stats():
    """İstatistikleri getir"""
    if not activity_log:
        return jsonify({
            'total': 0,
            'attacks': 0,
            'normal': 0,
            'accuracy': 0
        })
    
    # Sadece tahmin içeren logları al
    prediction_logs = [log for log in activity_log if 'prediction' in log]
    
    if not prediction_logs:
        return jsonify({
            'total': 0,
            'attacks': 0,
            'normal': 0,
            'accuracy': 0
        })
    
    attacks = sum(1 for log in prediction_logs if log['prediction'] == 1)
    normal = sum(1 for log in prediction_logs if log['prediction'] == 0)
    
    # Eğer gerçek etiketler varsa doğruluk hesapla
    correct = 0
    total_with_labels = 0
    for log in prediction_logs:
        if log.get('true_label') is not None:
            total_with_labels += 1
            if log['prediction'] == log['true_label']:
                correct += 1
    
    accuracy = (correct / total_with_labels * 100) if total_with_labels > 0 else 0
    
    return jsonify({
        'total': len(prediction_logs),
        'attacks': attacks,
        'normal': normal,
        'accuracy': round(accuracy, 2)
    })

@app.route('/api/activity/clear', methods=['POST'])
def clear_activity_logs():
    """Logları temizle"""
    global activity_log, current_row_index
    activity_log = []
    current_row_index = 0
    return jsonify({'success': True, 'message': 'Loglar temizlendi'})

@app.route('/api/system/info', methods=['GET'])
def get_system_info():
    """Sistem bilgilerini getir"""
    return jsonify({
        'app_name': 'YAKUT',
        'version': '1.0.0',
        'model_status': 'loaded' if model else 'not_loaded',
        'model_type': type(model).__name__ if model else 'N/A',
        'features_count': len(top_features) if top_features else 'all',
        'csv_loaded': csv_data is not None,
        'csv_rows': len(csv_data) if csv_data is not None else 0
    })

if __name__ == '__main__':
    print("🚀 YAKUT Backend başlatılıyor...")
    
    # Model ve veri yükle
    if load_model_and_data():
        print("✅ Sistem hazır!")
        
        # Arka plan izleme thread'ini başlat
        monitoring_thread = threading.Thread(target=background_monitoring, daemon=True)
        monitoring_thread.start()
        print("🔍 MITM tespit sistemi aktif")
    else:
        print("❌ Model veya veri yüklenemedi! Lütfen dosya yollarını kontrol edin.")
        print("📁 Beklenen dosyalar:")
        print("   - Models/yakut_xgboost_model.joblib")
        print("   - Models/yakut_scaler.joblib (opsiyonel)")
        print("   - Models/top_10_features.json (opsiyonel)")
        print("   - Data/All_Labelled.csv")
    
    # Flask uygulamasını başlat
    app.run(debug=True, host='0.0.0.0', port=5000)