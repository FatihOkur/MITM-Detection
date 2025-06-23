import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import joblib

df = pd.read_csv('Data/All_Labelled.csv')
print(df)

df_encoded = df.copy()
cat_cols = df_encoded.select_dtypes(include='object').columns

# NaN'leri "unknown" ile doldur ve label encode et
for col in cat_cols:
    df_encoded[col] = df_encoded[col].fillna("unknown")
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Eksik verileri kontrol et
missing_after_fill = df_encoded.isnull().sum()

# Sıfırdan büyük olanları yazdır (hala eksik olanlar)
missing_columns = missing_after_fill[missing_after_fill > 0]

if missing_columns.empty:
    print("✅ Tüm eksik veriler başarıyla doldurulmuş. Veri setinde eksik değer kalmadı.")
else:
    print("⚠️ Hala eksik değer içeren sütunlar:")
    print(missing_columns)

# 1. Kategorik sütunlar zaten encode edilmiş olmalı (önceki adımda yapıldı)
# 2. Özellik ve etiket ayrımı
X = df_encoded.drop(columns=['Label'])
y = df_encoded['Label']

# 3. Eğitim ve test verisine ayır (SMOTE sadece eğitim verisine uygulanır!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Ölçekleme (LSTM vs için de kullanılır)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE uygulaması
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Sonuç kontrolü
print("Orijinal eğitim veri dağılımı:\n", y_train.value_counts())
print("SMOTE sonrası eğitim veri dağılımı:\n", pd.Series(y_train_smote).value_counts())

# SMOTELU VERSION
# Model 1: Decision Tree
dt_params = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_distributions=dt_params,
    n_iter=10,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1
)
dt_search.fit(X_train_smote, y_train_smote)
dt_best = dt_search.best_estimator_
dt_pred = dt_best.predict(X_test_scaled)


# Model 2: Random Forest
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=rf_params,
    n_iter=10,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1
)
rf_search.fit(X_train_smote, y_train_smote)
rf_best = rf_search.best_estimator_
rf_pred = rf_best.predict(X_test_scaled)

# Model 3: XGBoost
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_distributions=xgb_params,
    n_iter=10,
    scoring='f1',
    cv=3,
    verbose=1,
    n_jobs=-1
)
xgb_search.fit(X_train_smote, y_train_smote)
xgb_best = xgb_search.best_estimator_
xgb_pred = xgb_best.predict(X_test_scaled)


# Model 4: LSTM
# LSTM için en fazla kullanılabilecek feature sayısı
# reshape input
max_features = min(100, X_train_scaled.shape[1])
X_lstm_train = X_train_smote[:, :max_features].reshape((-1, 1, max_features))
X_lstm_test = X_test_scaled[:, :max_features].reshape((-1, 1, max_features))
# ✅ SORUN ÇÖZÜMÜ: pandas Series'i numpy array'e çevir
y_lstm_train = np.array(y_train_smote)
y_lstm_test = np.array(y_test)

def manual_hyperparameter_tuning():
    print("\n=== Manuel Hyperparameter Tuning ===")
    
    # Parametre kombinasyonları
    param_combinations = [
        {'units': 32, 'dropout': 0.2, 'lr': 0.001, 'epochs': 5, 'batch_size': 64},
        {'units': 64, 'dropout': 0.3, 'lr': 0.001, 'epochs': 5, 'batch_size': 64},
        {'units': 128, 'dropout': 0.5, 'lr': 0.0005, 'epochs': 5, 'batch_size': 128},
        {'units': 64, 'dropout': 0.2, 'lr': 0.001, 'epochs': 10, 'batch_size': 128},
        {'units': 32, 'dropout': 0.3, 'lr': 0.0005, 'epochs': 5, 'batch_size': 64}
    ]
    
    best_score = 0
    best_params = None
    best_model = None
    
    # 2-fold cross validation
    kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    
    for i, params in enumerate(param_combinations):
        print(f"\nTest ediliyor {i+1}/{len(param_combinations)}: {params}")
        
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_lstm_train, y_lstm_train)):
            X_cv_train, X_cv_val = X_lstm_train[train_idx], X_lstm_train[val_idx]
            y_cv_train, y_cv_val = y_lstm_train[train_idx], y_lstm_train[val_idx]
            
            # Model oluştur
            model = Sequential([
                LSTM(params['units'], input_shape=(1, max_features)),
                Dropout(params['dropout']),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=params['lr']),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Eğit
            model.fit(
                X_cv_train, y_cv_train,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                verbose=0,
                validation_split=0.1
            )
            
            # Değerlendir
            y_pred = (model.predict(X_cv_val, verbose=0) > 0.5).astype(int).flatten()
            score = f1_score(y_cv_val, y_pred)
            cv_scores.append(score)
        
        # Ortalama skor
        avg_score = np.mean(cv_scores)
        print(f"  CV F1 Score: {avg_score:.4f} (±{np.std(cv_scores):.4f})")
        
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            
            # En iyi modeli tekrar eğit
            best_model = Sequential([
                LSTM(params['units'], input_shape=(1, max_features)),
                Dropout(params['dropout']),
                Dense(1, activation='sigmoid')
            ])
            
            best_model.compile(
                optimizer=Adam(learning_rate=params['lr']),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            best_model.fit(
                X_lstm_train, y_lstm_train,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                verbose=0,
                validation_split=0.1
            )
    
    print(f"\n🏆 En iyi parametreler: {best_params}")
    print(f"🏆 En iyi CV F1 Score: {best_score:.4f}")
    
    # Test seti ile tahmin
    final_predictions = best_model.predict(X_lstm_test, verbose=0)
    lstm_pred = (final_predictions > 0.5).astype(int).flatten()
    
    test_f1 = f1_score(y_lstm_test, lstm_pred)
    print(f"🎯 Test F1 Score: {test_f1:.4f}")
    
    return lstm_pred, best_model, best_params

# Fonksiyonu çalıştır
lstm_pred, lstm_best_model, lstm_best_params = manual_hyperparameter_tuning()

# Değerlendirme fonksiyonu
def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC AUC": roc_auc_score(y_true, y_pred)
    }

# Sonuçları topla
results = [
    evaluate_model("Decision Tree SMOTED", y_test, dt_pred),
    evaluate_model("Random Forest SMOTED", y_test, rf_pred),
    evaluate_model("XGBoost SMOTED", y_test, xgb_pred),
    evaluate_model("LSTM SMOTED", y_test, lstm_pred)
]

# Tabloya dök
results_df = pd.DataFrame(results)
print(results_df)

# Orijinal X sütunlarının isimlerini al (ölçeklenmeden önceki)
feature_names = X_train.columns

# 3. XGBoost Feature Importance
xgb_importance = pd.Series(xgb_best.feature_importances_, index=feature_names).sort_values(ascending=False).head(20)
plt.figure(figsize=(8, 5))
sns.barplot(x=xgb_importance.values, y=xgb_importance.index)
plt.title("XGBoost - En Önemli 20 Özellik")
plt.xlabel("Önemi")
plt.tight_layout()
plt.show()

# Eğer feature isimlerin varsa, array'leri tekrar DataFrame'e çevir
X_train_smote_df = pd.DataFrame(X_train_smote, columns=feature_names)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

# En önemli 10 özelliği al
top_10_features = xgb_importance.head(10).index.tolist()

# Eğitim ve test verilerini sadece bu 10 özellik ile filtrele
X_train_top10 = X_train_smote_df[top_10_features]
X_test_top10 = X_test_scaled_df[top_10_features]

# Yeni XGBoost modeli ile eğit
xgb_top10 = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_top10.fit(X_train_top10, y_train_smote)

# Tahmin yap
xgb_top10_pred = xgb_top10.predict(X_test_top10)

result_xgb_top10 = evaluate_model("XGBoost SMOTED with Top 10 features", y_test, xgb_top10_pred)
result_xgb_top10_df = pd.DataFrame([result_xgb_top10])
print(result_xgb_top10_df.to_string(index=False))

# Örnek: model bir RandomForest olabilir
joblib.dump(xgb_top10, "Models/yakut_xgboost_model.joblib")

# Model.py'nin en sonuna ekleyin (joblib.dump satırından hemen sonra):

# Scaler'ı kaydet
joblib.dump(scaler, "Models/yakut_scaler.joblib")

# Top 10 features'ı JSON olarak kaydet
import json
with open('Models/top_10_features.json', 'w') as f:
    json.dump(top_10_features, f, indent=2)

print("\n✅ Tüm dosyalar kaydedildi:")
print("   - Models/yakut_xgboost_model.joblib")
print("   - Models/yakut_scaler.joblib") 
print("   - Models/top_10_features.json")