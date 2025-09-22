# train_pipeline.py
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# EXACT order of the 43 feature columns used in training CSV
FEATURE_COLS = [
    'mean_spectral_centroid','std_spectral_centroid',
    'mean_spectral_bandwidth','std_spectral_bandwidth',
    'mean_spectral_contrast','mean_spectral_flatness','mean_spectral_rolloff',
    'zero_crossing_rate','rms_energy',
    'mean_pitch','min_pitch','max_pitch','std_pitch',
    'spectral_skew','spectral_kurtosis',
    'energy_entropy','log_energy',
    'mfcc_1_mean','mfcc_1_std','mfcc_2_mean','mfcc_2_std',
    'mfcc_3_mean','mfcc_3_std','mfcc_4_mean','mfcc_4_std',
    'mfcc_5_mean','mfcc_5_std','mfcc_6_mean','mfcc_6_std',
    'mfcc_7_mean','mfcc_7_std','mfcc_8_mean','mfcc_8_std',
    'mfcc_9_mean','mfcc_9_std','mfcc_10_mean','mfcc_10_std',
    'mfcc_11_mean','mfcc_11_std','mfcc_12_mean','mfcc_12_std',
    'mfcc_13_mean','mfcc_13_std'
]

# 1) Load CSV
df = pd.read_csv(r"C:\Users\jdpri\OneDrive\Documents\Human_Voice_Proj\Data\human_voice_no_duplicates.csv")
X = df[FEATURE_COLS].values        # RAW features (not scaled)
y = df["label"].values             # male=1, female=0

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3) Clean pipeline: RAW -> StandardScaler -> RandomForest
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_split=5,
        random_state=42, class_weight='balanced'
    ))
])

# 4) Fit on RAW training data
pipe.fit(X_train, y_train)

# 5) Evaluate on RAW test data (pipeline scales internally)
y_pred = pipe.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['Female (0)','Male (1)']))

# 6) Save ONE object (pipeline)
with open("rf_gender_pipeline.pkl", "wb") as f:
    pickle.dump(pipe, f)
