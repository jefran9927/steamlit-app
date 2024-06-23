import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load and prepare data
df_cleaned = pd.read_csv('data/cleaned_data.csv')

# Persiapan Data dengan fitur tambahan
features = ['Base MSRP', 'Model Year', 'Electric Range', 'Model', 'Make', 'Electric Vehicle Type']
df_cleaned = df_cleaned[features]

# Menghapus baris dengan nilai NaN
df_cleaned = df_cleaned.dropna()

# Menambahkan kolom Price Category berdasarkan Base MSRP
df_cleaned['Price Category'] = pd.cut(df_cleaned['Base MSRP'],
                                      bins=[20000, 50000, 845000, np.inf],
                                      labels=['Murah', 'Normal', 'Mahal'])

X = df_cleaned[['Base MSRP', 'Model Year', 'Electric Range']].values
y = df_cleaned['Base MSRP'].values  # Mengambil kolom 'Base MSRP' sebagai target

# Standardisasi Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Membangun Model KNN
knn = NearestNeighbors(n_neighbors=20, algorithm='auto')
knn.fit(X_scaled)

# Membangun Model Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

def recommend_vehicle_knn(msrp, model_year, electric_range, n_recommendations=5):
    input_data = np.array([[msrp, model_year, electric_range]])
    input_scaled = scaler.transform(input_data)
    distances, indices = knn.kneighbors(input_scaled, n_neighbors=n_recommendations * 2)
    
    recommendations = df_cleaned.iloc[indices[0]].copy()
    recommendations = recommendations.drop_duplicates(subset=['Model']).head(n_recommendations)
    return recommendations

def recommend_vehicle_rf(msrp, model_year, electric_range, n_recommendations=5):
    input_data = np.array([[msrp, model_year, electric_range]])
    input_scaled = scaler.transform(input_data)
    preds = rf.predict(input_scaled.reshape(1, -1))
    
    df_cleaned['Pred_Diff'] = np.abs(df_cleaned['Base MSRP'] - preds[0])
    recommendations = df_cleaned.sort_values(by='Pred_Diff').head(n_recommendations * 2).copy()
    recommendations = recommendations.drop_duplicates(subset=['Model']).head(n_recommendations)
    recommendations.drop(columns=['Pred_Diff'], inplace=True)
    return recommendations

def predict_price_category_knn(msrp, model_year, electric_range):
    input_data = np.array([[msrp, model_year, electric_range]])
    input_scaled = scaler.transform(input_data)
    distances, indices = knn.kneighbors(input_scaled, n_neighbors=5)
    
    recommended_categories = df_cleaned.iloc[indices[0]]['Price Category']
    predicted_category = recommended_categories.mode()[0]  # Menggunakan mode untuk menentukan kategori harga yang paling sering muncul
    return predicted_category

def predict_price_category_rf(msrp, model_year, electric_range):
    input_data = np.array([[msrp, model_year, electric_range]])
    input_scaled = scaler.transform(input_data)
    pred_msrp = rf.predict(input_scaled.reshape(1, -1))
    
    if pred_msrp < df_cleaned[df_cleaned['Price Category'] == 'Murah']['Base MSRP'].max():
        return 'Murah'
    elif pred_msrp < df_cleaned[df_cleaned['Price Category'] == 'Normal']['Base MSRP'].max():
        return 'Normal'
    else:
        return 'Mahal'
