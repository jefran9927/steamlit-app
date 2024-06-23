import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model import recommend_vehicle_knn, recommend_vehicle_rf, predict_price_category_knn, predict_price_category_rf

# Load and prepare data
df_cleaned = pd.read_csv('data/cleaned_data.csv')

st.set_page_config(page_title="Rekomendasi Kendaraan Listrik", layout="wide")

# Sidebar for input
st.sidebar.header("Input Parameters")
msrp = st.sidebar.number_input("MSRP", min_value=0.0, format="%.2f")
model_year = st.sidebar.number_input("Model Year", min_value=1900, format="%d")
electric_range = st.sidebar.number_input("Electric Range", min_value=0, format="%d")

tab1, tab2 = st.tabs(["Rekomendasi", "Prediksi Harga"])

with tab1:
    if st.sidebar.button("Get Recommendations"):
        recommendations_knn = recommend_vehicle_knn(msrp, model_year, electric_range)
        recommendations_rf = recommend_vehicle_rf(msrp, model_year, electric_range)
        predicted_category_knn = predict_price_category_knn(msrp, model_year, electric_range)
        predicted_category_rf = predict_price_category_rf(msrp, model_year, electric_range)

        # Tabs for displaying recommendations and visualizations
        rec_tab1, rec_tab2, rec_tab3 = st.tabs(["Rekomendasi KNN", "Rekomendasi Random Forest", "Visualisasi"])

        with rec_tab1:
            st.subheader("Rekomendasi dengan KNN")
            st.dataframe(recommendations_knn)

        with rec_tab2:
            st.subheader("Rekomendasi dengan Random Forest")
            st.dataframe(recommendations_rf)

        with rec_tab3:
            st.subheader("Visualisasi")

            # Visualization 1: Base MSRP Distribution
            fig1 = px.histogram(recommendations_knn, x="Base MSRP", title="Distribusi Base MSRP (KNN)")
            st.plotly_chart(fig1, use_container_width=True)

            # Visualization 2: Electric Range vs Model Year
            fig2 = px.scatter(recommendations_rf, x="Model Year", y="Electric Range", size="Base MSRP", color="Price Category",
                              title="Jangkauan Listrik vs Tahun Model (Random Forest)")
            st.plotly_chart(fig2, use_container_width=True)

        st.sidebar.subheader(f"Prediksi Kategori Harga dengan KNN: {predicted_category_knn}")
        st.sidebar.subheader(f"Prediksi Kategori Harga dengan Random Forest: {predicted_category_rf}")

    else:
        st.title("Rekomendasi Kendaraan Listrik")
        st.write("Masukkan parameter di sidebar untuk mendapatkan rekomendasi kendaraan listrik.")

with tab2:
    st.title("Prediksi Kategori Harga")

    model_choice = st.selectbox("Pilih Model untuk Prediksi", ["KNN", "Random Forest"])

    if model_choice == "KNN":
        predicted_category = predict_price_category_knn(msrp, model_year, electric_range)
    else:
        predicted_category = predict_price_category_rf(msrp, model_year, electric_range)

    st.subheader(f"Prediksi Kategori Harga: {predicted_category}")

    # Additional Visualization 3: Box Plot of Electric Range
    st.subheader("Visualisasi Tambahan")
    fig3 = px.box(df_cleaned, x="Price Category", y="Electric Range", title="Distribusi Jangkauan Listrik berdasarkan Kategori Harga")
    st.plotly_chart(fig3, use_container_width=True)
