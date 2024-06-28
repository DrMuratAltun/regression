import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn

# Modeli yükleyin
model = joblib.load("model.pkl")

# Streamlit uygulaması
def main():
    st.title("Araba Fiyat Tahmin Uygulaması")

    # Kullanıcı girdileri
    mileage = st.number_input("Mileage", min_value=0)
    cylinder = st.number_input("Cylinder", min_value=0, step=1)
    liter = st.number_input("Liter", min_value=0.0, step=0.1)
    cruise = st.selectbox("Cruise (1: Var, 0: Yok)", [1, 0])
    
    make_options = ["Cadillac", "Chevrolet", "Pontiac", "SAAB", "Saturn"]
    make_selection = st.selectbox("Make", options=make_options)
    
    # Trim seçenekleri
    trim_options = [
        "Aero Conv 2D", "Aero Sedan 4D", "Aero Wagon 4D",
        "Arc Conv 2D", "Arc Sedan 4D", "Arc Wagon 4D",
        "CX Sedan 4D", "CXL Sedan 4D", "CXS Sedan 4D",
        "Conv 2D", "Coupe 2D", "Custom Sedan 4D",
        "DHS Sedan 4D", "DTS Sedan 4D", "GT Coupe 2D",
        "GT Sedan 4D", "GT Sportwagon", "GTP Sedan 4D",
        "GXP Sedan 4D", "Hardtop Conv 2D", "L300 Sedan 4D",
        "LS Coupe 2D", "LS Hatchback 4D", "LS MAXX Hback 4D",
        "LS Sedan 4D", "LS Sport Coupe 2D", "LS Sport Sedan 4D",
        "LT Coupe 2D", "LT Hatchback 4D", "LT MAXX Hback 4D",
        "LT Sedan 4D", "Limited Sedan 4D", "Linear Conv 2D",
        "Linear Sedan 4D", "Linear Wagon 4D", "MAXX Hback 4D",
        "Quad Coupe 2D", "SE Sedan 4D", "SLE Sedan 4D",
        "SS Coupe 2D", "SS Sedan 4D", "SVM Hatchback 4D",
        "SVM Sedan 4D", "Sedan 4D", "Special Ed Ultra 4D",
        "Sportwagon 4D"
    ]
    trim_selection = st.selectbox("Trim", options=trim_options)

    # Tahmin butonu
    if st.button("Fiyat Tahmini Yap"):
        # Özellikleri hazırla
        input_data = np.zeros(len(model.feature_names_in_))
        features_df = pd.DataFrame([input_data], columns=model.feature_names_in_)
        features_df['Mileage'] = mileage
        features_df['Cylinder'] = cylinder
        features_df['Liter'] = liter
        features_df['Cruise'] = cruise
        features_df[f"Make_{make_selection}"] = 1
        features_df[f"Trim_{trim_selection}"] = 1
        
        # Tahmini hesapla
        prediction = model.predict(features_df)
        
        # Tahmini göster
        st.success(f"Tahmini Araba Fiyatı: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    main()
