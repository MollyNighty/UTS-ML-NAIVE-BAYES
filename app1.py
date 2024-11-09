import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Fungsi untuk memuat dataset
def load_data():
    fish_data = pd.read_csv('dataset/fish.csv')
    fruit_data = pd.read_excel('dataset/fruit.xlsx')
    return fish_data, fruit_data

# Fungsi untuk melatih model Naive Bayes
def train_model(data, target_column):
    # Memisahkan fitur dan target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Membagi data untuk training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Membuat dan melatih model
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Evaluasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

# Fungsi untuk menambahkan data baru dan mengklasifikasinya
def classify_new_data(model, input_data):
    prediction = model.predict([input_data])
    return prediction[0]

# Membuat aplikasi Streamlit
def main():
    st.title("Aplikasi Klasifikasi Naive Bayes")
    
    # Muat data
    fish_data, fruit_data = load_data()
    
    # Pilih dataset untuk digunakan
    dataset_option = st.selectbox("Pilih Dataset", ("Fish", "Fruit"))
    
    if dataset_option == "Fish":
        data = fish_data
        target_column = 'species'  # Sesuaikan dengan nama kolom target di dataset Fish
    else:
        data = fruit_data
        target_column = 'name'  # Sesuaikan dengan nama kolom target di dataset Fruit

    # Melatih model
    model, accuracy = train_model(data, target_column)
    st.write(f"Akurasi model: {accuracy*100:.2f}%")
    
    # Form untuk menambahkan data baru
    st.subheader("Tambah Data Baru untuk Klasifikasi")
    input_data = []
    for column in data.columns:
        if column != target_column:
            value = st.number_input(f"Masukkan nilai untuk {column}", min_value=0.0)
            input_data.append(value)
    
    if st.button("Klasifikasi"):
        # Klasifikasi data baru
        result = classify_new_data(model, input_data)
        st.write(f"Hasil klasifikasi: {result}")

if __name__ == "__main__":
    main()
