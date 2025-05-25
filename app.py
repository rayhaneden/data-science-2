import streamlit as st
import pandas as pd
import joblib
import numpy as np
import streamlit as st
import pandas as pd

# Load setiap scaler secara manual
pca_1 = joblib.load("model/pca_1.joblib")
scaler_Application_mode = joblib.load("model/scaler_Application_mode.joblib")
scaler_Previous_qualification_grade = joblib.load("model/scaler_Previous_qualification_grade.joblib")
scaler_Admission_grade = joblib.load("model/scaler_Admission_grade.joblib")
scaler_Displaced = joblib.load("model/scaler_Displaced.joblib")
scaler_Debtor = joblib.load("model/scaler_Debtor.joblib")
scaler_Tuition_fees_up_to_date = joblib.load("model/scaler_Tuition_fees_up_to_date.joblib")
scaler_Gender = joblib.load("model/scaler_Gender.joblib")
scaler_Scholarship_holder = joblib.load("model/scaler_Scholarship_holder.joblib")
scaler_Age_at_enrollment = joblib.load("model/scaler_Age_at_enrollment.joblib")
scaler_Curricular_units_1st_sem_enrolled = joblib.load("model/scaler_Curricular_units_1st_sem_enrolled.joblib")
scaler_Curricular_units_1st_sem_evaluations = joblib.load("model/scaler_Curricular_units_1st_sem_evaluations.joblib")
scaler_Curricular_units_1st_sem_approved = joblib.load("model/scaler_Curricular_units_1st_sem_approved.joblib")
scaler_Curricular_units_1st_sem_grade = joblib.load("model/scaler_Curricular_units_1st_sem_grade.joblib")
scaler_Curricular_units_2nd_sem_enrolled = joblib.load("model/scaler_Curricular_units_2nd_sem_enrolled.joblib")
scaler_Curricular_units_2nd_sem_evaluations = joblib.load("model/scaler_Curricular_units_2nd_sem_evaluations.joblib")
scaler_Curricular_units_2nd_sem_approved = joblib.load("model/scaler_Curricular_units_2nd_sem_approved.joblib")
scaler_Curricular_units_2nd_sem_grade = joblib.load("model/scaler_Curricular_units_2nd_sem_grade.joblib")



# Daftar kolom untuk PCA
pca_numerical_columns = [
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade'
]

def data_preprocessing(data):
    """Preprocessing data

    Args:
        data (Pandas DataFrame): Dataframe yang berisi semua data untuk prediksi

    Returns:
        Pandas DataFrame: Dataframe yang berisi semua data yang sudah dipreprocess
    """
    data = data.copy()
    df = pd.DataFrame()
    # Standarisasi kolom selain PCA (manual)
    df["Application_mode"] = scaler_Application_mode.transform(np.asarray(data["Application_mode"]).reshape(-1,1))[0]
    df["Previous_qualification_grade"] = scaler_Previous_qualification_grade.transform(np.asarray(data["Previous_qualification_grade"]).reshape(-1,1))[0]
    df["Admission_grade"] = scaler_Admission_grade.transform(np.asarray(data["Admission_grade"]).reshape(-1,1))[0]
    df["Displaced"] = scaler_Displaced.transform(np.asarray(data["Displaced"]).reshape(-1,1))[0]
    df["Debtor"] = scaler_Debtor.transform(np.asarray(data["Debtor"]).reshape(-1,1))[0]
    df["Tuition_fees_up_to_date"] = scaler_Tuition_fees_up_to_date.transform(np.asarray(data["Tuition_fees_up_to_date"]).reshape(-1,1))[0]
    df["Gender"] = scaler_Gender.transform(np.asarray(data["Gender"]).reshape(-1,1))[0]
    df["Scholarship_holder"] = scaler_Scholarship_holder.transform(np.asarray(data["Scholarship_holder"]).reshape(-1,1))[0]
    df["Age_at_enrollment"] = scaler_Age_at_enrollment.transform(np.asarray(data["Age_at_enrollment"]).reshape(-1,1))[0]

    # Standarisasi kolom PCA (untuk diproses PCA)
    data["Curricular_units_1st_sem_enrolled"] = scaler_Curricular_units_1st_sem_enrolled.transform(np.asarray(data["Curricular_units_1st_sem_enrolled"]).reshape(-1,1))[0]
    data["Curricular_units_1st_sem_evaluations"] = scaler_Curricular_units_1st_sem_evaluations.transform(np.asarray(data["Curricular_units_1st_sem_evaluations"]).reshape(-1,1))[0]
    data["Curricular_units_1st_sem_approved"] = scaler_Curricular_units_1st_sem_approved.transform(np.asarray(data["Curricular_units_1st_sem_approved"]).reshape(-1,1))[0]
    data["Curricular_units_1st_sem_grade"] = scaler_Curricular_units_1st_sem_grade.transform(np.asarray(data["Curricular_units_1st_sem_grade"]).reshape(-1,1))[0]
    data["Curricular_units_2nd_sem_enrolled"] = scaler_Curricular_units_2nd_sem_enrolled.transform(np.asarray(data["Curricular_units_2nd_sem_enrolled"]).reshape(-1,1))[0]
    data["Curricular_units_2nd_sem_evaluations"] = scaler_Curricular_units_2nd_sem_evaluations.transform(np.asarray(data["Curricular_units_2nd_sem_evaluations"]).reshape(-1,1))[0]
    data["Curricular_units_2nd_sem_approved"] = scaler_Curricular_units_2nd_sem_approved.transform(np.asarray(data["Curricular_units_2nd_sem_approved"]).reshape(-1,1))[0]
    data["Curricular_units_2nd_sem_grade"] = scaler_Curricular_units_2nd_sem_grade.transform(np.asarray(data["Curricular_units_2nd_sem_grade"]).reshape(-1,1))[0]

    # PCA (5 komponen)
    df[["pc1_1", "pc1_2", "pc1_3", "pc1_4", "pc1_5"]] = pca_1.transform(data[pca_numerical_columns].values.reshape(1, -1))

    return df

model = joblib.load("model/best_rfl.joblib")
result_target = joblib.load("model/encoder_target.joblib")
def prediction(data):
    """Making prediction
 
    Args:
        data (Pandas DataFrame): Dataframe that contain all the preprocessed data
 
    Returns:
        str: Prediction result (Good, Standard, or Poor)
    """
    result = model.predict(data)
    final_result = result_target.inverse_transform(result)[0]
    return final_result


application_mode_mapping = {
    1: "1st phase - general contingent",
    2: "Ordinance No. 612/93",
    5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses",
    10: "Ordinance No. 854-B/99",
    15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)",
    17: "2nd phase - general contingent",
    18: "3rd phase - general contingent",
    26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)",
    39: "Over 23 years old",
    42: "Transfer",
    43: "Change of course",
    44: "Technological specialization diploma holders",
    51: "Change of institution/course",
    53: "Short cycle diploma holders",
    57: "Change of institution/course (International)"
}


# Inisialisasi DataFrame kosong
data = pd.DataFrame()

# Daftar kolom dan deskripsi (berdasarkan paste.txt dan tambahan umum)
kolom_dan_deskripsi = {
    'Application_mode': 'Application mode (method of application used by the student)',
    'Previous_qualification_grade': 'Grade of previous qualification (0-200)',
    'Admission_grade': 'Admission grade (0-200)',
    'Displaced': 'Displaced person (1=yes, 0=no)',
    'Debtor': 'Debtor (1=yes, 0=no)',
    'Tuition_fees_up_to_date': 'Tuition fees up to date (1=yes, 0=no)',
    'Gender': 'Gender (1=male, 0=female)',
    'Scholarship_holder': 'Scholarship holder (1=yes, 0=no)',
    'Age_at_enrollment': 'Age at enrollment',
    'Curricular_units_1st_sem_enrolled': 'Curricular units 1st sem (enrolled)',
    'Curricular_units_1st_sem_evaluations': 'Curricular units 1st sem (evaluations)',
    'Curricular_units_1st_sem_approved': 'Curricular units 1st sem (approved)',
    'Curricular_units_1st_sem_grade': 'Curricular units 1st sem (grade)',
    'Curricular_units_2nd_sem_enrolled': 'Curricular units 2nd sem (enrolled)',
    'Curricular_units_2nd_sem_evaluations': 'Curricular units 2nd sem (evaluations)',
    'Curricular_units_2nd_sem_approved': 'Curricular units 2nd sem (approved)',
    'Curricular_units_2nd_sem_grade': 'Curricular units 2nd sem (grade)'
}

# Kolom 1: Application_mode, Previous_qualification_grade, Admission_grade
col1, col2, col3 = st.columns(3)
with col1:
    Application_mode = st.selectbox(
        label="Application mode",
        options=list(application_mode_mapping.keys()),
        format_func=lambda x: application_mode_mapping[x],
        index=0  # default pilihan pertama
    )
    data["Application_mode"] = [Application_mode]
with col2:
    Previous_qualification_grade = st.number_input(
        label=kolom_dan_deskripsi['Previous_qualification_grade'],
        value=100,
        min_value=0,
        max_value=200
    )
    data["Previous_qualification_grade"] = [Previous_qualification_grade]
with col3:
    Admission_grade = st.number_input(
        label=kolom_dan_deskripsi['Admission_grade'],
        value=100,
        min_value=0,
        max_value=200
    )
    data["Admission_grade"] = [Admission_grade]

# Kolom 2: Displaced, Debtor, Tuition_fees_up_to_date, Gender
col1, col2, col3, col4 = st.columns(4)
with col1:
    Displaced = st.selectbox(
        label=kolom_dan_deskripsi['Displaced'],
        options=[1, 0],
        format_func=lambda x: 'Yes' if x == 1 else 'No'
    )
    data["Displaced"] = [Displaced]
with col2:
    Debtor = st.selectbox(
        label=kolom_dan_deskripsi['Debtor'],
        options=[1, 0],
        format_func=lambda x: 'Yes' if x == 1 else 'No'
    )
    data["Debtor"] = [Debtor]
with col3:
    Tuition_fees_up_to_date = st.selectbox(
        label=kolom_dan_deskripsi['Tuition_fees_up_to_date'],
        options=[1, 0],
        format_func=lambda x: 'Yes' if x == 1 else 'No'
    )
    data["Tuition_fees_up_to_date"] = [Tuition_fees_up_to_date]
with col4:
    Gender = st.selectbox(
        label=kolom_dan_deskripsi['Gender'],
        options=[1, 0],
        format_func=lambda x: 'Male' if x == 1 else 'Female'
    )
    data["Gender"] = [Gender]

# Kolom 3: Scholarship_holder, Age_at_enrollment
col1, col2 = st.columns(2)
with col1:
    Scholarship_holder = st.selectbox(
        label=kolom_dan_deskripsi['Scholarship_holder'],
        options=[1, 0],
        format_func=lambda x: 'Yes' if x == 1 else 'No'
    )
    data["Scholarship_holder"] = [Scholarship_holder]
with col2:
    Age_at_enrollment = st.number_input(
        label=kolom_dan_deskripsi['Age_at_enrollment'],
        value=20,
        min_value=15,
        max_value=70
    )
    data["Age_at_enrollment"] = [Age_at_enrollment]

# Kolom 4: Curricular_units_1st_sem_enrolled, Curricular_units_1st_sem_evaluations, Curricular_units_1st_sem_approved, Curricular_units_1st_sem_grade
col1, col2, col3, col4 = st.columns(4)
with col1:
    Curricular_units_1st_sem_enrolled = st.number_input(
        label=kolom_dan_deskripsi['Curricular_units_1st_sem_enrolled'],
        value=5,
        min_value=0
    )
    data["Curricular_units_1st_sem_enrolled"] = [Curricular_units_1st_sem_enrolled]
with col2:
    Curricular_units_1st_sem_evaluations = st.number_input(
        label=kolom_dan_deskripsi['Curricular_units_1st_sem_evaluations'],
        value=5,
        min_value=0
    )
    data["Curricular_units_1st_sem_evaluations"] = [Curricular_units_1st_sem_evaluations]
with col3:
    Curricular_units_1st_sem_approved = st.number_input(
        label=kolom_dan_deskripsi['Curricular_units_1st_sem_approved'],
        value=4,
        min_value=0
    )
    data["Curricular_units_1st_sem_approved"] = [Curricular_units_1st_sem_approved]
with col4:
    Curricular_units_1st_sem_grade = st.number_input(
        label=kolom_dan_deskripsi['Curricular_units_1st_sem_grade'],
        value=12.5,
        min_value=0.0
    )
    data["Curricular_units_1st_sem_grade"] = [Curricular_units_1st_sem_grade]

# Kolom 5: Curricular_units_2nd_sem_enrolled, Curricular_units_2nd_sem_evaluations, Curricular_units_2nd_sem_approved, Curricular_units_2nd_sem_grade
col1, col2, col3, col4 = st.columns(4)
with col1:
    Curricular_units_2nd_sem_enrolled = st.number_input(
        label=kolom_dan_deskripsi['Curricular_units_2nd_sem_enrolled'],
        value=5,
        min_value=0
    )
    data["Curricular_units_2nd_sem_enrolled"] = [Curricular_units_2nd_sem_enrolled]
with col2:
    Curricular_units_2nd_sem_evaluations = st.number_input(
        label=kolom_dan_deskripsi['Curricular_units_2nd_sem_evaluations'],
        value=5,
        min_value=0
    )
    data["Curricular_units_2nd_sem_evaluations"] = [Curricular_units_2nd_sem_evaluations]
with col3:
    Curricular_units_2nd_sem_approved = st.number_input(
        label=kolom_dan_deskripsi['Curricular_units_2nd_sem_approved'],
        value=4,
        min_value=0
    )
    data["Curricular_units_2nd_sem_approved"] = [Curricular_units_2nd_sem_approved]
with col4:
    Curricular_units_2nd_sem_grade = st.number_input(
        label=kolom_dan_deskripsi['Curricular_units_2nd_sem_grade'],
        value=12.5,
        min_value=0.0
    )
    data["Curricular_units_2nd_sem_grade"] = [Curricular_units_2nd_sem_grade]

# Tampilkan hasil input
st.write("Data yang diinput:")
st.dataframe(data)

if st.button('Predict'):
    new_data = data_preprocessing(data=data)
    with st.expander("View the Preprocessed Data"):
        st.dataframe(data=new_data, width=800, height=10)
    st.write("Status Prediction: {}".format(prediction(new_data)))