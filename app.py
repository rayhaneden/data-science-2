import streamlit as st
import pandas as pd
import joblib
from data_preprocessing import data_preprocessing
from prediction import prediction

import streamlit as st
import pandas as pd

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