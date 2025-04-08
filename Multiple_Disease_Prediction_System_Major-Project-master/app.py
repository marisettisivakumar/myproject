import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Load models
working_dir = os.path.dirname(os.path.abspath(__file__))

diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'Tri Health Predictor',
        ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
        icons=['activity', 'heart', 'person'],
        menu_icon='hospital-fill',
        default_index=0
    )

# Function to create styled cards
def result_card(title, message, is_positive):
    color = "#FF4B4B" if is_positive else "#4BB543"
    st.markdown(
        f"""
        <div style="border-radius:10px;padding:15px;background-color:{color};color:white;text-align:center;font-size:18px;">
            <strong>{title}</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    st.subheader("Enter Patient Details:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.slider('Number of Pregnancies', 0, 20, 1)
        SkinThickness = st.slider('Skin Thickness (mm)', 0.0, 100.0, 20.0)
        DiabetesPedigreeFunction = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
    with col2:
        Glucose = st.slider('Glucose Level (mg/dL)', 0.0, 200.0, 100.0)
        Insulin = st.slider('Insulin Level (IU/mL)', 0.0, 500.0, 50.0)
        Age = st.slider('Age', 1, 100, 30)
    with col3:
        BloodPressure = st.slider('Blood Pressure (mmHg)', 0.0, 180.0, 80.0)
        BMI = st.slider('Body Mass Index', 10.0, 50.0, 25.0)

    if st.button('Predict Diabetes'):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        prediction = diabetes_model.predict([user_input])
        if prediction[0] == 1:
            result_card("Diabetes Detected", "Consult a doctor and follow a healthy lifestyle.", True)
        else:
            result_card("No Diabetes", "Maintain a balanced diet and regular exercise.", False)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    st.subheader("Enter Patient Details:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider('Age', 1, 100, 50)
        trestbps = st.slider('Resting Blood Pressure (mmHg)', 80, 200, 120)
        restecg = st.selectbox('Resting ECG Results', [0, 1, 2])
        oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0, 1.0)
    with col2:
        sex = st.radio('Sex', ['Male', 'Female'])
        chol = st.slider('Cholesterol (mg/dL)', 100, 400, 200)
        thalach = st.slider('Max Heart Rate Achieved', 60, 220, 150)
        slope = st.selectbox('Slope of Peak Exercise ST', [0, 1, 2])
    with col3:
        cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
        fbs = st.radio('Fasting Blood Sugar > 120 mg/dL', [0, 1])
        exang = st.radio('Exercise Induced Angina', [0, 1])
        ca = st.slider('Number of Major Vessels', 0, 4, 1)
    with col1:
        thal = st.selectbox('Thalassemia Type', [0, 1, 2, 3])  # Possible values: 0,1,2,3

    if st.button('Predict Heart Disease'):
        sex_val = 1 if sex == 'Male' else 0
        # user_input = [age, sex_val, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca]
        user_input = [age, sex_val, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        prediction = heart_disease_model.predict([user_input])
        if prediction[0] == 1:
            result_card("Heart Disease Detected", "Immediate consultation is recommended.", True)
        else:
            result_card("No Heart Disease", "Maintain a heart-healthy lifestyle.", False)

## Parkinson's Prediction Page
if selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction")
    st.subheader("Enter Patient Details:")
    
    cols = st.columns(5)
    input_features = {
        'MDVP:Fo(Hz)': (88, 260),
        'MDVP:Fhi(Hz)': (102, 592),
        'MDVP:Flo(Hz)': (65, 239),
        'MDVP:Jitter(%)': (0.0016, 0.033),
        'MDVP:Jitter(Abs)': (0.000007, 0.00026),
        'MDVP:RAP': (0.00068, 0.0214),
        'MDVP:PPQ': (0.00092, 0.0195),
        'Jitter:DDP': (0.00204, 0.0643),
        'MDVP:Shimmer': (0.00954, 0.119),
        'MDVP:Shimmer(dB)': (0.085, 1.3),
        'Shimmer:APQ3': (0.00455, 0.056),
        'Shimmer:APQ5': (0.0057, 0.079),
        'MDVP:APQ': (0.00719, 0.137),
        'Shimmer:DDA': (0.0136, 0.169),
        'NHR': (0.00065, 0.315),
        'HNR': (8.44, 33.04),
        'RPDE': (0.256, 0.685),
        'DFA': (0.574, 0.825),
        'spread1': (-7.96, -2.43),
        'spread2': (0.006, 0.450),
        'D2': (1.42, 3.67),
        'PPE': (0.044, 0.527)
    }

    user_inputs = []
    for i, (feature, (min_val, max_val)) in enumerate(input_features.items()):
        with cols[i % 5]:
            user_inputs.append(st.slider(feature, float(min_val), float(max_val), float((min_val + max_val) / 2)))
    
    if st.button("Predict Parkinson's Disease"):
        prediction = parkinsons_model.predict([user_inputs])
        if prediction[0] == 1:
            result_card("Parkinson's Detected", "Consult a neurologist for further guidance.", True)
        else:
            result_card("No Parkinson's Detected", "Maintain a healthy lifestyle.", False)

# Footer
st.markdown("""
**Developed By:**
- **M SIVA CHANDRA SAI KUMAR**
""")
