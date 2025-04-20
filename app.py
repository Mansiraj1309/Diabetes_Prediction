import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image

# Load the new dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Encode categorical columns
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0, 'Other': 2})
le = LabelEncoder()
df['smoking_history'] = le.fit_transform(df['smoking_history'])

# Grouped stats for display
mean_by_outcome = df.groupby('diabetes').mean()

# Split features and target
X = df.drop(columns='diabetes')
Y = df['diabetes']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=2)

# Model
classifier = svm.SVC(kernel='linear', class_weight='balanced')
classifier.fit(X_train, Y_train)

# Accuracy
train_acc = accuracy_score(classifier.predict(X_train), Y_train)
test_acc = accuracy_score(classifier.predict(X_test), Y_test)

# Streamlit App
def app():
    img = Image.open("img.jpeg")
    img = img.resize((200, 200))
    st.image(img, caption="Diabetes Image", width=200)

    st.title('Diabetes Prediction')

    st.sidebar.title('Input Features')
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
    age = st.sidebar.slider('Age', 0, 100, 30)
    hypertension = st.sidebar.selectbox('Hypertension', [0, 1])
    heart_disease = st.sidebar.selectbox('Heart Disease', [0, 1])
    smoking_history = st.sidebar.selectbox('Smoking History', list(le.classes_))
    bmi = st.sidebar.slider('BMI', 10.0, 70.0, 25.0)
    hba1c = st.sidebar.slider('HbA1c Level', 3.0, 10.0, 5.5)
    glucose = st.sidebar.slider('Blood Glucose Level', 50, 300, 100)

    # Encode inputs
    gender_val = {'Male': 1, 'Female': 0, 'Other': 2}[gender]
    smoking_val = le.transform([smoking_history])[0]

    input_data = np.array([[gender_val, age, hypertension, heart_disease, smoking_val, bmi, hba1c, glucose]])
    input_scaled = scaler.transform(input_data)

    prediction = classifier.predict(input_scaled)

    st.write('### Prediction Result:')
    if prediction[0] == 1:
        st.warning('This person **has** diabetes.')
    else:
        st.success('This person **does not** have diabetes.')

    st.header('Model Accuracy')
    st.write(f'Train accuracy: {train_acc:.2f}')
    st.write(f'Test accuracy: {test_acc:.2f}')

    st.header('Dataset Summary')
    st.write(df.describe())

    st.header('Mean Values by Outcome')
    st.write(mean_by_outcome)

if __name__ == '__main__':
    app()
