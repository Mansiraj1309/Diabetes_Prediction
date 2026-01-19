

Diabetes Prediction using Machine Learning

 Overview

This project is a Machine Learning–based system that predicts whether a person has diabetes using medical and lifestyle data. It follows a complete ML pipeline from data preprocessing to model deployment.

⚙️ Technologies
	•	Python
	•	Pandas, NumPy
	•	Scikit-learn
	•	Flask
	•	Pickle

📊 Dataset

The dataset contains patient health features such as age, BMI, glucose level, and blood pressure.
Target:
	•	0 → Non-Diabetic
	•	1 → Diabetic

🔄 Workflow
	1.	Data loading and preprocessing
	2.	Feature scaling and encoding
	3.	Train-test split
	4.	Model training using Support Vector Machine (SVM)
	5.	Model evaluation and saving
	6.	Deployment using Flask

📁 Project Structure

Diabetes_Prediction/
├── app.py
├── diabetes_prediction_2.0.ipynb
├── diabetes_prediction_dataset.csv
├── svm_model.pkl
├── scaler.pkl
├── label_encoder.pkl
└── README.md


How to Run

git clone https://github.com/Mansiraj1309/Diabetes_Prediction.git
pip install -r requirements.txt
python app.py

Open: http://127.0.0.1:5000/

Output

Predicts Diabetic / Non-Diabetic based on user input.

Conclusion

This project demonstrates an end-to-end ML classification system with real-world healthcare application.
