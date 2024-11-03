import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

data = {
    'Income': ['<30', '30-70', '30-70', '30-70', '30-70', '30-70', '>70', '>70', '<30', '30-70', '30-70', '30-70'],
    'Criminal_Record': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes'],
    'EXP': ['1-5', '1', '1', '1-5', '>5', '1-5', '>5', '>5', '1-5', '1-5', '1-5', '>5'],
    'Loan_Approved': ['Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

le_income = LabelEncoder()
le_criminal = LabelEncoder()
le_exp = LabelEncoder()
le_loan = LabelEncoder()

df['Income_encoded'] = le_income.fit_transform(df['Income'])
df['Criminal_Record_encoded'] = le_criminal.fit_transform(df['Criminal_Record'])
df['EXP_encoded'] = le_exp.fit_transform(df['EXP'])
df['Loan_Approved_encoded'] = le_loan.fit_transform(df['Loan_Approved'])

X = df[['Income_encoded', 'Criminal_Record_encoded', 'EXP_encoded']]
y = df['Loan_Approved_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

y_pred = nb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

case_income = le_income.transform(['30-70'])[0]
case_criminal = le_criminal.transform(['Yes'])[0]
case_exp = le_exp.transform(['>5'])[0]

case = [[case_income, case_criminal, case_exp]]
probability = nb_model.predict_proba(case)

print("\nResults:")
print("-" * 50)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nProbability for the specific case:")
print(f"Probability of Loan Approval: {probability[0][1]:.2%}")
print(f"Probability of Loan Rejection: {probability[0][0]:.2%}")

print("\nDetailed Classification Report:")
print("-" * 50)
print(classification_report(y_test, y_pred, 
                          target_names=['Rejected', 'Approved']))
print("\nOriginal Data with Encoded Values:")
print("-" * 50)
print(df.to_string())