import pandas as pd
import numpy as np
import pickle
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

def main():
    print("Step 1: Reading dataset...")
    # Load dataset
    data_path = os.path.join("Dataset", "loan_prediction.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"Loaded dataset with shape: {data.shape}")
    
    print("\nStep 2: Preprocessing and categorical mapping...")
    # Strip '+' from Dependents
    data['Dependents'] = data['Dependents'].str.replace('+', '', regex=False)
    
    # Missing value imputation
    # Categorical/discrete variables -> mode
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
        
    # Continuous variables -> mean
    data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mean())
    
    # Mappings
    gender_map = {'Female': 1, 'Male': 0}
    married_map = {'Yes': 1, 'No': 0}
    education_map = {'Graduate': 1, 'Not Graduate': 0}
    self_employed_map = {'Yes': 1, 'No': 0}
    property_area_map = {'Urban': 2, 'Semiurban': 1, 'Rural': 0}
    loan_status_map = {'Y': 1, 'N': 0}
    
    data['Gender'] = data['Gender'].map(gender_map)
    data['Married'] = data['Married'].map(married_map)
    data['Education'] = data['Education'].map(education_map)
    data['Self_Employed'] = data['Self_Employed'].map(self_employed_map)
    data['Property_Area'] = data['Property_Area'].map(property_area_map)
    data['Loan_Status'] = data['Loan_Status'].map(loan_status_map)
    
    # Cast columns to int64 to match the environment notebook
    data['Gender'] = data['Gender'].astype('int64')
    data['Married'] = data['Married'].astype('int64')
    data['Dependents'] = data['Dependents'].astype('int64')
    data['Self_Employed'] = data['Self_Employed'].astype('int64')
    data['LoanAmount'] = data['LoanAmount'].astype('int64')
    data['CoapplicantIncome'] = data['CoapplicantIncome'].astype('int64')
    data['Loan_Amount_Term'] = data['Loan_Amount_Term'].astype('int64')
    data['Credit_History'] = data['Credit_History'].astype('int64')
    
    print("Nulls remaining in dataset:", data.isnull().sum().sum())
    
    # Splitting features and target
    X = data.drop(columns=['Loan_ID', 'Loan_Status'])
    y = data['Loan_Status']
    
    print("\nStep 3: Target class counts before SMOTE:")
    print(y.value_counts())
    
    # Balance target classes using SMOTE
    smote = SMOTE(random_state=42)
    X_bal, y_bal = smote.fit_resample(X, y)
    
    print("Target class counts after SMOTE:")
    print(y_bal.value_counts())
    
    print("\nStep 4: Feature Scaling using StandardScaler...")
    scaler = StandardScaler()
    X_bal_scaled = scaler.fit_transform(X_bal)
    
    # Save the scaler object
    scaler_path = "scale1.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
    
    X_bal = pd.DataFrame(X_bal_scaled, columns=X_bal.columns)
    
    # Split training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.33, random_state=42)
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    print("\nStep 5: Training and evaluating classifiers...")
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gradient Boosting (XGB)": GradientBoostingClassifier(random_state=42)
    }
    
    best_acc = 0.0
    best_model_name = ""
    best_model = None
    
    for name, model in models.items():
        # Perform 5-fold cross validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        print(f"\nModel: {name}")
        print(f"  - CV Mean Accuracy: {np.mean(cv_scores):.4f}")
        print(f"  - Test Accuracy   : {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = model
            
    print("\n" + "=" * 50)
    print(f"Best Model Selected: {best_model_name} with Accuracy {best_acc:.4f}")
    print("=" * 50)
    
    # Detailed evaluation
    best_preds = best_model.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, best_preds))
    print("\nClassification Report:")
    print(classification_report(y_test, best_preds))
    
    # Save the best model
    model_path = "rdf.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"\nBest model saved to {model_path}")

if __name__ == "__main__":
    main()
