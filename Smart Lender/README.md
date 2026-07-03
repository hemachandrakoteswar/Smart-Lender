# Smart Lender: Loan Eligibility Prediction System

Smart Lender is a production-grade end-to-end Machine Learning web application designed to predict loan eligibility based on applicant demographics, financial standing, credit records, and property details. 

The system implements robust preprocessing, class balancing using SMOTE, and an ensemble Random Forest classifier which achieves a test set accuracy of **81.36%**. A Flask backend hosts the model, exposing a premium, modern, glassmorphic banking user interface.

---

## 1. Project Directory Structure
```text
smart-lender/
├── Dataset/
│   └── loan_prediction.csv
├── Training/
│   ├── Loan_Prediction_using_ML.ipynb
│   └── train.py
├── templates/
│   ├── home.html
│   ├── predict.html
│   └── submit.html
├── static/
│   └── css/
│       └── input.css
├── scale1.pkl
├── rdf.pkl
├── app1.py
├── vercel.json
├── requirements.txt
└── README.md
```

---

## 2. Technical Overview & Pipeline

### Epic 1 & 2: Dataset & Exploratory Data Analysis
- The project loads `loan_prediction.csv` containing columns for Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, and the target label `Loan_Status`.
- Exploratory data analysis (univariate, bivariate, and multivariate visualizations) is performed using `matplotlib` and `seaborn`.

### Epic 3: Preprocessing & Feature Engineering
- **Categorical Mappings**:
  - `Gender`: `{'Female': 1, 'Male': 0}`
  - `Married`: `{'Yes': 1, 'No': 0}`
  - `Education`: `{'Graduate': 1, 'Not Graduate': 0}`
  - `Self_Employed`: `{'Yes': 1, 'No': 0}`
  - `Property_Area`: `{'Urban': 2, 'Semiurban': 1, 'Rural': 0}`
  - `Loan_Status`: `{'Y': 1, 'N': 0}`
- **Dependents Sanitization**: Stripped `+` signs (e.g., `3+` -> `3`) and cast to integer.
- **Imputation**: Filled missing values in categorical fields with their `.mode()[0]` and numerical features with column means.
- **SMOTE (Synthetic Minority Over-sampling)**: Used to handle class imbalance, balancing the target classes.
- **StandardScaler**: Scaled features and exported the fitted scaler to `scale1.pkl`.
- **Train-Test Split**: Set `test_size=0.33` and `random_state=42`.

### Epic 4: Model Building & Selection
- Trained four classification models: Decision Tree, Random Forest, K-Nearest Neighbors, and Gradient Boosting.
- Compared performance using 5-fold cross-validation.
- Automatically saved the best performing model (Random Forest, accuracy: **81.36%**) as `rdf.pkl`.

### Epic 5: Flask Application
- The Flask app (`app1.py`) loads `scale1.pkl` and `rdf.pkl`.
- It processes user inputs, scales them, maps the output to `"Loan will be Approved"` or `"Loan will Not be Approved"`, and displays the result on a premium styled results page.

---

## 3. Installation Guide using Anaconda

### Step 1: Clone or Copy the Repository
Place the project directory on your system.

### Step 2: Open Anaconda Prompt
Open the **Anaconda Prompt** or terminal window.

### Step 3: Create a Virtual Environment
Run the following command to create a virtual environment with Python 3.10 (or later):
```bash
conda create -n smart-lender python=3.10 -y
```

### Step 4: Activate Environment
```bash
conda activate smart-lender
```

### Step 5: Install Dependencies
Navigate to your project directory and install the requirements:
```bash
cd "path/to/smart-lender"
pip install -r requirements.txt
```

---

## 4. How to Test & Run Locally

### Step 1: Run Model Training (Optional)
If you want to train the model from scratch and generate new pickles, run:
```bash
python Training/train.py
```

### Step 2: Run the Web App
Start the Flask development server:
```bash
python app1.py
```

### Step 3: Access the Interface
Open your browser and navigate to:
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

1. On the **Landing Page**, read the overview and click **Check Loan Eligibility**.
2. Fill out the **Applicant Details** form with sample parameters.
3. Click **Submit Application** to view the instant AI prediction.

---

## 5. Deployment on Vercel

To deploy this Flask application to Vercel, the project uses a custom `vercel.json` configuration file to define `app1.py` as the main serverless entry point and direct incoming traffic appropriately.

### Vercel Configuration (`vercel.json`)
The application root contains a `vercel.json` file with the following contents:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app1.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app1.py"
    }
  ]
}
```

### Deployment Steps
1. Install the Vercel CLI (if not already installed):
   ```bash
   npm install -g vercel
   ```
2. Log in to your Vercel account:
   ```bash
   vercel login
   ```
3. Run the deployment command from the project root:
   ```bash
   vercel
   ```
4. Follow the command-line prompts to link the project and deploy it. Vercel will automatically locate `vercel.json`, install dependencies from `requirements.txt`, compile the Python serverless function, and host the Smart Lender Loan Prediction application.
