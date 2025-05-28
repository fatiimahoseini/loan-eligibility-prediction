# Loan Eligibility Prediction 🏦

## Overview
This project aims to automate the loan approval process by predicting loan eligibility based on customer data. I implement a complete machine learning pipeline, from data preprocessing and exploratory data analysis (EDA) to training and evaluating various classification models, and finally, optimizing their performance. ✨

## Project Goal
The primary goal is to build a robust predictive model 🤖 that can effectively determine if an applicant is eligible for a loan, thereby streamlining the approval process and reducing manual effort.

## Dataset
The dataset used in this project is `loan.csv`, containing financial and demographic information of loan applicants. 📊
Below is a summary of the dataset's columns:

| Column Name | Data Type | Non-Null (%) | Unique Values (Examples) | Description |
|:------------|:----------|:-------------|:-------------------------|:------------|
| Loan_ID | object | 614 (0.00%) | LP001002, LP001003, LP001005, LP001006, LP001008... | Unique Loan ID. |
| Gender | object | 601 (2.12%) | Male, Female, nan | Applicant's gender (Male/Female). |
| Married | object | 611 (0.49%) | Yes, No, nan | Marital status (Yes/No). |
| Dependents | object | 599 (2.44%) | 0, 1, 2, 3+, nan | Number of dependents (0, 1, 2, 3+). |
| Education | object | 614 (0.00%) | Graduate, Not Graduate | Applicant's education level (Graduate/Not Graduate). |
| Self_Employed | object | 582 (5.21%) | No, Yes, nan | Whether applicant is self-employed (Yes/No). |
| ApplicantIncome | int64 | 614 (0.00%) | 5849, 3000, 2583, 6000, 5417... | Applicant's monthly income. |
| CoapplicantIncome | float64 | 614 (0.00%) | 0.0, 1500.0, 1833.0, 916.0, 1750.0... | Co-applicant's monthly income. |
| LoanAmount | float64 | 592 (3.58%) | 128.0, 66.0, 120.0, 141.0, 113.0... | Loan amount in thousands. |
| Loan_Amount_Term | float64 | 600 (2.28%) | 360.0, 120.0, 240.0, 180.0, 60.0... | Loan term in months. |
| Credit_History | float64 | 564 (8.14%) | 1.0, 0.0, nan | Credit history meets guidelines (1.0/0.0). |
| Property_Area | object | 614 (0.00%) | Urban, Rural, Semiurban | Property area (Rural/Semiurban/Urban). |
| Loan_Status | object | 614 (0.00%) | Y, N | Loan approved (Y/N) - Target Variable. |

## Key Features & Methodology

My approach involved the following key steps:

1.  **Data Preprocessing:** 🧹
    * Handling missing values using median (numerical) and mode (categorical) imputation.
    * Encoding categorical features (Label Encoding for binary, One-Hot Encoding for multi-class like `Dependents` and `Property_Area`).
    * Splitting data into stratified training, validation, and test sets (65-20-15).
    * Standardizing numerical features using `StandardScaler`.

2.  **Exploratory Data Analysis (EDA) & Feature Engineering:** 💡
    * Conducted EDA to understand data distributions and correlations. Identified skewed numerical features and outliers (e.g., `ApplicantIncome`, `LoanAmount`).
    * **Engineered New Features:** Created `TotalIncome`, `LoanAmount_per_TotalIncome`, and `LoanAmount_per_Term_Monthly` to capture more relevant financial indicators.
    * **Outlier Management:** Opted for **logarithmic transformation** (`np.log1p`) to mitigate the impact of skewed outliers, rather than removal, to preserve valuable data.
    * Handled any new missing values resulting from engineered features.

3.  **Model Training & Evaluation:** 📈
    * Trained and evaluated three classification models: Logistic Regression, K-Nearest Neighbors (KNN), and Artificial Neural Network (ANN).
    * Developed a modular `run_model_pipeline` function to streamline the training and evaluation process across different models and feature sets, enhancing code reusability and clarity.

4.  **Performance Optimization:** 🚀
    * Applied **Hyperparameter Tuning** using `GridSearchCV` for Logistic Regression on the log-transformed data to find optimal model parameters.

## Results Highlight

The feature engineering phase proved to be highly impactful, significantly boosting the performance of all models. 🏆

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
| :-------------------------------- | :------- | :-------- | :----- | :------- | :------ |
| **Logistic Regression (Before FE)** | 0.8197 | 0.81 | 0.9643 | 0.8804 | 0.7735 |
| **Logistic Regression (After FE)** | 0.8602 | 0.84 | 0.9843 | 0.9064 | 0.7613 |
| **Logistic Regression (Log-Transformed & Tuned)** | 0.8495 | 0.8289 | 0.9844 | 0.9000 | 0.7829 |
| **KNN (Before FE)** | 0.7295 | 0.7525 | 0.9048 | 0.8216 | 0.6778 |
| **KNN (After FE)** | 0.7741 | 0.7792 | 0.9375 | 0.8510 | 0.7416 |
| **ANN (Before FE)** | 0.6885 | 0.7805 | 0.7619 | 0.7711 | 0.6660 |
| **ANN (After FE)** | 0.8172 | 0.8405 | 0.9062 | 0.8721 | 0.7742 |

**Conclusion:**
**Logistic Regression (after initial feature engineering)** emerged as the top performer (Accuracy: 0.8602, F1-Score: 0.9064, Recall: 0.9843) on the unseen test set, making it the recommended choice due to its strong metrics and interpretability. The dramatic improvement observed in ANN after feature engineering also highlights its potential. ✨

## Future Work 🚀

* Conduct more exhaustive hyperparameter tuning for KNN and ANN.
* Experiment with Ensemble Methods (e.g., Random Forest, XGBoost) for potentially higher accuracy.
* Explore advanced outlier detection and treatment strategies.
* Investigate model interpretability tools (SHAP, LIME) for deeper insights into predictions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

[Fatima Hosseini] - [fatiimahoseini@gmail.com] - [Linkedin](https://www.linkedin.com/in/fatiimahoseini)
