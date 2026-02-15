# Bank Marketing Classification – Model Comparison

## a. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models to predict whether a client will subscribe to a term deposit based on demographic and marketing campaign attributes. The goal is to evaluate different classification algorithms using multiple performance metrics and identify the most effective model for this binary classification problem.

## b. Dataset Description

The dataset used in this project is the Bank Marketing Dataset from the UCI Machine Learning Repository.

Source - https://archive.ics.uci.edu/dataset/222/bank+marketing

Total Instances: 45,211 (Reduced to 10000 instances for training and 2000 instances for prediction due to Streamlit free tier limitations) 

Total Features: 16 input features

Target Variable: y (Binary – Yes/No)

Problem Type: Binary Classification

The dataset includes demographic features (age, job, marital status, education), financial details (balance, housing loan, personal loan), and campaign-related information (contact type, duration, previous outcome).

The dataset was preprocessed using encoding for categorical variables and feature scaling for numerical variables. Stratified sampling was used to preserve class distribution during data splitting.


## c. Models Used and Performance Comparison

The following six classification models were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

All models were evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1-Score
* AUC Score
* Matthews Correlation Coefficient (MCC)

Model Performance Comparison Table
| ML Model            | Accuracy | Precision | Recall | F1-Score | AUC Score | MCC Score |
| ------------------- | -------- | --------- | ------ | -------- | --------- | --------- |
| Logistic Regression | 89.30%   | 87.47%    | 89.30% | 87.82%   | 85.58%    | 0.371     |
| Decision Tree       | 87.10%   | 87.34%    | 87.10% | 87.22%   | 69.71%    | 0.387     |
| KNN                 | 89.00%   | 86.94%    | 89.00% | 87.32%   | 80.30%    | 0.342     |
| Naive Bayes         | 85.85%   | 86.76%    | 85.85% | 86.27%   | 82.35%    | 0.358     |
| Random Forest       | 90.10%   | 88.72%    | 90.10% | 88.98%   | 90.31%    | 0.437     |
| XGBoost             | 89.85%   | 88.85%    | 89.85% | 89.20%   | 90.81%    | 0.454     |


## d. Observations on Model Performance

| ML Model            | Observation about Model Performance                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Logistic Regression achieved strong overall accuracy (89.30%) and balanced precision-recall performance (87.47% precision, 89.30% recall). It provided a solid baseline model with a good AUC score (85.58%), demonstrating that linear decision boundaries are reasonably effective for this dataset. Its interpretability and efficiency make it a practical choice for deployment.                                                                          |
| Decision Tree       | Decision Tree model achieved reasonable accuracy (87.10%) but showed a significantly lower AUC score (69.71%), indicating poor probability calibration and weak class separability. This large gap suggests the model may be overfitting or struggling with class imbalance, making it less reliable for ranking predictions despite decent classification accuracy.                                                                                         |
| KNN                 | KNN achieved competitive accuracy (89.00%) and good recall, performing comparably to Logistic Regression. However, its AUC score (80.30%) and MCC (0.342) were lower than ensemble methods, indicating moderate performance in distinguishing between classes. KNN's performance may be sensitive to the choice of k and distance metric.                                                                                                                    |
| Naive Bayes         | Gaussian Naive Bayes model achieved moderate performance with 85.85% accuracy and 82.35% AUC. The feature independence assumption likely limited its predictive power, as bank marketing data often contains correlated features (e.g., age and job type). Despite this, it provided reasonable baseline performance with fast training time.                                                                                                                |
| Random Forest       | Random Forest demonstrated strong performance across all metrics, achieving 90.10% accuracy and 90.31% AUC score. The ensemble approach of aggregating multiple decision trees effectively reduced overfitting and improved generalization. Its high MCC score (0.437) indicates better balanced classification performance compared to standalone models.                                                                                           |
| XGBoost             | XGBoost achieved the best overall performance with the highest AUC score (90.81%) and MCC score (0.454) among all models. Its gradient boosting framework and regularization techniques provided the most balanced and robust predictions, effectively handling class imbalance and capturing complex non-linear patterns in the data. This makes it the best-performing model in this study.                                                             | 

## e. How to Run the Project
1. Clone the Repository
```bash
git clone https://github.com/Prabhakaran11/Bank_Marketing_Classification_Models
cd Bank_Marketing_Classification_Models
```

2. Create and Activate Virtual Environment (Recommended)
```bash
python -m venv venv
```

Activate environment:

Windows
```bash
venv\Scripts\activate
```

Mac/Linux
```bash
source venv/bin/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Run the Streamlit Application
```bash
streamlit run app.py
```

The application will open automatically in your browser.

5. Using the Application
Select a machine learning model from the sidebar.
Download the sample prediction dataset.
Upload the prediction dataset CSV file.
View:
* Evaluation metrics
* Confusion matrix
* Classification report

6. Deployment
The application is deployed on Streamlit Community Cloud and can be accessed using the provided live application link. Final Conclusion
Among all models evaluated, the **ensemble methods (XGBoost and Random Forest) demonstrated superior performance** compared to standalone models. XGBoost achieved the highest AUC score (90.81%) and MCC (0.454), indicating the best overall discrimination ability and balanced classification performance. Random Forest achieved marginally higher accuracy (90.10%) but slightly lower AUC (90.31%). The superior performance of ensemble models can be attributed to their ability to capture complex patterns, handle feature interactions, and achieve better generalization through aggregation of multiple learners.


## f. Repository Structure

Below is the project folder structure:

```
bank_marketing_classification_models/
│
├── app.py                         # Main Streamlit application
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
│
├── data/
│   ├── bank_training_data.csv     # Training dataset
│   ├── bank_prediction_data.csv   # Prediction dataset(for evaluation)
│
├── model/
│   ├── preprocessing.py           # Data preprocessing functions
│   ├── logistic_regression.py     # Logistic Regression implementation
│   ├── decision_tree.py           # Decision Tree implementation
│   ├── knn.py                     # K-Nearest Neighbors implementation
│   ├── naive_bayes.py             # Gaussian Naive Bayes implementation
│   ├── random_forest.py           # Random Forest implementation
│   ├── xgboost_model.py           # XGBoost implementation
│
└── .gitignore                     # Files excluded from version control
```