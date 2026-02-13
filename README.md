🏦 Bank Marketing Classification – Model Comparison

a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether a client will subscribe to a term deposit based on demographic and campaign-related features. The goal is to evaluate different algorithms using multiple performance metrics and identify the most effective model.

b. Dataset Description

The dataset used in this project is the Bank Marketing Dataset from the UCI Machine Learning Repository.

Total Instances: 41,188

Total Features: 16 input features

Target Variable: y (Binary – Yes/No)

Problem Type: Binary Classification

The dataset contains client demographic information (age, job, marital status, education), financial information (balance, housing loan, personal loan), and campaign-related features (contact type, duration, previous outcome).


c. Models Used and Performance Comparison

The following six classification models were implemented and evaluated on the same dataset using an 80:20 stratified train-test split.

| ML Model                 | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression      | 0.8829   | 0.8550 | 0.4750    | 0.1827 | 0.2639   | 0.2428 |
| Decision Tree            | 0.8575   | 0.6727 | 0.3913    | 0.4327 | 0.4110   | 0.3307 |
| KNN                      | 0.8917   | 0.7330 | 0.5789    | 0.2115 | 0.3099   | 0.3046 |
| Naive Bayes              | 0.8409   | 0.8195 | 0.3387    | 0.4038 | 0.3684   | 0.2796 |
| Random Forest (Ensemble) | 0.8884   | 0.8995 | 0.5254    | 0.2981 | 0.3804   | 0.3399 |
| XGBoost (Ensemble)       | 0.8895   | 0.9095 | 0.5312    | 0.3269 | 0.4048   | 0.3601 |


d. Observations on Model Performance

| ML Model            | Observation about Model Performance                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression | Achieved strong AUC but low recall due to conservative predictions under class imbalance. Serves as a solid baseline model. |
| Decision Tree       | Improved recall significantly but showed lower AUC, indicating weaker global class separation and potential overfitting.    |
| KNN                 | Achieved the highest accuracy and precision but struggled with recall due to dominance of majority class neighbors.         |
| Naive Bayes         | Provided balanced recall performance but overall moderate accuracy due to independence assumptions between features.        |
| Random Forest       | Demonstrated strong overall performance with high AUC and MCC, reducing overfitting compared to a single decision tree.     |
| XGBoost             | Delivered the best overall performance with highest AUC and MCC, showing superior ensemble learning capability.             |
