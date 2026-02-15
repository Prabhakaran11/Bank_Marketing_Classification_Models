# Bank Marketing Classification – Model Comparison

## a. Problem Statement
The objective of this project is to develop and compare multiple machine learning classification models to predict whether a client will subscribe to a term deposit based on demographic and marketing campaign attributes. The goal is to evaluate different classification algorithms using multiple performance metrics and identify the most effective model for this binary classification problem.

## b. Dataset Description
The dataset used in this project is the Bank Marketing Dataset from the UCI Machine Learning Repository.

* Source - https://archive.ics.uci.edu/dataset/222/bank+marketing
* Total Instances: 45,211 (Reduced to 10,000 instances for training and 2,000 instances for prediction due to Streamlit free tier limitations)
* Total Features: 16 input features
* Target Variable: y (Binary – Yes/No)
* Problem Type: Binary Classification

The dataset includes demographic features (age, job, marital status, education), financial details (balance, housing loan, personal loan), and campaign-related information (contact type, duration, previous outcome).

The dataset was preprocessed using encoding for categorical variables and feature scaling for numerical variables. Stratified sampling was used to preserve class distribution during data splitting. All models were configured with class balancing techniques (class_weight='balanced' for tree-based models, scale_pos_weight for XGBoost, and threshold adjustment for KNN and Naive Bayes) to handle the imbalanced dataset (approximately 88% non-subscribers, 12% subscribers).

## c. Models Used and Performance Comparison
The following six classification models were implemented and evaluated:

* Logistic Regression
* Decision Tree
* K-Nearest Neighbors (KNN)
* Naive Bayes (Gaussian)
* Random Forest (Ensemble)
* XGBoost (Ensemble)

All models were evaluated using the following metrics:

* Accuracy
* Precision (for positive class - subscriptions)
* Recall (for positive class - subscriptions)
* F1-Score
* AUC Score
* Matthews Correlation Coefficient (MCC)

### Model Performance Comparison Table

| ML Model            | Accuracy | Precision | Recall | F1-Score | AUC Score | MCC Score |
| ------------------- | -------- | --------- | ------ | -------- | --------- | --------- |
| Logistic Regression | 79.40%   | 33.21%    | 75.21% | 46.07%   | 86.02%    | 0.402     |
| Decision Tree       | 78.50%   | 32.44%    | 77.35% | 45.71%   | 83.75%    | 0.401     |
| KNN                 | 82.05%   | 36.32%    | 70.94% | 48.05%   | 84.85%    | 0.417     |
| Naive Bayes         | 85.85%   | 40.89%    | 47.01% | 43.74%   | 82.35%    | 0.358     |
| Random Forest       | 84.00%   | 40.09%    | 74.36% | 52.10%   | 89.86%    | 0.465     |
| XGBoost             | 84.80%   | 41.94%    | 77.78% | 54.49%   | 90.47%    | 0.495     |


## d. Observations on Model Performance

| ML Model            | Observation about Model Performance |
| ------------------- | ----------------------------------- |
| Logistic Regression | Logistic Regression with class balancing achieved 79.40% accuracy and strong recall (75.21%), making it effective at identifying potential subscribers. The model was configured with `class_weight='balanced'` to handle the imbalanced dataset. While precision is moderate (33.21%), the high recall ensures the model captures most subscription opportunities, which is critical for marketing campaigns. The AUC score of 86.02% demonstrates good discriminative ability, and its interpretability makes it a practical choice for deployment. |
| Decision Tree       | The Decision Tree model with regularization (max_depth=10, min_samples_split=20, min_samples_leaf=10) achieved 78.50% accuracy and strong recall (77.35%). The AUC score of 83.75% demonstrates good discriminative ability, a significant improvement over an unconstrained tree which would suffer from overfitting. While performance is solid, the model is naturally less robust than ensemble methods that aggregate multiple trees. The high recall makes it effective for capturing subscription opportunities. |
| KNN                 | K-Nearest Neighbors with distance-weighted voting and threshold adjustment achieved competitive performance with 82.05% accuracy and 84.85% AUC. Since KNN lacks built-in class balancing mechanisms, prediction threshold adjustment was applied to handle the imbalanced dataset, significantly improving recall to 70.94%. The model achieved an MCC of 0.417, ranking 4th overall. However, KNN's performance was limited by the curse of dimensionality with 16 features and sensitivity to local class distributions. |
| Naive Bayes         | Gaussian Naive Bayes achieved 85.85% accuracy but showed the lowest recall (47.01%) and MCC (0.358) among all models. The feature independence assumption significantly limited its predictive power, as bank marketing features exhibit strong correlations (e.g., age-job, balance-housing, education-job). Even with threshold adjustment to handle class imbalance, the model struggled to achieve the recall levels of other algorithms (70-78%). This demonstrates the importance of model assumptions: when features violate the independence assumption, Naive Bayes cannot capture the complex relationships that ensemble methods learn effectively. |
| Random Forest       | Random Forest demonstrated the strongest performance among standalone and ensemble models, achieving 84.00% accuracy and an outstanding AUC score of 89.86%. The ensemble approach of aggregating 100 decision trees with bootstrap sampling and random feature selection significantly improved upon the single Decision Tree model. With the highest MCC score (0.465) among non-boosting models and F1-score (52.10%), Random Forest provides robust and balanced predictions. The model effectively handles class imbalance through `class_weight='balanced'` while maintaining strong recall (74.36%) and improved precision (40.09%). |
| XGBoost             | XGBoost achieved the best overall performance across all metrics, with the highest AUC score (90.47%), MCC score (0.495), and F1-score (54.49%). The gradient boosting framework with `scale_pos_weight` for class imbalance handling provided superior discriminative ability compared to all other models. XGBoost's sequential learning approach, where each tree corrects errors from previous trees, combined with built-in regularization, resulted in the most balanced and robust predictions. The model excelled in both precision (41.94%) and recall (77.78%), making it the optimal choice for this bank marketing classification task. |

## e. How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/Prabhakaran11/Bank_Marketing_Classification_Models
cd Bank_Marketing_Classification_Models
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
python -m venv venv
```

Activate environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Models (Optional - Pre-trained models included)
If you want to retrain the models with your own parameters:
```bash
python train_all_models.py
```

This will train all six models and save them in the `saved_models/` directory.

### 5. Run the Streamlit Application
```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`

### 6. Using the Application
1. Select a machine learning model from the sidebar dropdown
2. Download the sample prediction dataset using the download button
3. Upload the prediction dataset CSV file (semicolon-delimited)
4. View the results:
   - **Evaluation metrics**: Accuracy, Precision, Recall, F1-Score, AUC, MCC
   - **Confusion matrix**: Visual representation of predictions
   - **Prediction distribution**: Class distribution chart
   - **Classification report**: Detailed per-class metrics
5. Download predictions as CSV file

### 7. Deployment
The application is deployed on Streamlit Community Cloud and can be accessed at: [Live Application Link]

## f. Repository Structure

Below is the project folder structure:
```
bank_marketing_classification_models/
│
├── app.py                         # Main Streamlit application
├── train_all_models.py            # Script to train and save all models
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
│
├── data/
│   ├── bank_training_data.csv     # Training dataset (10,000 samples)
│   └── bank_prediction_data.csv   # Prediction dataset (2,000 samples)
│
├── saved_models/                  # Pre-trained model files
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
├── model/
│   ├── preprocessing.py           # Data preprocessing functions
│   ├── logistic_regression.py     # Logistic Regression implementation
│   ├── decision_tree.py           # Decision Tree implementation
│   ├── knn.py                     # K-Nearest Neighbors implementation
│   ├── naive_bayes.py             # Gaussian Naive Bayes implementation
│   ├── random_forest.py           # Random Forest implementation
│   └── xgboost_model.py           # XGBoost implementation
│
└── .gitignore                     # Files excluded from version control
```

## Final Conclusion

Among all models evaluated, **XGBoost demonstrated the best overall performance** with the highest AUC score (90.47%), MCC score (0.495), and F1-score (54.49%). Random Forest achieved comparable performance with an AUC of 89.86% and MCC of 0.465, confirming that ensemble methods significantly outperform standalone models.

### Key Findings:

- **Ensemble models (XGBoost and Random Forest)** substantially outperformed standalone models (Logistic Regression, Decision Tree, KNN, Naive Bayes) due to their ability to capture complex patterns, handle feature interactions, and reduce overfitting through aggregation.

- **XGBoost's gradient boosting** approach provided a slight but consistent advantage over Random Forest's bagging approach, with improvements of 0.61% in AUC and 0.030 in MCC.

- **Class balancing** (using `class_weight='balanced'`, `scale_pos_weight`, and threshold adjustment) was critical for handling the imbalanced dataset (88% non-subscribers, 12% subscribers), enabling models to achieve high recall (70-78% for most models) while maintaining reasonable precision (32-42%).

- **Model progression** clearly demonstrated the value of ensembling: Decision Tree (83.75% AUC) → Random Forest (89.86% AUC) → XGBoost (90.47% AUC).

- **Naive Bayes's poor recall (47%)** highlighted the importance of choosing models that can handle correlated features, which are prevalent in bank marketing data. The feature independence assumption severely limited its ability to capture relationships between demographic and financial variables.

### Performance Rankings:

| Rank | Model | AUC | MCC | F1 | Key Strength |
|------|-------|-----|-----|-----|--------------|
|  1st | **XGBoost** | 90.47% | 0.495 | 54.49% | Best overall, highest discrimination |
|  2nd | **Random Forest** | 89.86% | 0.465 | 52.10% | Excellent robustness, high accuracy |
|  3rd | **Logistic Regression** | 86.02% | 0.402 | 46.07% | Interpretable, good baseline |
| 4th | **KNN** | 84.85% | 0.417 | 48.05% | Good with threshold tuning |
| 5th | **Decision Tree** | 83.75% | 0.401 | 45.71% | High recall, needs regularization |
| 6th | **Naive Bayes** | 82.35% | 0.358 | 43.74% | Fast but limited by assumptions |

### Recommendation

**XGBoost is the optimal model for production deployment**, offering the best balance of discrimination ability (90.47% AUC), precision (41.94%), recall (77.78%), and overall reliability (0.495 MCC). Its superior performance in identifying potential subscribers while minimizing false positives makes it ideal for optimizing marketing campaign ROI.

---

## 📚 Technologies Used

- **Python 3.8+**
- **Streamlit** - Web application framework
- **Scikit-learn** - Machine learning library
- **XGBoost** - Gradient boosting framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Imbalanced-learn** - Class balancing techniques

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

**[Prabhakaran Srinivasan]**
- GitHub: [@Prabhakaran11](https://github.com/Prabhakaran11)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Bank Marketing Dataset
- Streamlit Community Cloud for free hosting
