import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    matthews_corrcoef
)

# Import all model functions
from model.logistic_regression import train_and_evaluate_logistic_regression
from model.decision_tree import train_and_evaluate_decision_tree
from model.knn import train_and_evaluate_knn
from model.naive_bayes import train_and_evaluate_naive_bayes
from model.random_forest import train_and_evaluate_random_forest
from model.xgboost import train_and_evaluate_xgboost

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Classification Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #ddd;
        margin: 1rem 0;
    }
    .model-info {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown('<h1 class="main-header">🏦 Bank Marketing Classification – Model Comparison Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Train and evaluate machine learning models to predict term deposit subscriptions</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1f77b4 0%, #0d47a1 100%); 
                padding: 2rem; 
                border-radius: 10px; 
                text-align: center;
                margin-bottom: 1rem;'>
        <h2 style='color: white; margin: 0;'>🏦 ML Platform</h2>
        <p style='color: #e3f2fd; margin: 0.5rem 0 0 0;'>Bank Marketing Classification</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 🎯 Model Configuration")
    
    model_option = st.selectbox(
        "Select ML Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ],
        help="Choose the machine learning algorithm for prediction"
    )
    
    st.markdown("---")
    
    # Model descriptions
    model_descriptions = {
        "Logistic Regression": "📊 Linear model for binary classification. Fast and interpretable.",
        "Decision Tree": "🌳 Tree-based model that learns decision rules. Easy to visualize.",
        "KNN": "👥 Instance-based learning using nearest neighbors.",
        "Naive Bayes": "📈 Probabilistic classifier based on Bayes' theorem.",
        "Random Forest": "🌲 Ensemble of decision trees. High accuracy and robust.",
        "XGBoost": "⚡ Gradient boosting algorithm. State-of-the-art performance."
    }
    
    st.info(model_descriptions[model_option])
    
    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
    This app compares multiple ML models to predict term deposit subscription based on client and campaign data.
    
    **Dataset Features:**
    - Demographics
    - Campaign information
    - Economic indicators
    """)

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📥 Test Dataset")
    st.markdown("Download the test dataset to try the prediction functionality.")

with col2:
    with open("data/bank_prediction_data.csv", "rb") as file:
        st.download_button(
            label="⬇️ Download Test CSV",
            data=file,
            file_name="bank_test_sample.csv",
            mime="text/csv",
            use_container_width=True
        )

st.markdown("---")

# Model Training Section
with st.spinner(f'🔄 Training {model_option} model on full dataset...'):
    df_full = pd.read_csv("data/bank_training_data.csv", sep=';')
    
    if model_option == "Logistic Regression":
        model, _ = train_and_evaluate_logistic_regression(df_full)
    elif model_option == "Decision Tree":
        model, _ = train_and_evaluate_decision_tree(df_full)
    elif model_option == "KNN":
        model, _ = train_and_evaluate_knn(df_full)
    elif model_option == "Naive Bayes":
        model, _ = train_and_evaluate_naive_bayes(df_full)
    elif model_option == "Random Forest":
        model, _ = train_and_evaluate_random_forest(df_full)
    elif model_option == "XGBoost":
        model, _ = train_and_evaluate_xgboost(df_full)

st.success(f'✅ {model_option} model trained successfully!')

# Upload and Prediction Section
st.markdown("### 🔮 Make Predictions")

uploaded_file = st.file_uploader(
    "Upload your test dataset (CSV format)",
    type=["csv"],
    help="Upload a CSV file with the same format as the sample dataset"
)

if uploaded_file is not None:
    try:
        # Load and process test data
        df_test = pd.read_csv(uploaded_file, sep=';')
        
        # Display dataset preview
        with st.expander("📊 Preview Uploaded Dataset", expanded=False):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Records", len(df_test))
            col2.metric("Features", len(df_test.columns))
            col3.metric("Memory Usage", f"{df_test.memory_usage(deep=True).sum() / 1024:.2f} KB")
            st.dataframe(df_test.head(10), use_container_width=True)
        
        from model.preprocessing import load_and_preprocess_data
        
        X_test, y_test = load_and_preprocess_data(df_test)
        y_pred = model.predict(X_test)
        
        st.markdown("---")
        st.markdown("### 📊 Model Performance Results")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        
        # Calculate AUC Score
        try:
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc = None
        except ValueError as e:
            st.warning(f"⚠️ AUC Score could not be calculated: {str(e)}")
            auc = None
        except Exception as e:
            st.error(f"❌ Unexpected error calculating AUC: {str(e)}")
            auc = None

        # Calculate MCC Score
        mcc = matthews_corrcoef(y_test, y_pred)

        # Display metrics in cards
        metric_col1, metric_col2, metric_col3, metric_col4, metric_col5, metric_col6 = st.columns(6)
        
        with metric_col1:
            st.metric(
                label="🎯 Accuracy",
                value=f"{accuracy:.2%}",
                delta=None
            )
        
        with metric_col2:
            st.metric(
                label="🔍 Precision",
                value=f"{precision:.2%}",
                delta=None
            )
        
        with metric_col3:
            st.metric(
                label="📈 Recall",
                value=f"{recall:.2%}",
                delta=None
            )
        
        with metric_col4:
            st.metric(
                label="⚖️ F1-Score",
                value=f"{f1:.2%}",
                delta=None
            )
        
        with metric_col5:
            if auc is not None:
                st.metric(
                    label="📊 AUC Score",
                    value=f"{auc:.2%}",
                    delta=None
                )
            else:
                st.metric(
                    label="📊 AUC Score",
                    value="N/A",
                    delta=None,
                    help="Model does not support probability predictions"
                )
        
        with metric_col6:
            st.metric(
                label="🔗 MCC Score",
                value=f"{mcc:.3f}",
                delta=None,
                help="Matthews Correlation Coefficient (-1 to +1)"
            )
        
        st.markdown("---")
        
    # Visualization Section
        st.markdown("---")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.markdown("#### 🔥 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                cbar_kws={'label': 'Count'},
                square=True,
                linewidths=1,
                linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'}
            )
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            ax.set_title(f'Confusion Matrix - {model_option}', fontsize=13, fontweight='bold', pad=15)
            plt.tight_layout()
            st.pyplot(fig)

        with viz_col2:
            st.markdown("#### 📊 Prediction Distribution")
            
            fig, ax = plt.subplots(figsize=(6, 5))
            
            pred_counts = pd.Series(y_pred).value_counts().sort_index()
            colors = ['#ff6b6b', '#4ecdc4']
            labels = ['No Subscription', 'Subscription']
            
            # Donut chart
            wedges, texts, autotexts = ax.pie(
                pred_counts,
                labels=labels,
                colors=colors,
                autopct='%1.1f%%',
                startangle=90,
                explode=(0, 0.05),
                textprops={'fontsize': 10, 'fontweight': 'bold'},
                pctdistance=0.82,
                wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
            
            # Center text
            ax.text(0, 0, f'Total\n{sum(pred_counts)}', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
            
            ax.set_title('Distribution of Predictions', fontsize=13, fontweight='bold', pad=15)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Classification Report
        st.markdown("---")
        st.markdown("#### 📄 Detailed Classification Report")

        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()

        # Adding labels to explain what each row means
        index_labels = {
            '0': '0 (No Subscription)',
            '1': '1 (Yes Subscription) ⭐',
            'accuracy': 'Overall Accuracy',
            'macro avg': 'Macro Average',
            'weighted avg': 'Weighted Average'
        }

        # Apply labels
        report_df.index = report_df.index.map(lambda x: index_labels.get(str(x), str(x)))

        report_df = report_df.reset_index()
        report_df = report_df.rename(columns={'index': 'Class'})

        # Style the dataframe
        styled_df = report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score'])\
                                .format({'precision': '{:.3f}', 'recall': '{:.3f}', 'f1-score': '{:.3f}', 'support': '{:.0f}'})\
                                .set_properties(**{'text-align': 'center'})

        st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Add note below
        st.caption("⭐ The metrics in the cards above show performance for Class 1 (Yes Subscription) only, as this is our target class.")

        
        # Download predictions
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        
        results_df = df_test.copy()
        results_df['Prediction'] = y_pred
        results_df['Actual'] = y_test.values if hasattr(y_test, 'values') else y_test
        
        csv = results_df.to_csv(index=False)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv,
                file_name=f"predictions_{model_option.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has the correct format and delimiter (semicolon ';')")

else:
    # Empty state
    st.info("👆 Upload a test dataset to see predictions and model performance metrics")
    
    # Show example of expected format
    with st.expander("ℹ️ Expected CSV Format"):
        st.markdown("""
        Your CSV file should contain the following columns:
        - **age**: Client age
        - **job**: Type of job
        - **marital**: Marital status
        - **education**: Education level
        - **default**: Has credit in default?
        - **balance**: Account balance
        - **housing**: Has housing loan?
        - **loan**: Has personal loan?
        - **contact**: Contact communication type
        - **day**: Last contact day of month
        - **month**: Last contact month
        - **duration**: Last contact duration
        - **campaign**: Number of contacts during campaign
        - **pdays**: Days since last contact
        - **previous**: Number of previous contacts
        - **poutcome**: Outcome of previous campaign
        - **y**: Target variable (yes/no)
        
        Use semicolon (;) as delimiter.
        """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 2rem 0;'>
        <p>© 2026 | ML Assignment – Bank Marketing Classification</p>
    </div>
    """,
    unsafe_allow_html=True
)