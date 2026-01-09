import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Set page configuration
# Instead of st.title(...)


import base64

def set_bg_local(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_local("liver_image.jpg")  # works even with spaces



# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("project-data (1) (1).csv", delimiter=";")
    df = df.rename(columns={"protein   ": "protein"})
    
    # Convert protein to numeric
    df['protein'] = pd.to_numeric(df['protein'], errors='coerce')
    
    # Handle missing values
    df["albumin"] = df["albumin"].fillna(df["albumin"].mean())
    df["alkaline_phosphatase"] = df["alkaline_phosphatase"].fillna(df["alkaline_phosphatase"].mean())
    df["alanine_aminotransferase"] = df["alanine_aminotransferase"].fillna(df["alanine_aminotransferase"].mean())
    df["cholesterol"] = df["cholesterol"].fillna(df["cholesterol"].mean())
    df["protein"] = df["protein"].fillna(df["protein"].mean())
    
    return df

# Preprocess data for modeling
def preprocess_data(df):
    # Encode categorical variables
    le_sex = LabelEncoder()
    df['sex'] = le_sex.fit_transform(df['sex'])
    
    le_category = LabelEncoder()
    df['category_encoded'] = le_category.fit_transform(df['category'])
    
    # Features and target
    X = df.drop(['category', 'category_encoded'], axis=1)
    y = df['category_encoded']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le_category, scaler, X.columns

# Train models
def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(kernel='rbf', random_state=42,probability=True),
        "Decision tree": DecisionTreeClassifier(random_state=42),
        "xgbm": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42),
        "lgbm": LGBMClassifier(random_state=42),
        "KNN":  KNeighborsClassifier(n_neighbors=5)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

# Evaluate models
def evaluate_models(models, X_test, y_test, le_category):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le_category.classes_, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": cm
        }
    
    return results

# Main app
def main():
    # Instead of st.title(...)
    st.markdown('<h1 style="color:#FF4500;">🫀 Liver Disease Classification</h1>', unsafe_allow_html=True)


    st.markdown("""
    This app analyzes liver disease data and builds classification models to predict liver disease categories.
    """)
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "Exploratory Analysis", "Model Training", "Model Evaluation", "Prediction"])
    
    if page == "Data Overview":
        st.header("Data Overview")
        
        st.subheader("Dataset Sample")
        st.dataframe(df.head())
        
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Shape:**", df.shape)
            st.write("**Columns:**", list(df.columns))
        
        with col2:
            st.write("**Missing Values:**")
            missing_df = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"])
            st.dataframe(missing_df)
        
        st.subheader("Data Description")
        st.dataframe(df.describe())
        
        st.subheader("Category Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        df['category'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Category Distribution")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        
    elif page == "Exploratory Analysis":
        st.header("Exploratory Data Analysis")
        
        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        
        # Distribution of features
        st.subheader("Feature Distributions")
        selected_feature = st.selectbox("Select a feature to visualize", numeric_df.columns)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_feature], kde=True, ax=ax)
        ax.set_title(f"Distribution of {selected_feature}")
        st.pyplot(fig)
        
        # Box plots by category
        st.subheader("Feature Distribution by Category")
        feature_for_boxplot = st.selectbox("Select a feature for boxplot", numeric_df.columns)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(x='category', y=feature_for_boxplot, data=df, ax=ax)
        ax.set_title(f"{feature_for_boxplot} Distribution by Category")
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)
        
    elif page == "Model Training":
        st.header("Model Training")
        
        # Preprocess data
        X_train_scaled, X_test_scaled, y_train, y_test, le_category, scaler, feature_names = preprocess_data(df)
        
        # Train models
        if st.button("Train Models"):
            with st.spinner("Training models..."):
                models = train_models(X_train_scaled, y_train)
                results = evaluate_models(models, X_test_scaled, y_test, le_category)
                
                # Store in session state
                st.session_state.models = models
                st.session_state.results = results
                st.session_state.le_category = le_category
                st.session_state.scaler = scaler
                st.session_state.feature_names = feature_names
            
            st.success("Models trained successfully!")
            
            # Display accuracy scores
            st.subheader("Model Accuracy Scores")
            accuracy_data = []
            for model_name, result in results.items():
                accuracy_data.append({
                    "Model": model_name,
                    "Accuracy": f"{result['accuracy']:.4f}"
                })
            
            accuracy_df = pd.DataFrame(accuracy_data)
            st.dataframe(accuracy_df)
            
    elif page == "Model Evaluation":
        st.header("Model Evaluation")
        
        if "results" not in st.session_state:
            st.warning("Please train the models first on the 'Model Training' page.")
            return
            
        results = st.session_state.results
        le_category = st.session_state.le_category
        
        # Select model for detailed evaluation
        model_name = st.selectbox("Select a model for detailed evaluation", list(results.keys()))
        
        st.subheader(f"Evaluation Results for {model_name}")
        
        # Accuracy
        accuracy = results[model_name]["accuracy"]
        st.metric("Accuracy", f"{accuracy:.4f}")
        
        # Classification report
        st.subheader("Classification Report")
        report_df = pd.DataFrame(results[model_name]["report"]).transpose()
        st.dataframe(report_df)
        
        # Confusion matrix
        st.subheader("Confusion Matrix")
        cm = results[model_name]["confusion_matrix"]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=le_category.classes_, 
                    yticklabels=le_category.classes_, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix - {model_name}")
        st.pyplot(fig)
        
    elif page == "Prediction":
        st.header("Make Predictions")
        
        if "models" not in st.session_state:
            st.warning("Please train the models first on the 'Model Training' page.")
            return
            
        models = st.session_state.models
        le_category = st.session_state.le_category
        scaler = st.session_state.scaler
        feature_names = st.session_state.feature_names
        
        # Create input form
        st.subheader("Input Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=0, max_value=100, value=45)
            sex = st.selectbox("Sex", options=["m", "f"])
            albumin = st.number_input("Albumin", min_value=0.0, value=40.0)
            alkaline_phosphatase = st.number_input("Alkaline Phosphatase", min_value=0.0, value=70.0)
        
        with col2:
            alanine_aminotransferase = st.number_input("Alanine Aminotransferase", min_value=0.0, value=25.0)
            aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0.0, value=30.0)
            bilirubin = st.number_input("Bilirubin", min_value=0.0, value=10.0)
            cholinesterase = st.number_input("Cholinesterase", min_value=0.0, value=8.0)
        
        with col3:
            cholesterol = st.number_input("Cholesterol", min_value=0.0, value=5.0)
            creatinina = st.number_input("Creatinina", min_value=0.0, value=80.0)
            gamma_glutamyl_transferase = st.number_input("Gamma Glutamyl Transferase", min_value=0.0, value=40.0)
            protein = st.number_input("Protein", min_value=0.0, value=72.0)
        
        # Encode sex
        sex_encoded = 1 if sex == "m" else 0
        
        # Create feature array
        features = np.array([[age, sex_encoded, albumin, alkaline_phosphatase, 
                            alanine_aminotransferase, aspartate_aminotransferase, 
                            bilirubin, cholinesterase, cholesterol, creatinina, 
                            gamma_glutamyl_transferase, protein]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Select model for prediction
        model_name = st.selectbox("Select a model for prediction", list(models.keys()))
        model = models[model_name]
        
        if st.button("Predict"):
            # Make prediction
            prediction = model.predict(features_scaled)
            prediction_proba = model.predict_proba(features_scaled)
            
            # Get predicted class
            predicted_class = le_category.inverse_transform(prediction)[0]
            
            st.subheader("Prediction Result")
            st.success(f"The predicted category is: *{predicted_class}*")
            
            # Display prediction probabilities
            st.subheader("Prediction Probabilities")
            proba_df = pd.DataFrame({
                "Category": le_category.classes_,
                "Probability": prediction_proba[0]
            }).sort_values("Probability", ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Probability", y="Category", data=proba_df, ax=ax)
            ax.set_title("Prediction Probabilities")
            st.pyplot(fig)
            
            st.dataframe(proba_df)

if __name__ == "__main__":
    main()