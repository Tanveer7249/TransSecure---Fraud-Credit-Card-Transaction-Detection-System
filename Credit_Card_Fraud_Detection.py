"""
Credit Card Fraud Detection with Modern GUI
Complete implementation with data processing, model training, and interactive interface
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_auc_score, roc_curve, f1_score, accuracy_score)
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import io
import base64
from PIL import Image

warnings.filterwarnings('ignore')

class FraudDetectionSystem:
    """Complete Fraud Detection System"""
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.pt = PowerTransformer(method='yeo-johnson', standardize=True)
        self.models = {}
        self.results = {}
        
    def load_data(self, file_path):
        """Load and initial data exploration"""
        try:
            self.df = pd.read_csv(file_path)
            return True, f"Data loaded successfully! Shape: {self.df.shape}"
        except Exception as e:
            return False, f"Error loading data: {str(e)}"
    
    def get_data_overview(self):
        """Get comprehensive data overview"""
        if self.df is None:
            return "No data loaded"
        
        # Basic info
        info = f"""
        📊 **Dataset Overview**
        - Total Transactions: {len(self.df):,}
        - Features: {self.df.shape[1]}
        - Missing Values: {self.df.isnull().sum().sum()}
        
        📈 **Class Distribution**
        - Normal Transactions: {len(self.df[self.df['Class']==0]):,} ({len(self.df[self.df['Class']==0])/len(self.df)*100:.2f}%)
        - Fraudulent Transactions: {len(self.df[self.df['Class']==1]):,} ({len(self.df[self.df['Class']==1])/len(self.df)*100:.2f}%)
        
        💰 **Transaction Amount Statistics**
        - Mean: ${self.df['Amount'].mean():.2f}
        - Median: ${self.df['Amount'].median():.2f}
        - Max: ${self.df['Amount'].max():.2f}
        """
        return info
    
    def create_class_distribution_plot(self):
        """Create interactive class distribution plot"""
        if self.df is None:
            return None
        
        class_counts = self.df['Class'].value_counts()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Transaction Count', 'Percentage Distribution'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(x=['Normal', 'Fraud'], 
                   y=[class_counts[0], class_counts[1]],
                   marker_color=['#2ecc71', '#e74c3c'],
                   text=[f'{class_counts[0]:,}', f'{class_counts[1]:,}'],
                   textposition='auto'),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(labels=['Normal', 'Fraud'],
                   values=[class_counts[0], class_counts[1]],
                   marker_colors=['#2ecc71', '#e74c3c']),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Class Distribution Analysis",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def create_amount_distribution_plot(self):
        """Create amount distribution plot"""
        if self.df is None:
            return None
        
        fraud = self.df[self.df['Class'] == 1]['Amount']
        normal = self.df[self.df['Class'] == 0]['Amount']
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=normal,
            name='Normal',
            marker_color='#2ecc71',
            opacity=0.7,
            nbinsx=50
        ))
        
        fig.add_trace(go.Histogram(
            x=fraud,
            name='Fraud',
            marker_color='#e74c3c',
            opacity=0.7,
            nbinsx=50
        ))
        
        fig.update_layout(
            title='Transaction Amount Distribution by Class',
            xaxis_title='Amount ($)',
            yaxis_title='Count',
            barmode='overlay',
            height=400
        )
        
        return fig
    
    def preprocess_data(self, sampling_method='SMOTE', test_size=0.2):
        """Preprocess data with sampling"""
        if self.df is None:
            return "No data loaded"
        
        # Drop Time column
        df_processed = self.df.drop('Time', axis=1) if 'Time' in self.df.columns else self.df.copy()
        
        # Split features and target
        X = df_processed.drop(['Class'], axis=1)
        y = df_processed['Class']
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=1-test_size, test_size=test_size, random_state=100
        )
        
        # Scale Amount
        self.X_train['Amount'] = self.scaler.fit_transform(self.X_train[['Amount']])
        self.X_test['Amount'] = self.scaler.transform(self.X_test[['Amount']])
        
        # Apply PowerTransformer
        cols = self.X_train.columns
        self.X_train[cols] = self.pt.fit_transform(self.X_train)
        self.X_test[cols] = self.pt.transform(self.X_test)
        
        # Apply sampling
        if sampling_method == 'SMOTE':
            sampler = SMOTE(random_state=27)
        elif sampling_method == 'RandomOverSampler':
            sampler = RandomOverSampler(random_state=27)
        elif sampling_method == 'ADASYN':
            sampler = ADASYN(random_state=27)
        elif sampling_method == 'RandomUnderSampler':
            sampler = RandomUnderSampler(random_state=27)
        else:
            return f"""
            ✅ **Data Preprocessed Successfully**
            - Training samples: {len(self.X_train):,}
            - Test samples: {len(self.X_test):,}
            - No resampling applied
            """
        
        self.X_train, self.y_train = sampler.fit_resample(self.X_train, self.y_train)
        
        return f"""
        ✅ **Data Preprocessed Successfully**
        - Training samples: {len(self.X_train):,}
        - Test samples: {len(self.X_test):,}
        - Sampling method: {sampling_method}
        - Train fraud ratio: {sum(self.y_train)/len(self.y_train)*100:.2f}%
        """
    
    def train_model(self, model_name='Logistic Regression', progress=gr.Progress()):
        """Train selected model"""
        if self.X_train is None:
            return "Please preprocess data first", None
        
        progress(0, desc="Starting training...")
        
        try:
            if model_name == 'Logistic Regression':
                progress(0.3, desc="Training Logistic Regression...")
                params = {"C": [0.01, 0.1, 1, 10]}
                model = LogisticRegression(max_iter=1000)
                
            elif model_name == 'XGBoost':
                progress(0.3, desc="Training XGBoost...")
                params = {
                    'learning_rate': [0.2, 0.6],
                    'subsample': [0.6, 0.9]
                }
                model = XGBClassifier(max_depth=2, n_estimators=200, random_state=100)
                
            elif model_name == 'Decision Tree':
                progress(0.3, desc="Training Decision Tree...")
                params = {
                    'max_depth': [5, 10],
                    'min_samples_leaf': [50, 100]
                }
                model = DecisionTreeClassifier(random_state=100)
                
            elif model_name == 'Random Forest':
                progress(0.3, desc="Training Random Forest...")
                params = {
                    'max_depth': [5],
                    'n_estimators': [100, 200],
                    'max_features': [10]
                }
                model = RandomForestClassifier(random_state=100)
            
            progress(0.5, desc="Performing GridSearch...")
            
            # GridSearch with fewer folds for speed
            folds = KFold(n_splits=3, shuffle=True, random_state=4)
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params,
                scoring='roc_auc',
                cv=folds,
                verbose=0,
                n_jobs=-1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            progress(0.8, desc="Evaluating model...")
            
            # Best model
            best_model = grid_search.best_estimator_
            self.models[model_name] = best_model
            
            # Predictions
            y_train_pred = best_model.predict(self.X_train)
            y_test_pred = best_model.predict(self.X_test)
            
            y_train_proba = best_model.predict_proba(self.X_train)[:, 1]
            y_test_proba = best_model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            train_metrics = self._calculate_metrics(self.y_train, y_train_pred, y_train_proba)
            test_metrics = self._calculate_metrics(self.y_test, y_test_pred, y_test_proba)
            
            self.results[model_name] = {
                'train': train_metrics,
                'test': test_metrics,
                'best_params': grid_search.best_params_,
                'y_test_proba': y_test_proba
            }
            
            progress(1.0, desc="Complete!")
            
            # Create results visualization
            results_fig = self._create_results_visualization(model_name)
            
            report = f"""
            🎯 **{model_name} Training Complete**
            
            **Best Parameters:** {grid_search.best_params_}
            
            **📊 Training Set Performance:**
            - Accuracy: {train_metrics['accuracy']:.4f}
            - Precision: {train_metrics['precision']:.4f}
            - Recall (Sensitivity): {train_metrics['recall']:.4f}
            - F1-Score: {train_metrics['f1']:.4f}
            - ROC-AUC: {train_metrics['roc_auc']:.4f}
            
            **📊 Test Set Performance:**
            - Accuracy: {test_metrics['accuracy']:.4f}
            - Precision: {test_metrics['precision']:.4f}
            - Recall (Sensitivity): {test_metrics['recall']:.4f}
            - F1-Score: {test_metrics['f1']:.4f}
            - ROC-AUC: {test_metrics['roc_auc']:.4f}
            
            **🎯 Confusion Matrix (Test Set):**
            {test_metrics['confusion_matrix']}
            """
            
            return report, results_fig
            
        except Exception as e:
            return f"Error training model: {str(e)}", None
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate all metrics"""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': cm
        }
    
    def _create_results_visualization(self, model_name):
        """Create comprehensive results visualization"""
        if model_name not in self.results:
            return None
        
        test_metrics = self.results[model_name]['test']
        y_test_proba = self.results[model_name]['y_test_proba']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROC Curve', 'Confusion Matrix', 
                          'Metrics Comparison', 'Probability Distribution'),
            specs=[[{"type": "scatter"}, {"type": "heatmap"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_test_proba)
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC',
                      line=dict(color='#3498db', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random',
                      line=dict(color='gray', dash='dash')),
            row=1, col=1
        )
        
        # Confusion Matrix
        cm = test_metrics['confusion_matrix']
        fig.add_trace(
            go.Heatmap(z=cm, x=['Predicted Normal', 'Predicted Fraud'],
                      y=['Actual Normal', 'Actual Fraud'],
                      colorscale='RdYlGn_r', text=cm, texttemplate='%{text}',
                      showscale=False),
            row=1, col=2
        )
        
        # Metrics Bar Chart
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metrics_values = [test_metrics['accuracy'], test_metrics['precision'],
                         test_metrics['recall'], test_metrics['f1'], 
                         test_metrics['roc_auc']]
        
        fig.add_trace(
            go.Bar(x=metrics_names, y=metrics_values,
                  marker_color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'],
                  text=[f'{v:.3f}' for v in metrics_values],
                  textposition='auto'),
            row=2, col=1
        )
        
        # Probability Distribution
        fig.add_trace(
            go.Histogram(x=y_test_proba[self.y_test == 0], name='Normal',
                        marker_color='#2ecc71', opacity=0.7, nbinsx=50),
            row=2, col=2
        )
        fig.add_trace(
            go.Histogram(x=y_test_proba[self.y_test == 1], name='Fraud',
                        marker_color='#e74c3c', opacity=0.7, nbinsx=50),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text=f"{model_name} - Performance Analysis",
            showlegend=True,
            height=800
        )
        
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Metrics", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        fig.update_xaxes(title_text="Predicted Probability", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        return fig
    
    def compare_models(self):
        """Compare all trained models"""
        if not self.results:
            return "No models trained yet", None
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, results in self.results.items():
            test_metrics = results['test']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': test_metrics['accuracy'],
                'Precision': test_metrics['precision'],
                'Recall': test_metrics['recall'],
                'F1-Score': test_metrics['f1'],
                'ROC-AUC': test_metrics['roc_auc']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create comparison plot
        fig = go.Figure()
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric,
                x=df_comparison['Model'],
                y=df_comparison[metric],
                marker_color=colors[i],
                text=df_comparison[metric].round(3),
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Comparison - Test Set Performance',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=500
        )
        
        # Create summary text
        best_model = df_comparison.loc[df_comparison['ROC-AUC'].idxmax(), 'Model']
        best_roc = df_comparison.loc[df_comparison['ROC-AUC'].idxmax(), 'ROC-AUC']
        
        summary = f"""
        🏆 **Model Comparison Summary**
        
        **Best Model (by ROC-AUC):** {best_model} with ROC-AUC = {best_roc:.4f}
        
        **Detailed Comparison:**
        {df_comparison.to_string(index=False)}
        
        **💡 Recommendation:**
        - For **high-value transactions**: Prioritize models with high Recall to catch all frauds
        - For **low-value transactions**: Prioritize models with high Precision to reduce false alarms
        - For **balanced approach**: Choose {best_model} with best overall ROC-AUC score
        """
        
        return summary, fig
    
    def predict_transaction(self, amount, *features):
        """Predict if a transaction is fraudulent"""
        if not self.models:
            return "No models trained yet"
        
        # Use the first trained model
        model_name = list(self.models.keys())[0]
        model = self.models[model_name]
        
        # Create feature array
        feature_array = np.array([amount] + list(features)).reshape(1, -1)
        
        # Scale
        feature_array_scaled = feature_array.copy()
        feature_array_scaled[0, 0] = self.scaler.transform([[amount]])[0, 0]
        
        # Predict
        prediction = model.predict(feature_array_scaled)[0]
        probability = model.predict_proba(feature_array_scaled)[0, 1]
        
        result = f"""
        🔍 **Prediction Result**
        
        **Transaction Status:** {"🚨 FRAUDULENT" if prediction == 1 else "✅ LEGITIMATE"}
        **Fraud Probability:** {probability*100:.2f}%
        **Model Used:** {model_name}
        
        **Risk Level:** {"HIGH RISK" if probability > 0.7 else "MEDIUM RISK" if probability > 0.3 else "LOW RISK"}
        
        **Recommendation:** {"Block transaction and contact customer" if probability > 0.7 else "Review transaction manually" if probability > 0.3 else "Proceed with transaction"}
        """
        
        return result

# Initialize system
fraud_system = FraudDetectionSystem()

# Create Gradio Interface
def create_interface():
    """Create modern Gradio interface"""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="Credit Card Fraud Detection") as app:
        gr.Markdown("""
        # 🛡️ Credit Card Fraud Detection System
        ### Advanced Machine Learning Platform for Financial Security
        """)
        
        with gr.Tabs():
            # Tab 1: Data Upload and Overview
            with gr.Tab("📁 Data Upload & Overview"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(label="Upload Credit Card Dataset (CSV)", file_types=[".csv"])
                        load_btn = gr.Button("Load Data", variant="primary", size="lg")
                        load_status = gr.Textbox(label="Status", lines=2)
                    
                    with gr.Column(scale=1):
                        data_overview = gr.Textbox(label="Dataset Overview", lines=10)
                
                with gr.Row():
                    class_dist_plot = gr.Plot(label="Class Distribution")
                    amount_dist_plot = gr.Plot(label="Amount Distribution")
                
                load_btn.click(
                    fn=lambda f: fraud_system.load_data(f.name) if f else (False, "No file uploaded"),
                    inputs=[file_input],
                    outputs=[load_status]
                ).then(
                    fn=lambda: fraud_system.get_data_overview(),
                    outputs=[data_overview]
                ).then(
                    fn=lambda: fraud_system.create_class_distribution_plot(),
                    outputs=[class_dist_plot]
                ).then(
                    fn=lambda: fraud_system.create_amount_distribution_plot(),
                    outputs=[amount_dist_plot]
                )
            
            # Tab 2: Data Preprocessing
            with gr.Tab("⚙️ Data Preprocessing"):
                with gr.Row():
                    with gr.Column():
                        sampling_method = gr.Dropdown(
                            choices=['None', 'SMOTE', 'RandomOverSampler', 'ADASYN', 'RandomUnderSampler'],
                            value='SMOTE',
                            label="Sampling Method"
                        )
                        test_size = gr.Slider(minimum=0.1, maximum=0.4, value=0.2, step=0.05, 
                                            label="Test Set Size")
                        preprocess_btn = gr.Button("Preprocess Data", variant="primary", size="lg")
                    
                    with gr.Column():
                        preprocess_status = gr.Textbox(label="Preprocessing Results", lines=10)
                
                gr.Markdown("""
                **Sampling Methods:**
                - **None**: Use original imbalanced data
                - **SMOTE**: Synthetic Minority Over-sampling Technique (Recommended)
                - **RandomOverSampler**: Random duplication of minority class
                - **ADASYN**: Adaptive Synthetic Sampling
                - **RandomUnderSampler**: Random reduction of majority class
                """)
                
                preprocess_btn.click(
                    fn=fraud_system.preprocess_data,
                    inputs=[sampling_method, test_size],
                    outputs=[preprocess_status]
                )
            
            # Tab 3: Model Training
            with gr.Tab("🤖 Model Training"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_choice = gr.Dropdown(
                            choices=['Logistic Regression', 'XGBoost', 'Decision Tree', 'Random Forest'],
                            value='Logistic Regression',
                            label="Select Model"
                        )
                        train_btn = gr.Button("Train Model", variant="primary", size="lg")
                        
                        gr.Markdown("""
                        **Model Descriptions:**
                        - **Logistic Regression**: Fast, interpretable, good baseline
                        - **XGBoost**: High performance, handles imbalance well
                        - **Decision Tree**: Interpretable, fast training
                        - **Random Forest**: Robust, reduces overfitting
                        """)
                    
                    with gr.Column(scale=2):
                        training_results = gr.Textbox(label="Training Results", lines=20)
                
                results_plot = gr.Plot(label="Model Performance Analysis")
                
                train_btn.click(
                    fn=fraud_system.train_model,
                    inputs=[model_choice],
                    outputs=[training_results, results_plot]
                )
            
            # Tab 4: Model Comparison
            with gr.Tab("📊 Model Comparison"):
                compare_btn = gr.Button("Compare All Models", variant="primary", size="lg")
                
                with gr.Row():
                    comparison_text = gr.Textbox(label="Comparison Summary", lines=15)
                
                comparison_plot = gr.Plot(label="Model Comparison Chart")
                
                compare_btn.click(
                    fn=fraud_system.compare_models,
                    outputs=[comparison_text, comparison_plot]
                )
            
            # Tab 5: Prediction
            with gr.Tab("🔮 Make Prediction"):
                gr.Markdown("### Test Individual Transactions")
                
                with gr.Row():
                    with gr.Column():
                        pred_amount = gr.Number(label="Transaction Amount", value=100.0)
                        
                        # Simplified feature inputs (using V1-V10 for demo)
                        feature_inputs = []
                        for i in range(1, 11):
                            feature_inputs.append(gr.Number(label=f"Feature V{i}", value=0.0))
                        
                        predict_btn = gr.Button("Predict Transaction", variant="primary", size="lg")
                    
                    with gr.Column():
                        prediction_result = gr.Textbox(label="Prediction Result", lines=15)
                
                predict_btn.click(
                    fn=fraud_system.predict_transaction,
                    inputs=[pred_amount] + feature_inputs,
                    outputs=[prediction_result]
                )
        
        gr.Markdown("""
        ---
        ### 💡 Usage Instructions:
        1. **Upload Data**: Load your creditcard.csv file in the first tab
        2. **Preprocess**: Choose sampling method and preprocess the data
        3. **Train Models**: Select and train different models
        4. **Compare**: Compare all trained models to find the best one
        5. **Predict**: Test individual transactions for fraud detection
        
        ### 📝 Notes:
        - Dataset should contain PCA-transformed features (V1-V28), Amount, and Class columns
        - SMOTE is recommended for handling class imbalance
        - ROC-AUC is the primary metric for model selection
        - Train multiple models for comprehensive comparison
        """)
    
    return app

# Launch the application
if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True, server_name="0.0.0.0", server_port=7860)
