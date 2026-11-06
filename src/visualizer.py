"""
Smart Student Performance & Placement Prediction System
Visualizer Module: Provides interactive Plotly charts and publication-ready figures for EDA,
model evaluation, comparisons, and individual student prediction breakdowns.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Common styling palette
COLORS = {
    'primary': '#2563EB',      # Indigo/Blue
    'secondary': '#10B981',    # Emerald/Green
    'danger': '#EF4444',       # Coral/Red
    'warning': '#F59E0B',      # Amber
    'neutral': '#64748B',      # Slate
    'placed': '#10B981',
    'not_placed': '#EF4444',
    'card_bg': '#FFFFFF',
    'plot_bg': 'rgba(248, 250, 252, 0.6)'
}

def plot_target_distribution(df: pd.DataFrame, target_col: str = 'Placement_Status') -> go.Figure:
    """Creates a donut and bar chart showing class balance."""
    counts = df[target_col].value_counts().reset_index()
    counts.columns = ['Status', 'Count']
    counts['Percentage'] = (counts['Count'] / counts['Count'].sum() * 100).round(1)
    
    fig = px.pie(
        counts,
        names='Status',
        values='Count',
        hole=0.55,
        color='Status',
        color_discrete_map={'Placed': COLORS['placed'], 'Not Placed': COLORS['not_placed']},
        title=f"Class Distribution: {target_col}"
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hoverinfo='label+value+percent',
        marker=dict(line=dict(color='#ffffff', width=2))
    )
    fig.update_layout(
        template='plotly_white',
        margin=dict(t=50, b=30, l=30, r=30),
        legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='center', x=0.5)
    )
    return fig

def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> go.Figure:
    """Generates an interactive correlation heatmap of numeric features."""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
    corr = df[numeric_cols].corr().round(2)
    
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Matrix"
    )
    fig.update_layout(
        template='plotly_white',
        margin=dict(t=50, b=40, l=40, r=40)
    )
    return fig

def plot_feature_distribution(df: pd.DataFrame, feature: str, hue: str = 'Placement_Status') -> go.Figure:
    """Plots histogram with marginal box plot, grouped by placement status."""
    fig = px.histogram(
        df,
        x=feature,
        color=hue,
        marginal="box",
        barmode="overlay",
        opacity=0.65,
        color_discrete_map={'Placed': COLORS['placed'], 'Not Placed': COLORS['not_placed']},
        title=f"Distribution of {feature} by {hue}"
    )
    fig.update_layout(
        template='plotly_white',
        margin=dict(t=50, b=40, l=40, r=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig

def plot_bivariate_scatter(df: pd.DataFrame, feat_x: str, feat_y: str, hue: str = 'Placement_Status') -> go.Figure:
    """Creates a scatter plot comparing two numerical features against placement."""
    fig = px.scatter(
        df,
        x=feat_x,
        y=feat_y,
        color=hue,
        opacity=0.75,
        color_discrete_map={'Placed': COLORS['placed'], 'Not Placed': COLORS['not_placed']},
        title=f"{feat_x} vs {feat_y} with Placement Outcome",
        hover_data=['Student_ID'] if 'Student_ID' in df.columns else None
    )
    fig.update_layout(
        template='plotly_white',
        margin=dict(t=50, b=40, l=40, r=40)
    )
    return fig

def plot_confusion_matrix(cm_data: Dict[str, Any]) -> go.Figure:
    """Renders a formatted confusion matrix heatmap with TN, FP, FN, TP annotations."""
    matrix = cm_data['matrix']
    tn, fp, fn, tp = cm_data['tn'], cm_data['fp'], cm_data['fn'], cm_data['tp']
    
    annotations_text = [
        [f"True Negative (TN)<br><b>{tn}</b>", f"False Positive (FP)<br><b>{fp}</b>"],
        [f"False Negative (FN)<br><b>{fn}</b>", f"True Positive (TP)<br><b>{tp}</b>"]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=['Predicted: Not Placed', 'Predicted: Placed'],
        y=['Actual: Not Placed', 'Actual: Placed'],
        hoverongaps=False,
        colorscale='Blues',
        showscale=False
    ))
    
    for i in range(2):
        for j in range(2):
            fig.add_annotation(
                x=['Predicted: Not Placed', 'Predicted: Placed'][j],
                y=['Actual: Not Placed', 'Actual: Placed'][i],
                text=annotations_text[i][j],
                showarrow=False,
                font=dict(size=14, color='white' if matrix[i][j] > (np.max(matrix) * 0.5) else '#1E293B')
            )
            
    fig.update_layout(
        title="Confusion Matrix",
        template='plotly_white',
        margin=dict(t=50, b=40, l=60, r=40),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    return fig

def plot_roc_curve(roc_data: Dict[str, Any], auc_score: float) -> go.Figure:
    """Renders the Receiver Operating Characteristic (ROC) curve."""
    fpr = roc_data['fpr']
    tpr = roc_data['tpr']
    
    fig = go.Figure()
    
    # Area under curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'Logistic Regression (AUC = {auc_score:.3f})',
        line=dict(color=COLORS['primary'], width=3),
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.12)'
    ))
    
    # Baseline random guessing
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Chance (AUC = 0.500)',
        line=dict(color=COLORS['neutral'], dash='dash', width=2)
    ))
    
    fig.update_layout(
        title=f"ROC Curve (AUC: {auc_score:.3f})",
        xaxis_title="False Positive Rate (1 - Specificity)",
        yaxis_title="True Positive Rate (Sensitivity / Recall)",
        template='plotly_white',
        margin=dict(t=50, b=40, l=40, r=40),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5)
    )
    return fig

def plot_pr_curve(pr_data: Dict[str, Any]) -> go.Figure:
    """Plots the Precision-Recall curve."""
    precision = pr_data['precision']
    recall = pr_data['recall']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name='Precision-Recall Curve',
        line=dict(color=COLORS['secondary'], width=3),
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.12)'
    ))
    
    fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        template='plotly_white',
        margin=dict(t=50, b=40, l=40, r=40)
    )
    return fig

def plot_threshold_sensitivity(thresh_df: pd.DataFrame, current_th: float = 0.50) -> go.Figure:
    """Renders Precision, Recall, and F1 across classification thresholds."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresh_df['Threshold'], y=thresh_df['Precision'], name='Precision', line=dict(color=COLORS['primary'], width=2.5)))
    fig.add_trace(go.Scatter(x=thresh_df['Threshold'], y=thresh_df['Recall'], name='Recall', line=dict(color=COLORS['danger'], width=2.5)))
    fig.add_trace(go.Scatter(x=thresh_df['Threshold'], y=thresh_df['F1 Score'], name='F1 Score', line=dict(color=COLORS['secondary'], width=2.5)))
    fig.add_trace(go.Scatter(x=thresh_df['Threshold'], y=thresh_df['Accuracy'], name='Accuracy', line=dict(color=COLORS['neutral'], dash='dot', width=2)))
    
    # Vertical line indicating currently active threshold
    fig.add_vline(x=current_th, line_dash='dash', line_color='#1E293B', annotation_text=f"Selected: {current_th:.2f}")
    
    fig.update_layout(
        title="Metrics vs. Classification Threshold",
        xaxis_title="Classification Threshold",
        yaxis_title="Metric Score",
        template='plotly_white',
        margin=dict(t=50, b=40, l=40, r=40),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5)
    )
    return fig

def plot_coefficients_importance(df_coef: pd.DataFrame) -> go.Figure:
    """Plots horizontal bar chart of logistic regression beta weights."""
    sorted_df = df_coef.sort_values(by='Coefficient (Beta)', ascending=True)
    
    colors = [COLORS['placed'] if val >= 0 else COLORS['not_placed'] for val in sorted_df['Coefficient (Beta)']]
    
    fig = go.Figure(go.Bar(
        x=sorted_df['Coefficient (Beta)'],
        y=sorted_df['Feature'],
        orientation='h',
        marker_color=colors,
        text=sorted_df['Coefficient (Beta)'].apply(lambda x: f"{x:+.3f}"),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Logistic Regression Coefficients (Beta Weights)",
        xaxis_title="Coefficient Value (Log-Odds Impact)",
        yaxis_title="Features",
        template='plotly_white',
        margin=dict(t=50, b=40, l=120, r=40)
    )
    return fig

def plot_model_comparison_bars(comp_df: pd.DataFrame) -> go.Figure:
    """Compares metrics between Logistic Regression, Decision Tree, and Random Forest."""
    melted = comp_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    fig = px.bar(
        melted,
        x='Metric',
        y='Score',
        color='Model',
        barmode='group',
        text='Score',
        color_discrete_sequence=[COLORS['primary'], COLORS['warning'], COLORS['secondary']],
        title="Model Performance Comparison (Test Set)"
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        yaxis_range=[0, 1.15],
        template='plotly_white',
        margin=dict(t=50, b=40, l=40, r=40),
        legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5)
    )
    return fig

def plot_prediction_gauge(prob: float, threshold: float = 0.50) -> go.Figure:
    """Displays an animated gauge meter for student placement probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Placement Probability (%)", 'font': {'size': 20, 'color': '#1E293B'}},
        delta={'reference': threshold * 100, 'increasing': {'color': COLORS['placed']}, 'decreasing': {'color': COLORS['not_placed']}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
            'bar': {'color': COLORS['primary']},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#CBD5E1",
            'steps': [
                {'range': [0, threshold * 100], 'color': 'rgba(239, 68, 68, 0.15)'},
                {'range': [threshold * 100, 100], 'color': 'rgba(16, 185, 129, 0.15)'}
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 3},
                'thickness': 0.8,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(
        template='plotly_white',
        height=280,
        margin=dict(t=40, b=10, l=20, r=20)
    )
    return fig

def plot_decision_boundary_surface(
    xx: np.ndarray,
    yy: np.ndarray,
    zz: np.ndarray,
    feat_x: str,
    feat_y: str,
    df_sample: pd.DataFrame,
    threshold: float = 0.50
) -> go.Figure:
    """Renders 2D contour surface with actual student data points."""
    fig = go.Figure()
    
    # Probability contour
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=zz,
        colorscale='Viridis',
        opacity=0.8,
        contours=dict(
            start=0,
            end=1,
            size=0.1,
            showlabels=True,
            labelfont=dict(size=10, color='white')
        ),
        colorbar=dict(title='P(Placement)'),
        name='Probability Contour'
    ))
    
    # Decision boundary line (isocline at threshold)
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=zz,
        showscale=False,
        contours=dict(
            type='constraint',
            operation='=',
            value=threshold,
            coloring='none'
        ),
        line=dict(color='red', width=3, dash='solid'),
        name=f'Decision Boundary (p={threshold:.2f})'
    ))
    
    # Scatter overlay of data points
    sample = df_sample.sample(min(250, len(df_sample)), random_state=42)
    placed_sub = sample[sample['Placement_Status'] == 'Placed']
    not_placed_sub = sample[sample['Placement_Status'] == 'Not Placed']
    
    fig.add_trace(go.Scatter(
        x=placed_sub[feat_x],
        y=placed_sub[feat_y],
        mode='markers',
        name='Actual: Placed',
        marker=dict(color=COLORS['placed'], size=8, line=dict(color='white', width=1))
    ))
    
    fig.add_trace(go.Scatter(
        x=not_placed_sub[feat_x],
        y=not_placed_sub[feat_y],
        mode='markers',
        name='Actual: Not Placed',
        marker=dict(color=COLORS['not_placed'], size=8, line=dict(color='white', width=1))
    ))
    
    fig.update_layout(
        title=f"Decision Boundary & Probability Surface: {feat_x} vs {feat_y}",
        xaxis_title=feat_x,
        yaxis_title=feat_y,
        template='plotly_white',
        margin=dict(t=50, b=40, l=40, r=40),
        legend=dict(orientation='h', yanchor='bottom', y=-0.22, xanchor='center', x=0.5)
    )
    return fig
