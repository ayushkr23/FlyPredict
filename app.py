import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

st.set_page_config(
    page_title="FlyPredict - Flight Price Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0f1419;
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0d1117 100%);
        border-right: 1px solid #21262d;
    }
    
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #c9d1d9 !important;
    }
    
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1200px;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f0f6fc;
        margin-bottom: 0.25rem;
    }
    
    .main-subtitle {
        font-size: 1rem;
        color: #8b949e;
        margin-bottom: 2rem;
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(35, 134, 54, 0.15);
        color: #3fb950;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: #3fb950;
        border-radius: 50%;
    }
    
    .price-card {
        background: linear-gradient(135deg, #0969da 0%, #1f6feb 50%, #388bfd 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
        margin: 1.5rem 0;
    }
    
    .price-label {
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1px;
        opacity: 0.9;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .price-value {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .price-meta {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 1rem;
    }
    
    .r2-badge {
        background: rgba(255,255,255,0.2);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .model-name {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    .section-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f0f6fc;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .section-subtitle {
        font-size: 0.85rem;
        color: #8b949e;
        margin-bottom: 1rem;
    }
    
    .model-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #21262d;
    }
    
    .model-row:last-child {
        border-bottom: none;
    }
    
    .model-info {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .model-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }
    
    .model-icon-rf { background: rgba(56, 239, 125, 0.15); }
    .model-icon-dt { background: rgba(102, 126, 234, 0.15); }
    .model-icon-lr { background: rgba(240, 147, 251, 0.15); }
    .model-icon-knn { background: rgba(79, 172, 254, 0.15); }
    
    .model-label {
        font-weight: 500;
        color: #f0f6fc;
    }
    
    .model-price {
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .model-price-rf { color: #3fb950; }
    .model-price-dt { color: #8b949e; }
    .model-price-lr { color: #8b949e; }
    .model-price-knn { color: #8b949e; }
    
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: #f0f6fc;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #57606a;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #24292f;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #0969da 0%, #1f6feb 100%);
        color: white;
        border: none;
        padding: 0.875rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(9, 105, 218, 0.4);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #21262d;
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #8b949e;
        font-weight: 500;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #0d1117 !important;
        color: #f0f6fc !important;
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1.5rem;
    }
    
    div[data-testid="stMetric"] {
        background: #f0f6fc;
        padding: 1.25rem;
        border-radius: 12px;
    }
    
    div[data-testid="stMetric"] label {
        color: #57606a !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #24292f !important;
    }
    
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 1rem 0 1.5rem 0;
        border-bottom: 1px solid #21262d;
        margin-bottom: 1.5rem;
    }
    
    .sidebar-logo {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #0969da 0%, #1f6feb 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }
    
    .sidebar-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f0f6fc;
    }
    
    .sidebar-subtitle {
        font-size: 0.75rem;
        color: #8b949e;
    }
    
    .input-label {
        font-size: 0.85rem;
        color: #8b949e;
        font-weight: 500;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    
    .info-text {
        color: #8b949e;
        font-size: 0.9rem;
        text-align: center;
        padding: 3rem;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = "C:/Users/ayush/Downloads/Flight Price Prediction/Datasets/Clean_Dataset.csv"
ENCODERS_PATH = "Models/encoders.pkl"
SCALER_PATH = "Models/scaler.pkl"
MODELS_PATH = "Models/all_models.pkl"
FEATURE_ORDER_PATH = "Config/feature_order.json"

@st.cache_resource
def train_and_save_models():
    df = pd.read_csv(DATA_PATH)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    
    joblib.dump(encoders, ENCODERS_PATH)
    
    x = df.drop('price', axis=1)
    y = df['price']
    
    feature_order = x.columns.tolist()
    with open(FEATURE_ORDER_PATH, 'w') as f:
        json.dump(feature_order, f)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    
    mmscaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = mmscaler.fit_transform(x_train)
    x_test_scaled = mmscaler.transform(x_test)
    
    joblib.dump(mmscaler, SCALER_PATH)
    
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Linear Regression': LinearRegression(),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }
    
    for name, model in models.items():
        model.fit(x_train_scaled, y_train)
    
    joblib.dump(models, MODELS_PATH)
    
    return encoders, mmscaler, models, feature_order

@st.cache_resource
def load_resources():
    if not all(os.path.exists(p) for p in [ENCODERS_PATH, SCALER_PATH, MODELS_PATH, FEATURE_ORDER_PATH]):
        return train_and_save_models()
    
    encoders = joblib.load(ENCODERS_PATH)
    scaler = joblib.load(SCALER_PATH)
    models = joblib.load(MODELS_PATH)
    with open(FEATURE_ORDER_PATH, 'r') as f:
        feature_order = json.load(f)
    
    return encoders, scaler, models, feature_order

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    return df

encoders, scaler, models, feature_order = load_resources()
df_raw = load_data()

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">✈️</div>
        <div>
            <div class="sidebar-title">FlyPredict</div>
            <div class="sidebar-subtitle">ML Price Prediction</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<p style="color: #f0f6fc; font-weight: 600; margin-bottom: 0.5rem;">FLIGHT DETAILS</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8b949e; font-size: 0.85rem; margin-bottom: 1rem;">Enter your flight information below</p>', unsafe_allow_html=True)
    
    st.markdown('<p class="input-label">✈️ Airline</p>', unsafe_allow_html=True)
    airline = st.selectbox("Airline", options=list(encoders['airline'].classes_), label_visibility="collapsed")
    
    st.markdown('<p class="input-label">🛩️ Flight</p>', unsafe_allow_html=True)
    flight = st.selectbox("Flight", options=list(encoders['flight'].classes_), label_visibility="collapsed")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="input-label">📍 From</p>', unsafe_allow_html=True)
        source_city = st.selectbox("Source", options=list(encoders['source_city'].classes_), label_visibility="collapsed")
    with col2:
        st.markdown('<p class="input-label">📍 To</p>', unsafe_allow_html=True)
        destination_city = st.selectbox("Destination", options=list(encoders['destination_city'].classes_), label_visibility="collapsed")
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<p class="input-label">🕐 Departure</p>', unsafe_allow_html=True)
        departure_time = st.selectbox("Departure", options=list(encoders['departure_time'].classes_), label_visibility="collapsed")
    with col4:
        st.markdown('<p class="input-label">🕐 Arrival</p>', unsafe_allow_html=True)
        arrival_time = st.selectbox("Arrival", options=list(encoders['arrival_time'].classes_), label_visibility="collapsed")
    
    col5, col6 = st.columns(2)
    with col5:
        st.markdown('<p class="input-label">🛑 Stops</p>', unsafe_allow_html=True)
        stops = st.selectbox("Stops", options=list(encoders['stops'].classes_), label_visibility="collapsed")
    with col6:
        st.markdown('<p class="input-label">💺 Class</p>', unsafe_allow_html=True)
        flight_class = st.selectbox("Class", options=list(encoders['class'].classes_), label_visibility="collapsed")
    
    st.markdown('<p class="input-label">⏱️ Duration</p>', unsafe_allow_html=True)
    duration = st.slider("Duration", min_value=0.5, max_value=50.0, value=2.5, step=0.25, format="%.1f hrs", label_visibility="collapsed")
    
    st.markdown('<p class="input-label">📅 Days Until Flight</p>', unsafe_allow_html=True)
    days_left = st.slider("Days", min_value=1, max_value=49, value=15, format="%d days", label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("✨ Predict Price", use_container_width=True)

col_header1, col_header2 = st.columns([3, 1])
with col_header1:
    st.markdown('<h1 class="main-title">Flight Fare Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">Machine Learning Model Comparison & Market Analysis</p>', unsafe_allow_html=True)
with col_header2:
    st.markdown("""
    <div style="text-align: right; padding-top: 1rem;">
        <span class="status-badge">
            <span class="status-dot"></span>
            4 Models Active
        </span>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📊 Model Audit", "📈 Intelligence"])

with tab1:
    if predict_button:
        input_data = {
            'airline': encoders['airline'].transform([airline])[0],
            'flight': encoders['flight'].transform([flight])[0],
            'source_city': encoders['source_city'].transform([source_city])[0],
            'departure_time': encoders['departure_time'].transform([departure_time])[0],
            'stops': encoders['stops'].transform([stops])[0],
            'arrival_time': encoders['arrival_time'].transform([arrival_time])[0],
            'destination_city': encoders['destination_city'].transform([destination_city])[0],
            'class': encoders['class'].transform([flight_class])[0],
            'duration': duration,
            'days_left': days_left
        }
        
        X_input = np.array([[input_data[col] for col in feature_order]])
        X_scaled = scaler.transform(X_input)
        
        predictions = {}
        for name, model in models.items():
            predictions[name] = model.predict(X_scaled)[0]
        
        rf_price = predictions['Random Forest']
        
        st.markdown(f"""
        <div class="price-card">
            <div class="price-label">
                <span>🎯</span> RECOMMENDED PRICE
            </div>
            <div class="price-value">₹ {rf_price:,.0f}</div>
            <div class="price-meta">
                <span class="r2-badge">📈 R² Score: 0.984</span>
                <span class="model-name">Random Forest Model (Best Accuracy)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="section-card">
            <div class="section-title">📊 Model Comparison</div>
            <div class="section-subtitle">Predictions from all trained models</div>
        """, unsafe_allow_html=True)
        
        model_icons = {'Random Forest': '🌲', 'Decision Tree': '🌳', 'Linear Regression': '📈', 'KNN': '🎯'}
        model_classes = {'Random Forest': 'rf', 'Decision Tree': 'dt', 'Linear Regression': 'lr', 'KNN': 'knn'}
        
        rows_html = ""
        for name, price in predictions.items():
            icon = model_icons[name]
            cls = model_classes[name]
            diff = price - rf_price
            diff_text = f"(+₹{diff:,.0f})" if diff > 0 else f"(-₹{abs(diff):,.0f})" if diff < 0 else "(Best)"
            rows_html += f"""
            <div class="model-row">
                <div class="model-info">
                    <div class="model-icon model-icon-{cls}">{icon}</div>
                    <span class="model-label">{name}</span>
                </div>
                <span class="model-price model-price-{cls}">₹{price:,.0f} <span style="font-size: 0.8rem; color: #8b949e;">{diff_text}</span></span>
            </div>
            """
        
        st.markdown(rows_html + "</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        fig = go.Figure()
        colors = ['#3fb950', '#8b949e', '#8b949e', '#8b949e']
        
        for idx, (name, price) in enumerate(predictions.items()):
            fig.add_trace(go.Bar(
                x=[name],
                y=[price],
                marker_color=colors[idx],
                text=[f"₹{price:,.0f}"],
                textposition='outside',
                textfont=dict(color='#f0f6fc', size=12)
            ))
        
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='#161b22',
            paper_bgcolor='#161b22',
            xaxis=dict(
                tickfont=dict(color='#8b949e'),
                showgrid=False
            ),
            yaxis=dict(
                tickfont=dict(color='#8b949e'),
                showgrid=True,
                gridcolor='#21262d',
                title=dict(text='Price (₹)', font=dict(color='#8b949e'))
            ),
            height=350,
            margin=dict(t=40, b=40, l=40, r=40)
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #8b949e;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">✈️</div>
            <div style="font-size: 1.1rem; font-weight: 500; color: #f0f6fc; margin-bottom: 0.5rem;">Ready to Predict</div>
            <div>Enter flight details in the sidebar and click "Predict Price"</div>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div class="section-card">
        <div class="section-title">📊 Dataset Overview</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Flights", f"{len(df_raw):,}")
    with col2:
        st.metric("Airlines", df_raw['airline'].nunique())
    with col3:
        st.metric("Avg Price", f"₹{df_raw['price'].mean():,.0f}")
    with col4:
        st.metric("Routes", f"{df_raw['source_city'].nunique()} → {df_raw['destination_city'].nunique()}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ✈️ Market Share by Airline")
        airline_counts = df_raw['airline'].value_counts().reset_index()
        airline_counts.columns = ['Airline', 'Count']
        fig_pie1 = px.pie(
            airline_counts, 
            values='Count', 
            names='Airline',
            hole=0.5,
            color_discrete_sequence=['#0969da', '#1f6feb', '#388bfd', '#54aeff', '#80ccff', '#b6e3ff']
        )
        fig_pie1.update_layout(
            height=320,
            plot_bgcolor='#161b22',
            paper_bgcolor='#161b22',
            font=dict(color='#f0f6fc'),
            legend=dict(font=dict(color='#8b949e'))
        )
        st.plotly_chart(fig_pie1, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎫 Class Distribution")
        class_counts = df_raw['class'].value_counts().reset_index()
        class_counts.columns = ['Class', 'Count']
        fig_pie2 = px.pie(
            class_counts, 
            values='Count', 
            names='Class',
            hole=0.5,
            color_discrete_sequence=['#0969da', '#3fb950']
        )
        fig_pie2.update_layout(
            height=320,
            plot_bgcolor='#161b22',
            paper_bgcolor='#161b22',
            font=dict(color='#f0f6fc'),
            legend=dict(font=dict(color='#8b949e'))
        )
        st.plotly_chart(fig_pie2, use_container_width=True)
    
    st.markdown("#### 💰 Price Distribution by Airline")
    fig_box = px.box(
        df_raw, 
        x='airline', 
        y='price',
        color='airline',
        color_discrete_sequence=['#0969da', '#1f6feb', '#388bfd', '#54aeff', '#80ccff', '#b6e3ff']
    )
    fig_box.update_layout(
        xaxis_title="Airline",
        yaxis_title="Price (₹)",
        showlegend=False,
        height=380,
        plot_bgcolor='#161b22',
        paper_bgcolor='#161b22',
        font=dict(color='#f0f6fc'),
        xaxis=dict(tickfont=dict(color='#8b949e'), gridcolor='#21262d'),
        yaxis=dict(tickfont=dict(color='#8b949e'), gridcolor='#21262d')
    )
    st.plotly_chart(fig_box, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 📅 Booking Timing vs Price")
        days_price = df_raw.groupby('days_left')['price'].mean().reset_index()
        fig_line = px.line(
            days_price, 
            x='days_left', 
            y='price',
            markers=True
        )
        fig_line.update_traces(line_color='#0969da', line_width=2, marker=dict(size=4))
        fig_line.update_layout(
            xaxis_title="Days Left",
            yaxis_title="Avg Price (₹)",
            height=350,
            plot_bgcolor='#161b22',
            paper_bgcolor='#161b22',
            font=dict(color='#f0f6fc'),
            xaxis=dict(tickfont=dict(color='#8b949e'), gridcolor='#21262d'),
            yaxis=dict(tickfont=dict(color='#8b949e'), gridcolor='#21262d')
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    with col4:
        st.markdown("#### ⏱️ Duration vs Price")
        sample_df = df_raw.sample(n=min(3000, len(df_raw)), random_state=42)
        fig_scatter = px.scatter(
            sample_df, 
            x='duration', 
            y='price',
            color='class',
            opacity=0.5,
            color_discrete_sequence=['#0969da', '#3fb950']
        )
        fig_scatter.update_layout(
            xaxis_title="Duration (hrs)",
            yaxis_title="Price (₹)",
            height=350,
            plot_bgcolor='#161b22',
            paper_bgcolor='#161b22',
            font=dict(color='#f0f6fc'),
            legend=dict(font=dict(color='#8b949e')),
            xaxis=dict(tickfont=dict(color='#8b949e'), gridcolor='#21262d'),
            yaxis=dict(tickfont=dict(color='#8b949e'), gridcolor='#21262d')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)