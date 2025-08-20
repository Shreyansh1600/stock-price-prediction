import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

# Title
st.title("📈 Stock Price Prediction Dashboard")
st.markdown("A simple machine learning model to predict stock prices using historical data")

# Sidebar for user inputs
st.sidebar.header("Data Selection")
data_source = st.sidebar.selectbox("Choose Data Source", ["Sample Dataset", "Upload CSV"], index=0)
prediction_days = st.sidebar.slider("Days to Predict", 1, 30, 7)

def load_sample_data():
    """Load sample stock data from CSV file"""
    try:
        data = pd.read_csv('sample_data.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        return data
    except FileNotFoundError:
        st.error("Sample data file not found. Please ensure 'sample_data.csv' exists.")
        return None
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

def load_uploaded_data(uploaded_file):
    """Load data from uploaded CSV file"""
    try:
        data = pd.read_csv(uploaded_file)
        
        # Try to find date column
        date_cols = [col for col in data.columns if 'date' in col.lower()]
        if date_cols:
            data[date_cols[0]] = pd.to_datetime(data[date_cols[0]])
            data.set_index(date_cols[0], inplace=True)
        else:
            st.error("No date column found. Please ensure your CSV has a 'Date' column.")
            return None
            
        # Check for required columns
        required_cols = ['Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            return None
            
        return data
    except Exception as e:
        st.error(f"Error loading uploaded data: {e}")
        return None

def prepare_features(data):
    """Create simple features for prediction"""
    df = data.copy()
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Add High_Low_Ratio if High and Low columns exist
    if 'High' in df.columns and 'Low' in df.columns:
        df['High_Low_Ratio'] = df['High'] / df['Low']
    else:
        # Create synthetic High/Low if not available
        df['High_Low_Ratio'] = 1.02  # Default ratio
    
    # Drop NaN values
    df = df.dropna()
    return df

def train_model(df):
    """Train a simple linear regression model"""
    features = ['MA_5', 'MA_20', 'Price_Change', 'Volume_Change', 'High_Low_Ratio']
    X = df[features].values
    y = df['Close'].values
    
    # Split data (80% train, 20% test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    return model, X_test, y_test, y_pred

def predict_future(model, df, days):
    """Predict future prices"""
    features = ['MA_5', 'MA_20', 'Price_Change', 'Volume_Change', 'High_Low_Ratio']
    last_features = df[features].iloc[-1:].values
    
    predictions = []
    current_price = df['Close'].iloc[-1]
    
    for _ in range(days):
        pred_price = model.predict(last_features)[0]
        predictions.append(pred_price)
        
        # Update features for next prediction (simplified)
        last_features[0][2] = (pred_price - current_price) / current_price  # Price change
        current_price = pred_price
    
    return predictions

# File upload section
uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    st.sidebar.markdown("**CSV Format Required:**")
    st.sidebar.markdown("- Date column")
    st.sidebar.markdown("- Close column (required)")
    st.sidebar.markdown("- Volume column (required)")
    st.sidebar.markdown("- High, Low columns (optional)")

# Main app logic
if st.button("Analyze Stock Data"):
    # Load data based on selection
    if data_source == "Sample Dataset":
        data = load_sample_data()
        data_name = "Sample Stock Data"
    else:
        if uploaded_file is not None:
            data = load_uploaded_data(uploaded_file)
            data_name = uploaded_file.name
        else:
            st.error("Please upload a CSV file.")
            st.stop()
    
    if data is not None and not data.empty:
        # Prepare features
        df = prepare_features(data)
        
        if len(df) > 50:  # Ensure enough data
            # Train model
            model, X_test, y_test, y_pred = train_model(df)
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Stock Price History")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    mode='lines',
                    name='Close Price',
                    line=dict(color='blue')
                ))
                fig.update_layout(
                    title=f"{data_name} - Stock Price",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Show recent prices
                st.subheader("📈 Recent Prices")
                recent_data = data.tail(5)[['Close', 'Volume']]
                st.dataframe(recent_data)
            
            # Predictions
            st.subheader(f"🔮 Price Predictions (Next {prediction_days} days)")
            predictions = predict_future(model, df, prediction_days)
            
            # Create prediction dataframe
            future_dates = [data.index[-1] + timedelta(days=i+1) for i in range(prediction_days)]
            pred_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': predictions
            })
            
            # Display predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(pred_df)
            
            with col2:
                # Plot predictions
                fig_pred = go.Figure()
                
                # Historical prices (last 30 days)
                recent_data = data.tail(30)
                fig_pred.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_data['Close'],
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Predictions
                fig_pred.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions,
                    mode='lines+markers',
                    name='Predictions',
                    line=dict(color='red', dash='dash')
                ))
                
                fig_pred.update_layout(
                    title="Price Prediction",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    height=400
                )
                st.plotly_chart(fig_pred, use_container_width=True)
            
            # Summary
            current_price = data['Close'].iloc[-1]
            avg_prediction = np.mean(predictions)
            price_change = ((avg_prediction - current_price) / current_price) * 100
            
            st.subheader("📋 Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Current Price", f"₹{current_price:.2f}")
            
            with col2:
                st.metric("Avg Predicted Price", f"₹{avg_prediction:.2f}")
            
            with col3:
                st.metric("Expected Change", f"{price_change:+.1f}%")
            
            # Data info
            st.subheader("📊 Dataset Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(data))
            
            with col2:
                st.metric("Date Range", f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
            
        else:
            st.error("Not enough data to train the model. Need at least 50 records.")
    else:
        st.error("No data available. Please check your data source.")

# Footer
st.markdown("---")
st.markdown("**NOTE**: A learning model desinged in LMNIIT")
st.markdown("**Data Source**: Historical stock data from CSV files")
