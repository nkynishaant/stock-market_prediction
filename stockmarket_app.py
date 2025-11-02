import streamlit as st
import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# --- Configuration and Setup ---
st.set_page_config(layout="wide", page_title="S&P 500 Tomorrow's Trend Predictor")
st.title("S&P 500 Tomorrow's Trend Predictor 📈")
st.markdown("A Machine Learning model to predict if the S&P 500 index (^GSPC) will go up (1) or down/stay flat (0) tomorrow.")

# --- Data Loading and Preprocessing (Adapted from Notebook) ---
@st.cache_data
def load_data():
    """Loads historical S&P 500 data from Yahoo Finance or local CSV."""
    
    # Use the ticker for S&P 500 (^GSPC)
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    
    # Convert index to datetime (yfinance does this by default, but keeping for robustness)
    sp500.index = pd.to_datetime(sp500.index)
    
    # Remove unused columns
    sp500 = sp500.drop(columns=["Dividends", "Stock Splits"], errors='ignore')
    
    # Create target for tomorrow's close price
    sp500["Tomorrow"] = sp500["Close"].shift(-1)
    
    # Create the binary target variable (1 if price goes up, 0 otherwise)
    sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)
    
    # Filter data to start from 1990-01-01
    sp500 = sp500.loc["1990-01-01":].copy()
    
    return sp500.dropna() # Drop the last row with a NaN 'Tomorrow' price

sp500 = load_data()

# --- Feature Engineering (Adapted from Notebook) ---

@st.cache_data
def create_features(data):
    """Creates rolling mean ratios and trend features."""
    horizons = [2, 5, 60, 250, 1000]
    new_predictors = []
    
    for horizon in horizons:
        rolling_averages = data.rolling(horizon).mean()
        
        # Close Ratio
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_averages["Close"]
        
        # Trend (Sum of 'Target' in the past 'horizon' days)
        trend_column = f"Trend_{horizon}"
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]
        
        new_predictors.extend([ratio_column, trend_column])
        
    return data.dropna(), new_predictors

sp500_features, new_predictors = create_features(sp500.copy())

# --- Prediction Functions (Backtesting and Final Prediction) ---

# Define the prediction function with a probability threshold (as used in notebook)
def predict_with_proba(train, test, predictors, model, threshold=0.6):
    model.fit(train[predictors], train["Target"])
    # Use predict_proba to get probabilities
    preds_proba = model.predict_proba(test[predictors])[:, 1]
    
    # Apply the threshold for final binary prediction
    preds = (preds_proba >= threshold).astype(int)
    
    # Return as a DataFrame for consistency
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined

# Backtesting function
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict_with_proba(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

# --- Model Training and Backtesting (on app load) ---

# Define the final model configuration
final_model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)

# Backtest to get performance metrics (uses st.cache_data for speed)
@st.cache_data
def get_backtest_results(data, _model, predictors): # FIXED: Added underscore to '_model'
    """
    NOTE: Adding the underscore to '_model' prevents Streamlit from trying to 
    hash the unhashable RandomForestClassifier object.
    """
    predictions = backtest(data, _model, predictors) # Use _model inside the function
    # Calculate key metrics
    total_trades = predictions["Predictions"].sum()
    precision = precision_score(predictions["Target"], predictions["Predictions"])
    target_up_pct = predictions["Target"].value_counts(normalize=True).get(1, 0)
    return predictions, total_trades, precision, target_up_pct

predictions_df, total_trades, precision, target_up_pct = get_backtest_results(
    sp500_features, final_model, new_predictors
) # The call remains the same, passing 'final_model'

# --- Final Prediction for Tomorrow ---

# Train the model on ALL available data with features
final_train_data = sp500_features.iloc[:]
final_model.fit(final_train_data[new_predictors], final_train_data["Target"])

# Get the last row of data for today's price and features
latest_row = final_train_data.iloc[[-1]]

# The prediction for tomorrow is based on the features calculated from today's data
# Use the predict_proba method to get the confidence level
tomorrow_proba = final_model.predict_proba(latest_row[new_predictors])[:, 1][0]
tomorrow_prediction = 1 if tomorrow_proba >= 0.6 else 0


# --- Streamlit UI Layout ---

st.header("Latest Prediction (for Tomorrow's Close)")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Last Known Close Price", 
              value=f"${latest_row['Close'].values[0]:,.2f}",
              delta=f"{latest_row['Close'].values[0] - latest_row['Open'].values[0]:,.2f}")

with col2:
    st.metric(label="Predicted Trend", 
              value="UP" if tomorrow_prediction == 1 else "DOWN/FLAT",
              delta_color="off")
    if tomorrow_prediction == 1:
        st.success("The model predicts the S&P 500 will close **higher** tomorrow.")
    else:
        st.warning("The model predicts the S&P 500 will close **lower or flat** tomorrow.")

with col3:
    st.metric(label="Prediction Confidence (Prob. of UP)", 
              value=f"{tomorrow_proba:.2f} (>{0.6})",
              delta_color="off")


st.header("Model Performance (Backtest since 1993)")
st.divider()

col_perf1, col_perf2, col_perf3 = st.columns(3)

with col_perf1:
    st.metric(label="Backtested Precision Score", 
              value=f"{precision:.4f}",
              help="The percentage of 'UP' predictions that were correct (Target=1 when Prediction=1). A higher number is better.")

with col_perf2:
    st.metric(label="Total 'UP' Trades Predicted", 
              value=f"{total_trades:,.0f} / {len(predictions_df):,.0f}",
              help=f"Number of days the model predicted an 'UP' move.")

with col_perf3:
    st.metric(label="Baseline (Actual UP days %)", 
              value=f"{target_up_pct:.4f}",
              help="The percentage of days the S&P 500 actually closed higher, if you always predicted UP.")

st.subheader("Backtest: Actual Target vs. Predictions")
st.line_chart(predictions_df, y=["Target", "Predictions"])
st.caption("Target (Blue) is the actual outcome (1=Up, 0=Down/Flat). Predictions (Red) is the model's prediction.")

st.subheader("Raw Data and Features (Last 5 Days)")
st.dataframe(sp500_features.tail(5), use_container_width=True)