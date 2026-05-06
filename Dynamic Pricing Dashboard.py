# Dynamic Pricing Dashboard (Streamlit)
# Save this file as app.py and run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.random.seed(42)
rows = 500

demand = np.random.randint(50, 500, rows)
season = np.random.choice([0, 1], rows, p=[0.4, 0.6])
competitor_price = np.random.randint(80, 150, rows)

best_price = (
    0.3 * demand
    + 10 * season
    + 0.5 * competitor_price
    + np.random.normal(0, 10, rows)
)

df = pd.DataFrame({
    "demand": demand,
    "season": season,
    "competitor_price": competitor_price,
    "best_price": best_price
})


X = df[["demand", "season", "competitor_price"]]
y = df["best_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)


st.title("📊 Dynamic Pricing Model Dashboard")
st.write("Predict optimal product prices based on demand, season, and competitor pricing.")

# Input form
st.sidebar.header("Enter Scenario")
demand_input = st.sidebar.slider("Demand (units)", 50, 500, 200)
season_input = st.sidebar.radio("Season", ["Off-season", "Peak"])
competitor_price_input = st.sidebar.slider("Competitor Price", 80, 150, 100)

season_value = 1 if season_input == "Peak" else 0

# Prediction
new_data = pd.DataFrame({
    "demand": [demand_input],
    "season": [season_value],
    "competitor_price": [competitor_price_input]
})

predicted_price = model.predict(new_data)[0]
st.metric("💡 Suggested Price", f"${predicted_price:.2f}")


st.subheader("Data Insights")

# Demand vs Price
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

sns.scatterplot(x="demand", y="best_price", data=df, alpha=0.6, ax=ax[0])
ax[0].set_title("Demand vs Best Price")

sns.scatterplot(x="competitor_price", y="best_price", hue="season", data=df, alpha=0.6, palette="Set2", ax=ax[1])
ax[1].set_title("Competitor Price vs Best Price")

sns.boxplot(x="season", y="best_price", data=df, palette="coolwarm", ax=ax[2])
ax[2].set_xticklabels(["Off-season", "Peak"])
ax[2].set_title("Price Distribution by Season")

st.pyplot(fig)

if st.checkbox("Show raw data"):
    st.dataframe(df.head(20))