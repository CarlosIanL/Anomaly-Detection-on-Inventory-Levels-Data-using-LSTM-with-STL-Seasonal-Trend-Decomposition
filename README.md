This project investigates using Long Short-Term Memory (LSTM) with Seasonal-Trend Decomposition (STL) to detect anomalies in Bengco's Agricultural Supply inventory data.

Problem: Bengco relies on a manual inventory system, leading to potential inaccuracies and inefficiencies.

Solution: Develop a web application that automatically detects anomalies in inventory levels using a machine learning model.

Methodology:
-Data Collection: Inventory data from the past 4 years is collected.

Data Preprocessing:
-Relevant data like inventory levels and dates are extracted.
-Anomalies caused by external events (disasters, pests, theft) are filtered.
-Train-test split is performed (85% training, 15% testing/validation).

Model Development:
-STL decomposition separates the data into trend, seasonality, and residual components.
-An LSTM autoencoder model is trained on the residuals to learn normal patterns.
-The model reconstructs the data, and anomalies are identified by comparing the original data to the reconstruction with high error (Mean Squared Error).

Web Application Development:
-A web application is built using Python and Flask framework.
-Users can upload CSV files containing inventory data.
-The application processes the data and detects anomalies using the trained LSTM model.

Example outputs:

Upload Page
![Upload Page](https://github.com/CarlosIanL/Anomaly-Detection-on-Inventory-Levels-Data-using-LSTM-with-STL-Seasonal-Trend-Decomposition/assets/132331338/032ad514-0ecd-4b2a-b8e6-59f21d8bd52e)

