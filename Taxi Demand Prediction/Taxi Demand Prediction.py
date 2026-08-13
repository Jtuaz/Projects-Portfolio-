# Imports for data manipulation, machine learning models, scaling, and visualization

import numpy as np  # Matrix and numerical operations
import pandas as pd  # Dataframe manipulation and parquet file parsing
import matplotlib.pyplot as plt  # Plotting and data visualization
import seaborn as sns  # Enhanced statistical data visualization
from sklearn.model_selection import train_test_split  # Dataset splitting utility
from sklearn.linear_model import LinearRegression  # Baseline linear regression algorithm
from sklearn.ensemble import RandomForestRegressor  # Ensemble random forest regressor
from sklearn.ensemble import GradientBoostingRegressor  # Gradient boosting regressor
from sklearn.preprocessing import StandardScaler  # Feature normalization scaler
from sklearn.cluster import KMeans  # K-Means clustering algorithm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score  # Regression and classification evaluation metrics

# Define base folder path containing parquet files
BASE_DATASET_DIRECTORY_PATH = r"D:\Big Data\Project"

# Read raw parquet datasets into individual DataFrames
YELLOW_TAXI_DATAFRAME = pd.read_parquet(BASE_DATASET_DIRECTORY_PATH + r"\yellow_tripdata_2025-01.parquet")  # Read Yellow taxi dataset
GREEN_TAXI_DATAFRAME  = pd.read_parquet(BASE_DATASET_DIRECTORY_PATH + r"\green_tripdata_2025-01.parquet")  # Read Green taxi dataset
FOR_HIRE_VEHICLE_DATAFRAME = pd.read_parquet(BASE_DATASET_DIRECTORY_PATH + r"\fhv_tripdata_2025-01.parquet")  # Read FHV dataset

# Display initial record counts and column structures across all datasets
print("Yellow:", YELLOW_TAXI_DATAFRAME.shape)  # Output dimensions of yellow dataset
print("Green:", GREEN_TAXI_DATAFRAME.shape)  # Output dimensions of green dataset
print("FHV:", FOR_HIRE_VEHICLE_DATAFRAME.shape)  # Output dimensions of FHV dataset
print(YELLOW_TAXI_DATAFRAME.columns)  # Display initial column names for yellow
print(GREEN_TAXI_DATAFRAME.columns)  # Display initial column names for green
print(FOR_HIRE_VEHICLE_DATAFRAME.columns)  # Display initial column names for FHV

# Standardize column headers across Yellow Taxi DataFrame
YELLOW_TAXI_DATAFRAME = YELLOW_TAXI_DATAFRAME.rename(columns={
    'tpep_pickup_datetime': 'pickup_datetime',
    'tpep_dropoff_datetime': 'dropoff_datetime',
    'PULocationID': 'location_id',
    'DOLocationID': 'dropoff_location_id'
})

# Standardize column headers across Green Taxi DataFrame
GREEN_TAXI_DATAFRAME = GREEN_TAXI_DATAFRAME.rename(columns={
    'lpep_pickup_datetime': 'pickup_datetime',
    'lpep_dropoff_datetime': 'dropoff_datetime',
    'PULocationID': 'location_id',
    'DOLocationID': 'dropoff_location_id'
})

# Standardize column headers across For-Hire Vehicle DataFrame
FOR_HIRE_VEHICLE_DATAFRAME = FOR_HIRE_VEHICLE_DATAFRAME.rename(columns={
    'PUlocationID': 'location_id',
    'DOlocationID': 'dropoff_location_id',
    'dropOff_datetime': 'dropoff_datetime'
})

# Filter DataFrames to retain only relevant modeling features
YELLOW_TAXI_DATAFRAME = YELLOW_TAXI_DATAFRAME[['pickup_datetime', 'location_id']]  # Keep essential yellow columns
GREEN_TAXI_DATAFRAME  = GREEN_TAXI_DATAFRAME[['pickup_datetime', 'location_id']]  # Keep essential green columns
FOR_HIRE_VEHICLE_DATAFRAME = FOR_HIRE_VEHICLE_DATAFRAME[['pickup_datetime', 'location_id']]  # Keep essential FHV columns

# Tag dataset source for classification tracking
YELLOW_TAXI_DATAFRAME['service'] = 'yellow'  # Label yellow trip rows
GREEN_TAXI_DATAFRAME['service'] = 'green'  # Label green trip rows
FOR_HIRE_VEHICLE_DATAFRAME['service'] = 'fhv'  # Label FHV trip rows

# Concatenate all taxi services into a single unified dataset
COMBINED_TAXI_DATAFRAME = pd.concat([YELLOW_TAXI_DATAFRAME, GREEN_TAXI_DATAFRAME, FOR_HIRE_VEHICLE_DATAFRAME], ignore_index=True)
print("Combined shape:", COMBINED_TAXI_DATAFRAME.shape)  # Display total row count
print(COMBINED_TAXI_DATAFRAME.head())  # Preview combined dataset top rows

# Parse timestamps and handle invalid entries
COMBINED_TAXI_DATAFRAME['pickup_datetime'] = pd.to_datetime(COMBINED_TAXI_DATAFRAME['pickup_datetime'], errors='coerce')  # Coerce unparseable dates to NaT
COMBINED_TAXI_DATAFRAME = COMBINED_TAXI_DATAFRAME.dropna(subset=['pickup_datetime', 'location_id'])  # Drop missing timestamp or location entries
COMBINED_TAXI_DATAFRAME['location_id'] = COMBINED_TAXI_DATAFRAME['location_id'].astype(int)  # Typecast location IDs to integers

# Extract time components for cyclical feature encoding
COMBINED_TAXI_DATAFRAME['hour'] = COMBINED_TAXI_DATAFRAME['pickup_datetime'].dt.hour  # Extract hour of day
COMBINED_TAXI_DATAFRAME['day_of_week'] = COMBINED_TAXI_DATAFRAME['pickup_datetime'].dt.dayofweek  # Extract day of week index
COMBINED_TAXI_DATAFRAME['hour_sin'] = np.sin(2 * np.pi * COMBINED_TAXI_DATAFRAME['hour'] / 24)  # Sine transformation for hourly seasonality
COMBINED_TAXI_DATAFRAME['hour_cos'] = np.cos(2 * np.pi * COMBINED_TAXI_DATAFRAME['hour'] / 24)  # Cosine transformation for hourly seasonality
COMBINED_TAXI_DATAFRAME['time_bin'] = COMBINED_TAXI_DATAFRAME['pickup_datetime'].dt.floor('h')  # Truncate timestamps to hourly intervals

# Aggregate total trips grouped by pickup location and hourly time bin
AGGREGATED_DEMAND_DATAFRAME = COMBINED_TAXI_DATAFRAME.groupby(['location_id', 'time_bin']).size().reset_index(name='trip_count')
print(AGGREGATED_DEMAND_DATAFRAME.head())  # Inspect aggregated demand table

# Re-extract temporal features post-aggregation
AGGREGATED_DEMAND_DATAFRAME['hour'] = AGGREGATED_DEMAND_DATAFRAME['time_bin'].dt.hour  # Hour of day feature
AGGREGATED_DEMAND_DATAFRAME['day_of_week'] = AGGREGATED_DEMAND_DATAFRAME['time_bin'].dt.dayofweek  # Day of week feature
AGGREGATED_DEMAND_DATAFRAME['hour_sin'] = np.sin(2 * np.pi * AGGREGATED_DEMAND_DATAFRAME['hour'] / 24)  # Sine encoded hour
AGGREGATED_DEMAND_DATAFRAME['hour_cos'] = np.cos(2 * np.pi * AGGREGATED_DEMAND_DATAFRAME['hour'] / 24)  # Cosine encoded hour

# Split features and target labels
FEATURE_MATRIX = AGGREGATED_DEMAND_DATAFRAME[['location_id', 'hour', 'day_of_week', 'hour_sin', 'hour_cos']]  # Input predictor matrix
TARGET_VECTOR = AGGREGATED_DEMAND_DATAFRAME['trip_count']  # Target variable to predict

# Split dataset into training and validation subsets
TRAIN_FEATURE_MATRIX, TEST_FEATURE_MATRIX, TRAIN_TARGET_VECTOR, TEST_TARGET_VECTOR = train_test_split(
    FEATURE_MATRIX, TARGET_VECTOR, test_size=0.2, random_state=42
)  # Perform 80/20 train-test split

# Linear Regression Model Training & Evaluation
LINEAR_REGRESSION_MODEL = LinearRegression()  # Instantiate Linear Regression model
LINEAR_REGRESSION_MODEL.fit(TRAIN_FEATURE_MATRIX, TRAIN_TARGET_VECTOR)  # Fit model on training data
PREDICTED_TARGET_LINEAR_REGRESSION = LINEAR_REGRESSION_MODEL.predict(TEST_FEATURE_MATRIX)  # Predict test set
ROOT_MEAN_SQUARED_ERROR_LINEAR_REGRESSION = mean_squared_error(TEST_TARGET_VECTOR, PREDICTED_TARGET_LINEAR_REGRESSION) ** 0.5  # Calculate RMSE
print("Linear Regression RMSE:", ROOT_MEAN_SQUARED_ERROR_LINEAR_REGRESSION)  # Print linear model error

# Random Forest Model Training & Evaluation
RANDOM_FOREST_MODEL = RandomForestRegressor(n_estimators=50, random_state=42)  # Instantiate Random Forest with 50 trees
RANDOM_FOREST_MODEL.fit(TRAIN_FEATURE_MATRIX, TRAIN_TARGET_VECTOR)  # Fit model on training data
PREDICTED_TARGET_RANDOM_FOREST = RANDOM_FOREST_MODEL.predict(TEST_FEATURE_MATRIX)  # Predict test set
ROOT_MEAN_SQUARED_ERROR_RANDOM_FOREST = mean_squared_error(TEST_TARGET_VECTOR, PREDICTED_TARGET_RANDOM_FOREST) ** 0.5  # Calculate RMSE
print("Random Forest RMSE:", ROOT_MEAN_SQUARED_ERROR_RANDOM_FOREST)  # Print random forest error

# Gradient Boosting Model Training & Evaluation
GRADIENT_BOOSTING_MODEL = GradientBoostingRegressor(n_estimators=50, random_state=42)  # Instantiate Gradient Boosting Regressor
GRADIENT_BOOSTING_MODEL.fit(TRAIN_FEATURE_MATRIX, TRAIN_TARGET_VECTOR)  # Fit model on training data
PREDICTED_TARGET_GRADIENT_BOOSTING = GRADIENT_BOOSTING_MODEL.predict(TEST_FEATURE_MATRIX)  # Predict test set
ROOT_MEAN_SQUARED_ERROR_GRADIENT_BOOSTING = mean_squared_error(TEST_TARGET_VECTOR, PREDICTED_TARGET_GRADIENT_BOOSTING) ** 0.5  # Calculate RMSE
print("Gradient Boosting RMSE:", ROOT_MEAN_SQUARED_ERROR_GRADIENT_BOOSTING)  # Print gradient boosting error

# Print direct summary comparison of model RMSE metrics
print("\nModel Comparison:")
print("Linear Regression RMSE:", ROOT_MEAN_SQUARED_ERROR_LINEAR_REGRESSION)
print("Random Forest RMSE:", ROOT_MEAN_SQUARED_ERROR_RANDOM_FOREST)
print("Gradient Boosting RMSE:", ROOT_MEAN_SQUARED_ERROR_GRADIENT_BOOSTING)

# Helper function to compute and output comprehensive regression performance metrics
def EVALUATE_REGRESSION_MODEL(TRUE_TARGET_VALUES, PREDICTED_TARGET_VALUES, MODEL_NAME_STRING):
    CALCULATED_ROOT_MEAN_SQUARED_ERROR = mean_squared_error(TRUE_TARGET_VALUES, PREDICTED_TARGET_VALUES) ** 0.5  # Calculate Root Mean Squared Error
    CALCULATED_MEAN_ABSOLUTE_ERROR = mean_absolute_error(TRUE_TARGET_VALUES, PREDICTED_TARGET_VALUES)  # Calculate Mean Absolute Error
    CALCULATED_R2_SCORE = r2_score(TRUE_TARGET_VALUES, PREDICTED_TARGET_VALUES)  # Calculate R-Squared goodness of fit
    print(f"{MODEL_NAME_STRING} Performance:")
    print(f"RMSE: {CALCULATED_ROOT_MEAN_SQUARED_ERROR:.2f}, MAE: {CALCULATED_MEAN_ABSOLUTE_ERROR:.2f}, R²: {CALCULATED_R2_SCORE:.4f}")
    return CALCULATED_ROOT_MEAN_SQUARED_ERROR, CALCULATED_MEAN_ABSOLUTE_ERROR, CALCULATED_R2_SCORE

# Compute complete metric profiles for all three models
METRICS_LINEAR_REGRESSION = EVALUATE_REGRESSION_MODEL(TEST_TARGET_VECTOR, PREDICTED_TARGET_LINEAR_REGRESSION, "Linear Regression")
METRICS_RANDOM_FOREST = EVALUATE_REGRESSION_MODEL(TEST_TARGET_VECTOR, PREDICTED_TARGET_RANDOM_FOREST, "Random Forest")
METRICS_GRADIENT_BOOSTING = EVALUATE_REGRESSION_MODEL(TEST_TARGET_VECTOR, PREDICTED_TARGET_GRADIENT_BOOSTING, "Gradient Boosting")

# Evaluate High vs Low Demand binary classification accuracy via F1-Score
MEDIAN_DEMAND_THRESHOLD = TEST_TARGET_VECTOR.median()  # Determine median trip volume threshold
BINARY_TRUE_CLASS_TARGET = (TEST_TARGET_VECTOR > MEDIAN_DEMAND_THRESHOLD).astype(int)  # Binarize ground truth labels
BINARY_PREDICTED_CLASS_LINEAR_REGRESSION = (PREDICTED_TARGET_LINEAR_REGRESSION > MEDIAN_DEMAND_THRESHOLD).astype(int)  # Binarize linear predictions
BINARY_PREDICTED_CLASS_RANDOM_FOREST = (PREDICTED_TARGET_RANDOM_FOREST > MEDIAN_DEMAND_THRESHOLD).astype(int)  # Binarize random forest predictions
BINARY_PREDICTED_CLASS_GRADIENT_BOOSTING = (PREDICTED_TARGET_GRADIENT_BOOSTING > MEDIAN_DEMAND_THRESHOLD).astype(int)  # Binarize gradient boosting predictions

# Print classification F1-Score metrics
print("\nOptional F1-Scores for High vs Low Demand Classification:")
print("Linear Regression F1:", f1_score(BINARY_TRUE_CLASS_TARGET, BINARY_PREDICTED_CLASS_LINEAR_REGRESSION))
print("Random Forest F1:", f1_score(BINARY_TRUE_CLASS_TARGET, BINARY_PREDICTED_CLASS_RANDOM_FOREST))
print("Gradient Boosting F1:", f1_score(BINARY_TRUE_CLASS_TARGET, BINARY_PREDICTED_CLASS_GRADIENT_BOOSTING))

# Create aggregated prediction comparison dataset for visual plotting
PREDICTION_AGGREGATED_DATAFRAME = AGGREGATED_DEMAND_DATAFRAME.copy()  # Copy aggregated demand dataframe
PREDICTION_AGGREGATED_DATAFRAME['pred_linear'] = LINEAR_REGRESSION_MODEL.predict(FEATURE_MATRIX)  # Full dataset linear predictions
PREDICTION_AGGREGATED_DATAFRAME['pred_rf'] = RANDOM_FOREST_MODEL.predict(FEATURE_MATRIX)  # Full dataset random forest predictions
PREDICTION_AGGREGATED_DATAFRAME['pred_gb'] = GRADIENT_BOOSTING_MODEL.predict(FEATURE_MATRIX)  # Full dataset gradient boosting predictions

# Identify top 5 pickup locations ranked by total volume
TOP_DEMAND_LOCATION_IDENTIFIERS = PREDICTION_AGGREGATED_DATAFRAME.groupby('location_id')['trip_count'].sum().sort_values(ascending=False).head(5).index.tolist()

# Plot temporal actual vs predicted curves for top 5 demand locations
for CURRENT_LOCATION_IDENTIFIER in TOP_DEMAND_LOCATION_IDENTIFIERS:
    LOCATION_SPECIFIC_DATAFRAME = PREDICTION_AGGREGATED_DATAFRAME[PREDICTION_AGGREGATED_DATAFRAME['location_id'] == CURRENT_LOCATION_IDENTIFIER].copy()  # Filter location data
    LOCATION_SPECIFIC_DATAFRAME['time_bin'] = pd.to_datetime(LOCATION_SPECIFIC_DATAFRAME['time_bin'])  # Convert time bins to datetime
    plt.figure()  # Initialize plot figure
    sns.lineplot(data=LOCATION_SPECIFIC_DATAFRAME, x='time_bin', y='trip_count', label='Actual', color='black')  # Plot ground truth trip count
    sns.lineplot(data=LOCATION_SPECIFIC_DATAFRAME, x='time_bin', y='pred_linear', label='Linear Regression', color='red')  # Plot linear model prediction
    sns.lineplot(data=LOCATION_SPECIFIC_DATAFRAME, x='time_bin', y='pred_rf', label='Random Forest', color='green')  # Plot random forest prediction
    sns.lineplot(data=LOCATION_SPECIFIC_DATAFRAME, x='time_bin', y='pred_gb', label='Gradient Boosting', color='blue')  # Plot gradient boosting prediction
    plt.title(f'Taxi Demand & Model Predictions at Location {CURRENT_LOCATION_IDENTIFIER}')  # Chart title
    plt.xlabel('Time')  # X-axis label
    plt.ylabel('Trip Count')  # Y-axis label
    plt.xticks(rotation=45)  # Rotate x-axis timestamp labels
    plt.legend()  # Render chart legend
    plt.tight_layout()  # Optimize layout spacing
    plt.show()  # Display chart

# Generate RMSE Comparison Bar Chart
ROOT_MEAN_SQUARED_ERROR_DICTIONARY = {
    'Linear Regression': ROOT_MEAN_SQUARED_ERROR_LINEAR_REGRESSION,
    'Random Forest': ROOT_MEAN_SQUARED_ERROR_RANDOM_FOREST,
    'Gradient Boosting': ROOT_MEAN_SQUARED_ERROR_GRADIENT_BOOSTING
}
plt.figure()  # Initialize bar chart figure
sns.barplot(x=list(ROOT_MEAN_SQUARED_ERROR_DICTIONARY.keys()), y=list(ROOT_MEAN_SQUARED_ERROR_DICTIONARY.values()), palette=['red', 'green', 'blue'])  # Plot error comparison bars
plt.ylabel('RMSE')  # Y-axis label
plt.title('Model Performance Comparison')  # Title
plt.show()  # Display comparison plot

# CLUSTERING ANALYSIS SECTION
sns.set(style="whitegrid")  # Configure plot style background grid
plt.rcParams["figure.figsize"] = (14, 6)  # Configure global figure dimensions

# Compute average hourly trip demand grouped by location and hour
HOURLY_LOCATION_AVERAGE_DATAFRAME = AGGREGATED_DEMAND_DATAFRAME.groupby(['location_id', 'hour']).agg(
    avg_trips=('trip_count', 'mean')
).reset_index()

# Scale clustering features to mean 0 and variance 1
FEATURE_STANDARDIZATION_SCALER = StandardScaler()  # Instantiate standard scaler
STANDARDIZED_CLUSTER_FEATURES = FEATURE_STANDARDIZATION_SCALER.fit_transform(HOURLY_LOCATION_AVERAGE_DATAFRAME[['location_id', 'hour', 'avg_trips']])  # Scale input features

# Find optimal cluster count using Elbow Method
KMEANS_INERTIA_VALUES_LIST = []  # Initialize inertia tracking list
CLUSTER_K_RANGE = range(2, 11)  # Test cluster ranges from k=2 to k=10
for CURRENT_CLUSTER_K in CLUSTER_K_RANGE:
    KMEANS_TEST_MODEL = KMeans(n_clusters=CURRENT_CLUSTER_K, random_state=42)  # Instantiate trial K-Means model
    KMEANS_TEST_MODEL.fit(STANDARDIZED_CLUSTER_FEATURES)  # Fit clustering on scaled features
    KMEANS_INERTIA_VALUES_LIST.append(KMEANS_TEST_MODEL.inertia_)  # Append sum of squared error distance

# Print calculated Elbow Method inertia scores
print("\nElbow Method Values:")
print("k\tInertia")
for CURRENT_CLUSTER_K, INERTIA_VALUE in zip(CLUSTER_K_RANGE, KMEANS_INERTIA_VALUES_LIST):
    print(f"{CURRENT_CLUSTER_K}\t{INERTIA_VALUE}")

# Plot Elbow Method curve to visualize optimal k selection
plt.figure()  # Initialize elbow plot figure
plt.plot(CLUSTER_K_RANGE, KMEANS_INERTIA_VALUES_LIST, 'o-', color='blue')  # Plot inertia points and connecting line
plt.xlabel('Number of Clusters (k)')  # X-axis label
plt.ylabel('Inertia')  # Y-axis label
plt.title('Elbow Method for Optimal k')  # Chart title
plt.show()  # Render elbow plot

# Perform K-Means Clustering using selected optimal cluster count
OPTIMAL_CLUSTER_COUNT_K = 4  # Selected optimal k value from elbow curve
FINAL_KMEANS_MODEL = KMeans(n_clusters=OPTIMAL_CLUSTER_COUNT_K, random_state=42)  # Instantiate final K-Means estimator
HOURLY_LOCATION_AVERAGE_DATAFRAME['cluster'] = FINAL_KMEANS_MODEL.fit_predict(STANDARDIZED_CLUSTER_FEATURES)  # Assign cluster labels

# Reshape data into matrix format (Location ID x Hour) for spatial-temporal heatmaps
DEMAND_HEATMAP_PIVOT_TABLE = HOURLY_LOCATION_AVERAGE_DATAFRAME.pivot(index='location_id', columns='hour', values='avg_trips')

# Plot demand heatmap across locations and hours of the day
plt.figure(figsize=(14, 12))  # Set figure size for detailed matrix view
sns.heatmap(DEMAND_HEATMAP_PIVOT_TABLE, cmap='YlOrRd', linewidths=0.5)  # Plot heatmap with Yellow-Orange-Red palette
plt.title('Average Taxi Demand per Location per Hour')  # Heatmap title
plt.xlabel('Hour of Day')  # X-axis label
plt.ylabel('Location ID')  # Y-axis label
plt.show()  # Render demand heatmap

# Display cluster assignments for individual locations
print("Cluster assignment for top locations:")
print(HOURLY_LOCATION_AVERAGE_DATAFRAME[['location_id', 'cluster']].drop_duplicates().sort_values('cluster'))  # Print location cluster mapping

# EXPORT PREDICTIONS FOR PRESENTATIONS & REPORTS
POWERPOINT_PREDICTION_EXPORT_TABLE = TEST_FEATURE_MATRIX.copy()  # Copy test feature set
POWERPOINT_PREDICTION_EXPORT_TABLE = POWERPOINT_PREDICTION_EXPORT_TABLE.reset_index(drop=True)  # Reset table row indexing
POWERPOINT_PREDICTION_EXPORT_TABLE['actual_trip_count'] = TEST_TARGET_VECTOR.values  # Append ground truth target column
POWERPOINT_PREDICTION_EXPORT_TABLE['pred_linear'] = PREDICTED_TARGET_LINEAR_REGRESSION  # Append linear regression predictions
POWERPOINT_PREDICTION_EXPORT_TABLE['pred_random_forest'] = PREDICTED_TARGET_RANDOM_FOREST  # Append random forest predictions
POWERPOINT_PREDICTION_EXPORT_TABLE['pred_gradient_boost'] = PREDICTED_TARGET_GRADIENT_BOOSTING  # Append gradient boosting predictions

print("\n=== POWERPOINT PREDICTION TABLE (TOP 20 ROWS) ===")
print(POWERPOINT_PREDICTION_EXPORT_TABLE.head(20))  # Display preview of exported prediction table
POWERPOINT_PREDICTION_EXPORT_TABLE.to_csv("taxi_model_predictions_ppt.csv", index=False)  # Export test predictions table to CSV

# Create and export full time-series dataset with model predictions for chart rendering
TIME_SERIES_CHART_EXPORT_DATAFRAME = AGGREGATED_DEMAND_DATAFRAME[['location_id', 'hour', 'trip_count']].copy()  # Prepare time series structure
TIME_SERIES_CHART_EXPORT_DATAFRAME['pred_linear'] = LINEAR_REGRESSION_MODEL.predict(FEATURE_MATRIX)  # Full set linear predictions
TIME_SERIES_CHART_EXPORT_DATAFRAME['pred_random_forest'] = RANDOM_FOREST_MODEL.predict(FEATURE_MATRIX)  # Full set random forest predictions
TIME_SERIES_CHART_EXPORT_DATAFRAME['pred_gradient_boost'] = GRADIENT_BOOSTING_MODEL.predict(FEATURE_MATRIX)  # Full set gradient boosting predictions
TIME_SERIES_CHART_EXPORT_DATAFRAME.to_csv("taxi_time_series_predictions.csv", index=False)  # Export full time-series predictions to CSV

print("\nFiles saved:")
print("- taxi_model_predictions_ppt.csv (table for slides)")
print("- taxi_time_series_predictions.csv (for charts)")

# Align indices on aggregated dataset for direct target assignment
AGGREGATED_DEMAND_DATAFRAME = AGGREGATED_DEMAND_DATAFRAME.reset_index(drop=True)  # Reset index on main demand dataframe
AGGREGATED_DEMAND_DATAFRAME['pred_rf'] = RANDOM_FOREST_MODEL.predict(FEATURE_MATRIX)  # Assign overall random forest predictions
AGGREGATED_DEMAND_DATAFRAME['actual'] = TARGET_VECTOR.values  # Assign target vector array

# Prepare time-series plotting dataframe for Random Forest overview
FULL_TIME_SERIES_PLOT_DATAFRAME = AGGREGATED_DEMAND_DATAFRAME[['time_bin', 'location_id', 'actual', 'pred_rf']].copy()  # Copy plotting subset
FULL_TIME_SERIES_PLOT_DATAFRAME = FULL_TIME_SERIES_PLOT_DATAFRAME.sort_values('time_bin')  # Sort chronologically by time bin

# Render time series plot comparing actual demand versus Random Forest predictions
plt.figure(figsize=(12, 6))  # Initialize figure size
plt.plot(FULL_TIME_SERIES_PLOT_DATAFRAME['time_bin'], FULL_TIME_SERIES_PLOT_DATAFRAME['actual'], label='Actual Demand', color='black')  # Plot actual curve
plt.plot(FULL_TIME_SERIES_PLOT_DATAFRAME['time_bin'], FULL_TIME_SERIES_PLOT_DATAFRAME['pred_rf'], label='Predicted Demand (RF)', color='green')  # Plot predicted curve
plt.title("Actual vs Predicted Taxi Demand")  # Chart title
plt.xlabel("Time")  # X-axis label
plt.ylabel("Trip Count")  # Y-axis label
plt.xticks(rotation=45)  # Rotate x-axis dates
plt.legend()  # Render legend
plt.tight_layout()  # Apply tight layout padding
plt.show()  # Display time series plot

# EXPORT AGGREGATED TIME BIN DATASET FOR EXCEL REPORTING
TIME_BIN_EXCEL_EXPORT_DATAFRAME = AGGREGATED_DEMAND_DATAFRAME[['location_id', 'time_bin', 'trip_count']].copy()  # Prepare time bin export dataframe
TIME_BIN_EXCEL_EXPORT_DATAFRAME['pred_rf'] = RANDOM_FOREST_MODEL.predict(FEATURE_MATRIX)  # Append Random Forest predictions
TIME_BIN_EXCEL_EXPORT_DATAFRAME = TIME_BIN_EXCEL_EXPORT_DATAFRAME.sort_values(['location_id', 'time_bin'])  # Sort hierarchy by location and time
TIME_BIN_EXCEL_EXPORT_DATAFRAME.to_csv("taxi_time_bin_data.csv", index=False)  # Export final dataset to CSV format for Excel
print("Export complete: taxi_time_bin_data.csv")  # Print export confirmation message
