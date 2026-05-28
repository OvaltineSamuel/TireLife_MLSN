import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

# 1. Load the Dataset
print("Loading data...")
df = pd.read_csv("Synthetic Automobile Tyre RUL Data.csv")

# NOTE: Verify the exact name of your target column in the CSV. 
# For this example, I am assuming the target column is called 'expected_tyre_life(km)' or 'RUL'. 
# Update this variable if it is named differently in the raw data!
TARGET_COL = "expected_tyre_life(km)" 

# 2. Split the Data
# We do a standard 80/20 split for training and validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training on {len(train_df)} rows, Validating on {len(val_df)} rows.")

# 3. Define the Data Configuration
# We categorize your features based on the core and additional features list
data_config = DataConfig(
    target=[TARGET_COL],
    continuous_cols=[
        "current_tread_depth(mm)",
        "Standard_tread_depth(mm)",
        "kilometers_driven(km)",
        "average_inflation_pressure(psi)",
        "recommended_inflation_pressure(psi)",
        "vehicle_sprung_mass(kg)",
        "vehicle_acceleration(0-100 km/h in seconds)",
        "maximum_power(hp)",
        "maximum_torque(N/m)",
        "maximum_speed (km/h)",
        "vehicle_mileage(mpg)",
        "average_tread_temperature(celsius)",
        "tyre_age(years)",
        "number_of_punctures"
    ],
    categorical_cols=[
        "tread_wear_rating (UTQG)",
        "retreaded",
        "road_condition",
        "weather_condition",
        "axle_type(driven/dead)",
        "vehicle_model",
        "fuel_type",
        "transmission_type",
        "tyre_brand",
        "tyre_size",
        "tread_material",
        "country"
    ],
    # This automatically scales numerical data and ordinal-encodes categorical data
    normalize_continuous_features=True 
)

# 4. Define the Trainer Configuration
trainer_config = TrainerConfig(
    auto_lr_find=False, 
    batch_size=1024,   
    max_epochs=20,     
    early_stopping="valid_loss",
    early_stopping_patience=3, 
    accelerator="auto", # Let the library auto-detect your Mac's CPU or M-chip
    devices=1           # Tell it to use 1 processor
)

optimizer_config = OptimizerConfig()

# 5. Define the Model Architecture (MLP Baseline)
model_config = CategoryEmbeddingModelConfig(
    task="regression", # Predicting a continuous number (RUL)
    layers="1024-512-256", # The size of the hidden layers in the network
    activation="ReLU",
    learning_rate=1e-3
)

# 6. Initialize the Model
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

# 7. Train the Model!
print("Starting training...")
tabular_model.fit(train=train_df, validation=val_df)

# 8. Evaluate
result = tabular_model.evaluate(val_df)
print("\nValidation Results:", result)

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# 1. Have your model make predictions on the validation set
# (PyTorch Tabular returns the original dataframe with a new column added for predictions)
pred_df = tabular_model.predict(val_df)

# NOTE: PyTorch Tabular usually names the prediction column by adding 'prediction' to the target name.
# It should be named: 'expected_tyre_life(km)_prediction'
# Let's assign the actual vs predicted columns to variables

# Grab the ACTUAL values from the original validation dataframe
actual_values = val_df["expected_tyre_life(km)"]

# Grab the PREDICTED values from the prediction dataframe
predicted_values = pred_df["expected_tyre_life(km)_prediction"]

# 2. Calculate the readable metrics
mae = mean_absolute_error(actual_values, predicted_values)
rmse = root_mean_squared_error(actual_values, predicted_values)
r2 = r2_score(actual_values, predicted_values)

# 3. Print them out!
print(f"Mean Absolute Error (MAE): Off by ~{mae:.2f} kilometers on average")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kilometers")
print(f"R-Squared (Accuracy equivalent): {r2:.4f} (or {r2 * 100:.2f}%)")