import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from Processor import main


# Data preprocessing
n = 1000000
columns_to_drop = ['Address', 'Bedrooms', 'URL', 'Price_paid', 'Lease_Term', 'EPC_Rating_Current', 'EPC_Rating_Potential', 'Year_Built', 'EPC_Date']
columns_to_encode = ['Type', 'Tenure', 'New_build']
main(n, columns_to_drop, columns_to_encode)

file_path = '/Users/user/Documents/UCL work/Final Dissertation/Transaction Data/Cleaned_data/final_processed.csv'
data = pd.read_csv(file_path)
scaler = StandardScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

column_index = 1

# Extract the relevant mean and standard deviation values for the specific column
y_mean_val = scaler.mean_[column_index]
y_std_val = scaler.scale_[column_index]

# Apply Winsorization
data_winsorized = data.copy()
numeric_columns = data.select_dtypes(include=[np.number]).columns
for column in numeric_columns:
    data_winsorized[column] = winsorize(data[column], limits=[0.05, 0.05])

target = 'price_per_sqr'
X = data.drop(columns=[target,'Average price'])
y = data[target]

# Stratified split to maintain distribution
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(X, pd.qcut(y, 10, labels=False)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the ANN model with L2 regularization
model = Sequential()
model.add(Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.1))  # Reduced dropout rate

model.add(Dense(128, kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.1))

model.add(Dense(1, activation='linear'))

# Compile the model
optimizer = Adam(learning_rate=0.001) #0.0003
model.compile(optimizer=optimizer, loss='mean_absolute_error')

#TODO try L2 loss

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=20, min_lr=1e-6)

# Train the model
history = model.fit(X_train, y_train, epochs=1000, batch_size=128, validation_split=0.2, 
                    verbose=1, callbacks=[early_stopping, reduce_lr])

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
y_test = y_test * y_std_val + y_mean_val
y_pred = y_pred * y_std_val + y_mean_val
y = y * y_std_val + y_mean_val
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
average_price = y.mean()

print(f'R-squared: {r2}')
print(f'MAE: {mae}')
print(f'RMSE: {rmse}')
print(f'MAPE: {mape}')
print(f'Average Price: {average_price}')

# Custom function to calculate validation loss without regularization
def custom_val_loss():
    val_predictions = model.predict(X_test)
    return mean_absolute_error(y_test, val_predictions)

# Plot training history
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss (with regularization)')
plt.title('Training and Validation Loss (with regularization)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot scatter plot between actual and predicted values
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.text(0.1, 0.9, f'Data Points: {len(y_test)}', transform=plt.gca().transAxes)
plt.tight_layout()
plt.show()

