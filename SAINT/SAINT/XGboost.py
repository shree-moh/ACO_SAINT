import numpy as np
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score
import xgboost as xgb

# Lists to store evaluation metrics for each fold
pd_list = []
pf_list = []
bal_list = []
fir_list = []
accuracy_list = []

# Function to evaluate the classifier and print confusion matrix
def classifier_eval(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix: \n', cm)

    # Extracting TP, TN, FP, FN
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    # Calculating metrics
    PD = TP / (TP + FN)
    PF = FP / (FP + TN)
    balance = 1 - np.sqrt((1 - PD)**2 + PF**2) / np.sqrt(2)
    FI = (cm[1, 1] + cm[0, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    FIR = (PD - FI) / PD
    accuracy = accuracy_score(y_test, y_pred)

    return PD, PF, balance, FIR, accuracy

# CSV file path
csv_file_path = "EQ.csv"

# Read CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Extract features (X) and target variable (y) from the DataFrame
X = df.drop(columns=['class'])
y = df['class']

# Split the data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Set up k-fold cross-validation
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Initialize MinMaxScaler for normalization
scaler = MinMaxScaler()
X_test_normalized = scaler.fit_transform(X_test)

# Perform k-fold cross-validation
for train_index, val_index in kf.split(X_train):
    X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

    # Preprocessing
    # Perform Min-Max normalization
    X_fold_train_normalized = scaler.fit_transform(X_fold_train)
    X_fold_val_normalized = scaler.transform(X_fold_val)

    # Use SMOTE to oversample the training data
    smote = SMOTE(random_state=42)
    X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train_normalized, y_fold_train)

    # Initialize and train the XGBoost model
    model = xgb.XGBClassifier(n_estimators=2048,
                              learning_rate=0.01,
                              max_depth=5,
                              objective='binary:logistic',
                              eval_metric=['logloss', 'auc']
                              )
    model.fit(X_fold_train_resampled, y_fold_train_resampled)

    # Make predictions
    y_pred = model.predict(X_test_normalized)

    # Calculate metrics and accuracy for this fold
    PD, PF, balance, FIR, accuracy = classifier_eval(y_test, y_pred)

    # Store metrics for later averaging
    pd_list.append(PD)
    pf_list.append(PF)
    bal_list.append(balance)
    fir_list.append(FIR)
    accuracy_list.append(accuracy)

# Print average metrics across all folds
print('avg_PD: {}'.format((sum(pd_list) / len(pd_list))))
print('avg_PF: {}'.format((sum(pf_list) / len(pf_list))))
print('avg_balance: {}'.format((sum(bal_list) / len(bal_list))))
print('avg_FIR: {}'.format((sum(fir_list) / len(fir_list))))
print('avg_accuracy: {}'.format((sum(accuracy_list) / len(accuracy_list))))