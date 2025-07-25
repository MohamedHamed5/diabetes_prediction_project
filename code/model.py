

# Project Name: Diabetes Prediction Model
# Date: 7/18/25
# Created By: Al-Arif Team
# - Ahmed Mohamed Ahmed Ali
# - Momen Ahmed Ahmed Elenany
# - Mohamed Ahmed AlSayed Hamed
# - Felopater Magdy Helmy
# - Mohamed Mostafa Ahmed Mohammed
# - Mariam Mohey Ibrahiem Arafa 

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score ,roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from imblearn.over_sampling import SMOTE 
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs('figures', exist_ok=True)

# Load the dataset
data = pd.read_csv(r'E:\diabetes_prediction_project\diabetes_012_health_indicators_BRFSS2015.csv')
print(data.shape)
print(data.info()) # no nulls, all cols are numbers
print(data.describe())

# Separate features and target
df = data.copy()
X = df.drop('Diabetes_012', axis=1)
y = df['Diabetes_012']

# Calculate the Correlation Matrix
numeric_df = df.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_df.corr()

# Draw a heatmap for all numerical cols
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title("Heatmap of Feature Correlations")
plt.savefig('figures/correlation_heatmap.png')
plt.show()
target_corr = correlation_matrix['Diabetes_012'].sort_values(ascending=False)

# Draw a Heatmap for Correlation with the Target Variable Only
plt.figure(figsize=(8, 10))
sns.heatmap(target_corr.to_frame(), annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title("Correlation of Features with diabetes")
plt.savefig('figures/heatmap_Target.png')
plt.show()

# Draw a Boxpolt for some features
features = ["BMI", "MentHlth", "PhysHlth"]
outlier_info = {}

for feature in features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[feature])
    plt.title(f"Boxplot of {feature}")
    plt.ylabel(feature)
    plt.savefig(f'figures/boxplot_{feature.lower()}.png')
    plt.show()
    
    # Calculate Q1, Q3, and IQR
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)][feature]

    outlier_info[feature] = {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "num_outliers": len(outliers),
        "outlier_values_sample": outliers.unique().tolist() if len(outliers.unique()) < 10 else outliers.sample(min(10, len(outliers))).tolist()
    }

# Print outlier info
print("Outlier Analysis:")
for feature, info in outlier_info.items():
    print(f"\n--- {feature} ---")
    print(f"Q1: {info['Q1']}")
    print(f"Q3: {info['Q3']}")
    print(f"IQR: {info['IQR']}")
    print(f"Lower Bound: {info['lower_bound']}")
    print(f"Upper Bound: {info['upper_bound']}")
    print(f"Number of Outliers: {info['num_outliers']}")
    print(f"Sample Outlier Values: {info['outlier_values_sample']}")

# Draw The Histogram for some features
for col in features:
    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x=col, bins=30, kde=True, color='skyblue')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figures/histogram_{col.lower()}.png')
    plt.show()

'''
(Synthetic Minority Over-sampling Technique) from imblearn:
A method used to create (new) synthetic samples of the minority class, 
with the aim of balancing imbalanced data before training the model
'''

# Split before SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE on training data only
smote = SMOTE(
    sampling_strategy='auto',
    random_state=42,
    k_neighbors=3,
)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print("Class distribution after SMOTE:\n", pd.Series(y_train_balanced).value_counts())

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train_balanced, y_train_balanced)

# Predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

# Accuaracy
print(f'Model accuracy score with 100 decision-trees {accuracy_score(y_test, y_pred):.5f}')

# Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# One-hot encode y_test for ROC AUC
label_binarizer = LabelBinarizer()
y_test_binarized = label_binarizer.fit_transform(y_test)

# ROC AUC score
roc_auc_ovr = roc_auc_score(y_test_binarized, y_proba, multi_class='ovr', average='weighted')
print(f"\nROC AUC (One-vs-Rest, Weighted): {roc_auc_ovr:.4f}")

# Plot ROC curves
plt.figure(figsize=(10, 8))
for i in range(len(label_binarizer.classes_)):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_proba[:, i])
    plt.plot(fpr, tpr, label=f'Class {label_binarizer.classes_[i]} (AUC = {roc_auc_score(y_test_binarized[:, i], y_proba[:, i]):.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression (SMOTE applied)')
plt.legend()
plt.grid(True)
plt.savefig('figures/roc_curve_with_smote.png')
print("ROC curve saved as roc_curve_with_smote.png")
plt.close()

# save Scaler
import joblib
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

