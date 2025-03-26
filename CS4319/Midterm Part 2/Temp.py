import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load datasets
train_file_path = "train_data.csv"
test_file_path = "test_data.csv"

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Extract features and labels
X_train = train_df.drop(columns=["class"])
y_train = train_df["class"]
X_test = test_df.drop(columns=["class"])
y_test = test_df["class"]

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    "MLP": MLPClassifier(max_iter=100000, random_state=81),
    "Logistic Regression": LogisticRegression(max_iter=100000, random_state=81),
    "KNN": KNeighborsClassifier()
}

# Perform hyperparameter tuning using 5-fold cross-validation
param_grids = {
    "MLP": {"hidden_layer_sizes": [(50, 50), (100, 50), (50, 100), (100, 100)]},
    "Logistic Regression": {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
    "KNN": {"n_neighbors": range(1, 21)}
}

tuned_models = {}
for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5)
    grid_search.fit(X_train_scaled, y_train)
    tuned_models[model_name] = grid_search.best_estimator_

# Perform cross-validation for each feature separately
feature_scores = {}

for feature in X_train.columns:
    X_feature = X_train[[feature]]  # Use only one feature
    
    scores = {}
    for model_name, model in tuned_models.items():
        score = np.mean(cross_val_score(model, X_feature, y_train, cv=5))
        scores[model_name] = score
    
    feature_scores[feature] = scores

# Find the best feature for each model
best_features = {model: max(feature_scores, key=lambda f: feature_scores[f][model]) for model in tuned_models}

# Train and evaluate models using the best feature
results_single_feature = {}

for model_name, feature in best_features.items():
    model = tuned_models[model_name]
    model.fit(X_train[[feature]], y_train)
    accuracy = model.score(X_test[[feature]], y_test)
    results_single_feature[model_name] = (feature, accuracy)

# Train and evaluate models using all features
results_all_features = {}

for model_name, model in tuned_models.items():
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    results_all_features[model_name] = accuracy

# Print results along with the number of iterations for applicable models
print("Results using the best single feature:")
for model, (feature, accuracy) in results_single_feature.items():
    print(f"{model}: Best Feature = {feature}, Accuracy = {accuracy:.4f}")
    if hasattr(tuned_models[model], "n_iter_"):
        print(f"  Iterations: {tuned_models[model].n_iter_}")

print("\nResults using all features:")
for model, accuracy in results_all_features.items():
    print(f"{model}: Accuracy = {accuracy:.4f}")
    if hasattr(tuned_models[model], "n_iter_"):
        print(f"  Iterations: {tuned_models[model].n_iter_}")
