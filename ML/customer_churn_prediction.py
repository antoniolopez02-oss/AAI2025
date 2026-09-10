"""Predict whether a telecommunications customer is likely to churn."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Data source: IBM watsonx-ai-samples, Telco Customer Churn dataset.
# https://github.com/IBM/watsonx-ai-samples/blob/master/cloud/data/customer_churn/WA_FnUseC_TelcoCustomerChurn.csv
# The dataset contains 7,043 customer records and includes service history,
# monthly charges, contract details, payment method, and churn status.
DATA_FILE = Path(__file__).with_name("telco_customer_churn.csv")
RANDOM_STATE = 42
CHURN_THRESHOLD = 0.50

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod"]
TARGET = "Churn"
REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def load_churn_data(file_path=DATA_FILE):
    """Load and clean the customer churn dataset."""
    data = pd.read_csv(file_path)

    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"The dataset is missing these columns: {missing_text}")

    data = data[REQUIRED_COLUMNS].copy()
    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data[TARGET] = data[TARGET].map({"Yes": 1, "No": 0})
    data = data.dropna()

    if len(data) < 100:
        raise ValueError("The assignment requires at least 100 valid records.")
    if data[TARGET].nunique() != 2:
        raise ValueError("The churn column must contain both churn classes.")

    return data


def build_model():
    """Create a pipeline that prepares the data and trains logistic regression."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def readable_feature_name(feature_name):
    """Remove pipeline prefixes to make coefficient names easier to read."""
    return feature_name.replace("numeric__", "").replace("categorical__", "")


def print_coefficient_explanations(model):
    """Print coefficients and explain what their signs mean."""
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    coefficient_pairs = [
        (readable_feature_name(name), coefficient)
        for name, coefficient in zip(feature_names, coefficients)
    ]
    coefficient_pairs.sort(key=lambda item: abs(item[1]), reverse=True)

    print("\nModel coefficients:")
    for feature, coefficient in coefficient_pairs:
        print(f"  {feature}: {coefficient:,.3f}")

    print("\nCoefficient explanation:")
    print("  Positive coefficients increase predicted churn likelihood.")
    print("  Negative coefficients decrease predicted churn likelihood.")
    print(
        "  Numerical features are scaled, so their coefficients describe a "
        "one-standard-deviation change rather than a one-unit change."
    )
    print(
        "  Categorical coefficients compare each displayed category with the "
        "category dropped by OneHotEncoder."
    )


def main():
    """Train, evaluate, and use the customer churn model."""
    data = load_churn_data()
    print(f"Valid customer records loaded: {len(data):,}")
    print(f"Observed churn rate: {data[TARGET].mean():.1%}")

    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = build_model()
    model.fit(X_train, y_train)

    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (test_probabilities >= CHURN_THRESHOLD).astype(int)
    accuracy = accuracy_score(y_test, test_predictions)
    roc_auc = roc_auc_score(y_test, test_probabilities)

    new_customer = pd.DataFrame(
        {
            "tenure": [12],
            "MonthlyCharges": [85.00],
            "TotalCharges": [1020.00],
            "Contract": ["Month-to-month"],
            "InternetService": ["Fiber optic"],
            "PaymentMethod": ["Electronic check"],
        }
    )

    churn_probability = model.predict_proba(new_customer)[0, 1]
    churn_prediction = int(churn_probability >= CHURN_THRESHOLD)

    print(f"Test accuracy at the 0.50 threshold: {accuracy:.1%}")
    print(f"Test ROC AUC: {roc_auc:.3f}")
    print(f"\nNew customer churn probability: {churn_probability:.1%}")
    print(f"Churn prediction (1 = at risk, 0 = not at risk): {churn_prediction}")
    print(
        f"Interpretation: The model estimates a {churn_probability:.1%} chance "
        "that this customer will leave."
    )

    if churn_prediction == 1:
        print(
            "Business use: Flag this customer for a retention offer, service "
            "review, or a more suitable contract before the customer leaves."
        )
    else:
        print(
            "Business use: Continue normal support and monitor the customer for "
            "future increases in churn risk."
        )

    print_coefficient_explanations(model)


if __name__ == "__main__":
    main()
