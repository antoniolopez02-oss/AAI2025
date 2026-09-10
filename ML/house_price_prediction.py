
"""Predict house prices from square footage and broad location."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Data source: OpenML House Prices dataset (Ames, Iowa), data ID 42165.
# https://www.openml.org/search?type=data&status=active&id=42165
# The CSV contains 1,460 real home sales. "square_footage" comes from
# GrLivArea, "price" comes from SalePrice, and the original Ames neighborhoods
# were grouped into the assignment's three broad labels: Downtown, Rural,
# and Suburb.
DATA_FILE = Path(__file__).with_name("ames_housing_prices.csv")
REQUIRED_COLUMNS = ["square_footage", "location", "price"]
RANDOM_STATE = 42


def load_housing_data(file_path=DATA_FILE):
    """Load the housing CSV and check that it is ready for the model."""
    data = pd.read_csv(file_path)

    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"The dataset is missing these columns: {missing_text}")

    data = data[REQUIRED_COLUMNS].copy()
    data["square_footage"] = pd.to_numeric(
        data["square_footage"], errors="coerce"
    )
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna()
    data = data[(data["square_footage"] > 0) & (data["price"] > 0)]

    if len(data) < 100:
        raise ValueError("The assignment requires at least 100 valid records.")
    if "Downtown" not in data["location"].unique():
        raise ValueError("The dataset must include Downtown for the prediction.")

    return data


def build_model():
    """Create a pipeline that encodes location and trains linear regression."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "location",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["location"],
            )
        ],
        remainder="passthrough",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def print_coefficient_explanations(model):
    """Print the model coefficients and explain them in plain language."""
    encoder = model.named_steps["preprocessor"].named_transformers_["location"]
    feature_names = list(encoder.get_feature_names_out(["location"]))
    feature_names.append("square_footage")

    coefficients = model.named_steps["regressor"].coef_
    coefficient_by_feature = dict(zip(feature_names, coefficients))

    print("\nModel coefficients:")
    for feature, coefficient in coefficient_by_feature.items():
        print(f"  {feature}: {coefficient:,.2f}")

    square_footage_effect = coefficient_by_feature["square_footage"]
    print("\nPlain-language explanation:")
    print(
        "  Holding location constant, each additional square foot is "
        f"associated with about ${square_footage_effect:,.2f} in predicted price."
    )

    # Downtown is the baseline because OneHotEncoder drops the first category.
    for location in ("Rural", "Suburb"):
        location_effect = coefficient_by_feature[f"location_{location}"]
        direction = "higher" if location_effect >= 0 else "lower"
        print(
            f"  A similar-size house in {location} is predicted to be "
            f"${abs(location_effect):,.2f} {direction} than one in Downtown."
        )


def main():
    """Load data, train the model, evaluate it, and make the required prediction."""
    data = load_housing_data()
    print(f"Valid housing records loaded: {len(data):,}")

    # Features used to make predictions and the target we want to predict.
    features = data[["square_footage", "location"]]
    target = data["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    model = build_model()
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_predictions)
    r_squared = r2_score(y_test, test_predictions)

    # Required prediction: a 2,000-square-foot house in Downtown.
    new_house = pd.DataFrame(
        {"square_footage": [2000], "location": ["Downtown"]}
    )
    predicted_price = model.predict(new_house)[0]

    print(
        "Predicted price for a 2,000 sq ft house in Downtown: "
        f"${predicted_price:,.2f}"
    )
    print(f"Test mean absolute error: ${mae:,.2f}")
    print(f"Test R-squared: {r_squared:.3f}")

    print_coefficient_explanations(model)


if __name__ == "__main__":
    main()
