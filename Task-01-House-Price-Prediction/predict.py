"""
Interactive House Price Prediction Tool
Allows user to input house features and get instant price predictions.
"""

import os
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_artifacts():
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
    multi_model_path = os.path.join(BASE_DIR, 'models', 'multiple_linear_regression.pkl')
    simple_model_path = os.path.join(BASE_DIR, 'models', 'simple_linear_regression.pkl')

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(multi_model_path, 'rb') as f:
        multi_model = pickle.load(f)
    with open(simple_model_path, 'rb') as f:
        simple_model = pickle.load(f)

    return scaler, multi_model, simple_model

def get_float_input(prompt, default_val):
    raw = input(f"{prompt} [Default: {default_val}]: ").strip()
    if not raw:
        return default_val
    try:
        return float(raw)
    except ValueError:
        print(f"Invalid input. Using default: {default_val}")
        return default_val

def interactive_session():
    scaler, multi_model, simple_model = load_artifacts()

    print("=" * 65)
    print("      🏠 INTERACTIVE HOUSE PRICE PREDICTOR TOOL 🏠      ")
    print("=" * 65)
    print("Enter the details of the house below to estimate its market price.")
    print("You can press [Enter] to accept the recommended default values.\n")

    while True:
        try:
            sqft = get_float_input("1. Living Area Square Footage (e.g., 800 - 4500)", 2000.0)
            bedrooms = get_float_input("2. Number of Bedrooms (e.g., 1 - 6)", 3.0)
            bathrooms = get_float_input("3. Number of Bathrooms (e.g., 1.0 - 4.5)", 2.0)
            total_rooms = get_float_input("4. Total Rooms Above Ground (e.g., 4 - 12)", max(int(bedrooms) + 3, 5))
            year_built = get_float_input("5. Year Built (e.g., 1960 - 2024)", 2010.0)
            lot_area = get_float_input("6. Lot Area in Sq Ft (e.g., 3000 - 20000)", round(sqft * 4.5))
            garage_cars = get_float_input("7. Garage Capacity (Number of Cars, 0 - 3)", 2.0)
            quality = get_float_input("8. Overall Quality Rating (Scale: 1 to 10)", 7.0)

            # Predict using Simple Linear Regression (Square Footage only)
            simple_pred = simple_model.predict(pd.DataFrame({'SquareFootage': [sqft]}))[0]

            # Predict using Multiple Linear Regression (All features normalized)
            input_df = pd.DataFrame([{
                'SquareFootage': sqft,
                'TotalRooms': int(total_rooms),
                'Bedrooms': int(bedrooms),
                'Bathrooms': bathrooms,
                'YearBuilt': int(year_built),
                'LotArea': lot_area,
                'GarageCars': int(garage_cars),
                'OverallQuality': int(quality)
            }])
            input_scaled = scaler.transform(input_df)
            multi_pred = multi_model.predict(input_scaled)[0]

            print("\n" + "-" * 65)
            print("                 PREDICTION RESULTS                     ")
            print("-" * 65)
            print(f" Property Specs: {sqft:,.0f} sqft | {int(bedrooms)} beds | {bathrooms} baths | Built: {int(year_built)} | Quality: {int(quality)}/10")
            print(f"\n [Model 1] Simple Linear Regression (SqFt only):")
            print(f"           >>> Estimated Price: ${simple_pred:,.2f}")
            print(f"\n [Model 2] Multiple Linear Regression (All Features):")
            print(f"           >>> Estimated Price: ${multi_pred:,.2f}")
            print("-" * 65)

        except Exception as e:
            print(f"An error occurred: {e}")

        again = input("\nWould you like to predict another house? (y/n): ").strip().lower()
        if again not in ['y', 'yes']:
            print("\nThank you for using the House Price Predictor!")
            break
        print("\n" + "=" * 65 + "\n")

if __name__ == '__main__':
    interactive_session()
