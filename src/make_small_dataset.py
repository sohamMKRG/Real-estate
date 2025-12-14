# src/make_small_dataset.py
import pandas as pd

INPUT_PATH = "data/processed_housing.csv"
OUTPUT_PATH = "data/processed_housing_small.csv"

def main():
    df = pd.read_csv(INPUT_PATH)

    # Keep ONLY columns required for app + models
    cols_to_keep = [
        "State",
        "City",
        "Locality",
        "Property_Type",
        "BHK",
        "Size_in_SqFt",
        "Price_in_Lakhs",
        "Price_per_SqFt",
        "Furnished_Status",
        "Floor_No",
        "Total_Floors",
        "Age_of_Property",
        "Nearby_Schools",
        "Nearby_Hospitals",
        "Public_Transport_Accessibility",
        "Parking_Space",
        "Security",
        "Amenities",
        "Facing",
        "Owner_Type",
        "Availability_Status",
        "Good_Investment",
        "Future_Price_5Y",
    ]

    df_small = df[cols_to_keep]

    df_small.to_csv(OUTPUT_PATH, index=False)

    print("✅ Small dataset created:", OUTPUT_PATH)
    print("📦 Rows:", df_small.shape[0])
    print("📊 Columns:", df_small.shape[1])

if __name__ == "__main__":
    main()
