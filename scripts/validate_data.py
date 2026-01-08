from pc_price_forecast.io.read_csv import read_raw_csv
from pc_price_forecast.data.validate import validate_raw_df

def main():
    df = read_raw_csv()
    validate_raw_df(df)
    print("✅ Raw data validation passed.")

if __name__ == "__main__":
    main()
