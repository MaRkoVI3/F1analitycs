import urllib.request
import json
import os
import time

def fetch_season_data(year, data_type='results'):
    """
    Fetch data from Ergast API (via Jolpi mirror) for a given year and data type (results or qualifying).
    Returns the parsed JSON data.
    """
    # Use Jolpi mirror of Ergast API to avoid potential blocks
    url = f"https://api.jolpi.ca/ergast/f1/{year}/{data_type}.json?limit=1000"
    # Set a User-Agent to avoid 403 Forbidden errors
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                data = json.load(response)
                return data
            else:
                print(f"Error fetching {data_type} for {year}: HTTP {response.status}")
                return None
    except Exception as e:
        print(f"Exception fetching {data_type} for {year}: {e}")
        return None

def save_data(data, year, data_type):
    """
    Save the fetched data to a JSON file in the data/raw directory.
    """
    # Create directory if it doesn't exist
    raw_dir = "/home/bobo/Projects/f1_predictor/data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    filename = f"{year}_{data_type}.json"
    filepath = os.path.join(raw_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {data_type} data for {year} to {filepath}")

def main():
    """
    Main function to fetch data for years 2021-2026 for both results and qualifying.
    For 2026, only races that have occurred up to the current date are considered.
    """
    years = list(range(2021, 2027))  # 2021 to 2026 inclusive
    data_types = ['results', 'qualifying']

    for year in years:
        for data_type in data_types:
            print(f"Fetching {data_type} for {year}...")
            data = fetch_season_data(year, data_type)
            if data:
                save_data(data, year, data_type)
            # Be respectful to the API - add a small delay between requests
            time.sleep(0.5)

    print("Data fetching complete.")

if __name__ == "__main__":
    main()