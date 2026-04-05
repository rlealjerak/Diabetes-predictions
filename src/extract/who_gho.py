import requests
import json
import sqlite3
import os

# constants
INDICATORS = ['NCD_DIABETES_PREVALENCE_AGESTD', 'NCD_BMI_MEAN', 'NCD_GLUC_04', 'NCD_PAA', 'NCDMORT3070']
BASE_API_URL = 'https://ghoapi.azureedge.net/api'
DB_PATH = 'db/diabetes_trends.db'
RAW_FILE_PATH = 'data/raw/who/'


def extract_who_data(indicator_code):
    url = f"{BASE_API_URL}/{indicator_code}?$filter=Dim1 eq 'SEX_BTSX'"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        # save raw JSON to data/raw/who/{indicator_code}.json
        os.makedirs(RAW_FILE_PATH, exist_ok=True)
        file_path = os.path.join(RAW_FILE_PATH, f'{indicator_code}.json')
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {file_path}")

        # parse .value array and insert into SQLite
        if 'value' in data and data['value']:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            for item in data['value']:
                cursor.execute(
                    'INSERT OR IGNORE INTO raw_who_gho (indicator_code, spatial_dim, time_dim, dim1, numeric_value, low, high) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (indicator_code, item['SpatialDim'], item['TimeDim'], item['Dim1'], item['NumericValue'], item['Low'], item['High'])
                )
            conn.commit()
            conn.close()
            print(f"Inserted {len(data['value'])} rows for {indicator_code}")
        else:
            print(f"No data found for {indicator_code}")
    else:
        print(f"Failed to fetch {indicator_code}: HTTP {response.status_code}")

def run():
    for code in INDICATORS:
        extract_who_data(code)


if __name__ == '__main__':
    run()
