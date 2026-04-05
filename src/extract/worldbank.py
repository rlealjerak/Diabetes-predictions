import requests
import json
import sqlite3
import os 

# constants 
INDICATORS = ['SH.STA.OWAD.ZS', 'NY.GDP.PCAP.PP.CD', 'SP.URB.TOTL.IN.ZS', 'SH.XPD.CHEX.PC.CD', 'SH.XPD.CHEX.GD.ZS', 'SP.POP.65UP.TO.ZS', 'SP.DYN.LE00.IN', 'SP.POP.TOTL'] 
BASE_API_URL = 'https://api.worldbank.org/v2/country/all/indicator/'
RAW_FILE_PATH = 'data/raw/worldbank/' 
DB_PATH = 'db/diabetes_trends.db' 

def extract_worldbank_data(indicator_code): 
    url = f"{BASE_API_URL}{indicator_code}?format=json&per_page=10000" 
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json() 

        # Save rat JSON 
        os.makedirs(RAW_FILE_PATH, exist_ok=True)
        file_path = os.path.join(RAW_FILE_PATH, f'{indicator_code}.json')
        with open(file_path, 'w') as f: 
            json.dump(data, f, indent=2)
        print(f"Saved {file_path}") 

        # Parse data and insert into SQLite
        if isinstance(data, list) and len(data) > 1 and data[1]:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            for item in data[1]:
                cursor.execute(
                    'INSERT OR IGNORE INTO raw_worldbank (indicator_code, country_code, year, value) VALUES (?, ?, ?, ?)',                                       
                    (indicator_code, item['countryiso3code'], item['date'], item['value'])
                )
            conn.commit()
            conn.close()
            print(f"Inserted {len(data[1])} rows for {indicator_code}")
        else:
            print(f"No data found for {indicator_code}")
    else: 
        print(f"Failed to fetch {indicator_code}: HTTP {response.status_code}")

def run(): 
    for code in INDICATORS:
        extract_worldbank_data(code) 

if __name__ == '__main__': 
    run()