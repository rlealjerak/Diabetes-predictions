
CREATE TABLE IF NOT EXISTS raw_who_gho (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator_code TEXT NOT NULL,
    spatial_dim    TEXT NOT NULL,
    time_dim       INTEGER NOT NULL,
    dim1           TEXT,
    numeric_value  REAL,
    low            REAL,
    high           REAL,
    ingested_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS raw_worldbank ( 
    id             INTEGER PRIMARY KEY AUTOINCREMENT,                                                                                                    
      indicator_code TEXT NOT NULL,                                                                                                                        
      country_code   TEXT NOT NULL,                                                                                                                        
      year           INTEGER NOT NULL,                                                                                                                     
      value          REAL,
      ingested_at    TEXT DEFAULT (datetime('now'))
); 