import sqlite3
import json
import hashlib
import time
import os

class DB:
    def __init__(self, path='temphal.db'):
        self.path = path
        self.con = sqlite3.connect(path, check_same_thread=False)
        self._init()

    def _init(self):
        # Create tables as defined in phase 3 details
        self.con.executescript('''
            CREATE TABLE IF NOT EXISTS claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim TEXT,
                temporal_type TEXT,
                volatility_score REAL,
                entity_type TEXT,
                ground_truth_date TEXT,
                source TEXT,
                split TEXT
            );
            
            CREATE TABLE IF NOT EXISTS response_cache (
                query_hash TEXT PRIMARY KEY,
                raw_response TEXT,
                claims_json TEXT,
                grounded_json TEXT,
                ts REAL
            );
            
            CREATE TABLE IF NOT EXISTS embeddings_cache (
                passage_hash TEXT PRIMARY KEY,
                passage_text TEXT,
                embedding_json TEXT
            );
        ''')
        
        # Comprehensive Migration: Ensure column names are consistent
        cursor = self.con.cursor()
        
        # Check columns in response_cache
        cursor.execute("PRAGMA table_info(response_cache)")
        cols = {row[1] for row in cursor.fetchall()}
        
        # Migrate 'timestamp' to 'ts'
        if 'timestamp' in cols and 'ts' not in cols:
            print("Migrating: 'timestamp' -> 'ts'...")
            self.con.execute("ALTER TABLE response_cache RENAME COLUMN timestamp TO ts")
            self.con.commit()
            
        # Migrate 'grounded_response' to 'grounded_json'
        if 'grounded_response' in cols and 'grounded_json' not in cols:
            print("Migrating: 'grounded_response' -> 'grounded_json'...")
            self.con.execute("ALTER TABLE response_cache RENAME COLUMN grounded_response TO grounded_json")
            self.con.commit()
            
        # Add 'grounded_json' if it's missing (and no rename was possible)
        if 'grounded_json' not in cols and 'grounded_response' not in cols:
            print("Migrating: Adding missing 'grounded_json'...")
            self.con.execute("ALTER TABLE response_cache ADD COLUMN grounded_json TEXT")
            self.con.commit()
            
        # Add 'ts' if it's missing (and no rename was possible)
        if 'ts' not in cols and 'timestamp' not in cols:
            print("Migrating: Adding missing 'ts'...")
            self.con.execute("ALTER TABLE response_cache ADD COLUMN ts REAL")
            self.con.commit()
            
        self.con.commit()

    def cache_get(self, query: str):
        """
        Retrieves a cached result if it exists and is less than 24 hours old.
        """
        h = hashlib.sha256(query.encode()).hexdigest()
        # 86400 seconds = 24 hours
        row = self.con.execute(
            'SELECT grounded_json FROM response_cache WHERE query_hash=? AND ts>?',
            (h, time.time() - 86400)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def cache_set(self, query: str, result: dict):
        """
        Stores a result in the cache using explicit column names.
        """
        h = hashlib.sha256(query.encode()).hexdigest()
        self.con.execute(
            '''INSERT OR REPLACE INTO response_cache 
               (query_hash, raw_response, claims_json, grounded_json, ts)
               VALUES (?, ?, ?, ?, ?)''',
            (h, result.get('raw', ''), 
             json.dumps(result.get('claims', [])), 
             json.dumps(result), 
             time.time())
        )
        self.con.commit()

    def close(self):
        self.con.close()
