import sqlite3
import os

DB_NAME = "temphal.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Create claims table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim TEXT NOT NULL,
            temporal_type TEXT,
            volatility_score REAL,
            ground_truth_date TEXT,
            entity_type TEXT,
            source TEXT,
            split TEXT
        )
    ''')

    # Create response_cache table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS response_cache (
            query_hash TEXT PRIMARY KEY,
            raw_response TEXT,
            claims_json TEXT,
            risk_scores_json TEXT,
            grounded_response TEXT,
            timestamp TEXT
        )
    ''')

    # Create embeddings_cache table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings_cache (
            passage_hash TEXT PRIMARY KEY,
            passage_text TEXT,
            embedding_json TEXT,
            retrieved_for TEXT
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Database {DB_NAME} initialized successfully.")

if __name__ == "__main__":
    init_db()
