from SPARQLWrapper import SPARQLWrapper, JSON
import json
import time
import os
import sqlite3
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

DB_NAME = "temphal.db"

def get_wikidata_sparql():
    sparql = SPARQLWrapper('https://query.wikidata.org/sparql')
    sparql.addCustomHttpHeader('User-Agent', 'TempHal-Research/1.0 (student project)')
    sparql.setReturnFormat(JSON)
    return sparql

def extract_volatile_claims(limit_per_query=1000, total_limit=3000):
    """Extracts volatile claims (CEOs) from WikiData."""
    sparql = get_wikidata_sparql()
    
    QUERY = '''
    SELECT ?item ?itemLabel ?ceo ?ceoLabel ?start ?end WHERE {{
      ?item p:P169 ?stmt .
      ?stmt ps:P169 ?ceo .
      OPTIONAL {{ ?stmt pq:P580 ?start . }}
      OPTIONAL {{ ?stmt pq:P582 ?end   . }}
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language 'en'. }}
    }} LIMIT {limit} OFFSET {offset}
    '''
    
    volatile_claims = []
    for offset in range(0, total_limit, limit_per_query):
        print(f"Fetching volatile claims (offset {offset})...")
        sparql.setQuery(QUERY.format(limit=limit_per_query, offset=offset))
        try:
            res = sparql.query().convert()
            for r in res['results']['bindings']:
                item_label = r['itemLabel']['value']
                ceo_label = r['ceoLabel']['value']
                # Skip if labels are same or generic
                if item_label == ceo_label or "Q" in item_label:
                    continue
                
                row = {
                    'claim': f"{ceo_label} is the CEO of {item_label}",
                    'temporal_type': 'volatile',
                    'entity_type': 'organization',
                    'ground_truth_date': r.get('start', {}).get('value', ''),
                    'source': 'wikidata_p169',
                    'volatility_score': 0.8
                }
                volatile_claims.append(row)
        except Exception as e:
            print(f'Offset {offset} failed: {e}')
        time.sleep(2)
    
    print(f"Extracted {len(volatile_claims)} volatile claims.")
    return volatile_claims

def extract_stable_claims(limit_per_query=500, total_limit=1000):
    """Extracts stable claims (Birth dates of historical figures) from WikiData."""
    sparql = get_wikidata_sparql()
    
    # Birth dates of people who died before 2020
    QUERY = '''
    SELECT ?item ?itemLabel ?birth ?death WHERE {{
      ?item wdt:P31 wd:Q5 . # human
      ?item wdt:P569 ?birth .
      ?item wdt:P570 ?death .
      FILTER(?death < "2020-01-01T00:00:00Z"^^xsd:dateTime)
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language 'en'. }}
    }} LIMIT {limit} OFFSET {offset}
    '''
    
    stable_claims = []
    for offset in range(0, total_limit, limit_per_query):
        print(f"Fetching stable claims (offset {offset})...")
        sparql.setQuery(QUERY.format(limit=limit_per_query, offset=offset))
        try:
            res = sparql.query().convert()
            for r in res['results']['bindings']:
                item_label = r['itemLabel']['value']
                date_val = r['birth']['value']
                
                # Format claim
                year = date_val.split('-')[0].replace('+', '')
                row = {
                    'claim': f"{item_label} was born in {year}",
                    'temporal_type': 'stable',
                    'entity_type': 'person',
                    'ground_truth_date': date_val,
                    'source': 'wikidata_p569',
                    'volatility_score': 0.1
                }
                stable_claims.append(row)
        except Exception as e:
            print(f'Offset {offset} failed: {e}')
        time.sleep(2)
        
    print(f"Extracted {len(stable_claims)} stable claims.")
    return stable_claims

def load_external_data():
    """Loads claims from TruthfulQA."""
    print("Loading TruthfulQA dataset...")
    try:
        tqa = load_dataset('truthful_qa', 'generation', split='validation')
        uncertain_keywords = ['current', 'now', 'today', 'presently', 'latest', 'moment', 'who is', 'what is']
        tqa_claims = [
          {'claim': r['best_answer'], 
           'temporal_type': 'uncertain', 
           'source': 'truthfulqa',
           'volatility_score': 0.5} 
          for r in tqa 
          if any(kw in r['question'].lower() for kw in uncertain_keywords)
        ]
    except Exception as e:
        print(f"Error loading TruthfulQA: {e}")
        tqa_claims = []
    
    # Skipping FEVER due to compatibility issues with newer 'datasets' versions
    fever_claims = []
    
    print(f"Loaded {len(fever_claims)} FEVER claims (skipped) and {len(tqa_claims)} TruthfulQA claims.")
    return fever_claims, tqa_claims

def finalize_and_store(all_claims):
    """Splits and stores the claims in the SQLite database."""
    df = pd.DataFrame(all_claims)
    
    # Ensure we have enough data for the targets
    # Target distribution: 40% stable, 40% volatile, 20% uncertain
    # Total ~2000
    
    volatile_df = df[df['temporal_type'] == 'volatile'].sample(min(800, len(df[df['temporal_type'] == 'volatile'])))
    stable_df = df[df['temporal_type'] == 'stable'].sample(min(800, len(df[df['temporal_type'] == 'stable'])))
    uncertain_df = df[df['temporal_type'] == 'uncertain'].sample(min(400, len(df[df['temporal_type'] == 'uncertain'])))
    
    final_df = pd.concat([volatile_df, stable_df, uncertain_df])
    
    print(f"Final dataset size: {len(final_df)}")
    print(final_df['temporal_type'].value_counts())
    
    # Stratified split
    train_df, test_val_df = train_test_split(
        final_df, test_size=0.4, stratify=final_df['temporal_type'], random_state=42
    )
    val_df, test_df = train_test_split(
        test_val_df, test_size=0.5, stratify=test_val_df['temporal_type'], random_state=42
    )
    
    train_df['split'] = 'train'
    val_df['split'] = 'validation'
    test_df['split'] = 'test'
    
    final_with_splits = pd.concat([train_df, val_df, test_df])
    
    # Store in SQLite
    conn = sqlite3.connect(DB_NAME)
    final_with_splits.to_sql('claims', conn, if_exists='append', index=False)
    conn.close()
    print("Claims stored in SQLite successfully.")

def main():
    # 1. WikiData Volatile
    volatile = extract_volatile_claims(total_limit=3000)
    
    # 2. WikiData Stable
    stable_wd = extract_stable_claims(total_limit=2000)
    
    # 3. External Data
    fever, tqa = load_external_data()
    
    # Combine
    all_claims = volatile + stable_wd + fever + tqa
    
    # 4. Finalize and store
    finalize_and_store(all_claims)

if __name__ == "__main__":
    main()
