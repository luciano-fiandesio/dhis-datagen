import argparse
import json
import psycopg2
from typing import Dict, List

def load_db_config(config_file: str = "db_config.json") -> Dict:
    """Load database configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def get_column_info(connection, table_name: str) -> List[Dict]:
    """Extract column information from the database"""
    cursor = connection.cursor()
    
    # Query to get column information
    query = """
        SELECT column_name, data_type, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position;
    """
    
    cursor.execute(query, (table_name,))
    columns = cursor.fetchall()
    
    config = []
    for col in columns:
        column_name, data_type, max_length = col
        
        # Skip geometry columns
        if data_type == 'geometry':
            continue
            
        # Map PostgreSQL types to our generator types
        if data_type in ('character varying', 'text', 'character'):
            generator_type = "string"
            generator = "random"  # Always use random for strings
            config_entry = {
                "column": column_name,
                "type": generator_type,
                "generator": generator
            }
            if max_length:
                config_entry["size"] = max_length
                
        elif data_type in ('integer', 'bigint', 'smallint'):
            config_entry = {
                "column": column_name,
                "type": "int",
                "generator": "random"
            }
            
        elif data_type in ('timestamp', 'timestamp without time zone', 'timestamp with time zone'):
            config_entry = {
                "column": column_name,
                "type": "datetime",
                "generator": "random"
            }
            
        elif data_type in ('double precision', 'real', 'numeric'):
            config_entry = {
                "column": column_name,
                "type": "float",
                "generator": "random"
            }
            
        else:
            # Default to string type for unknown types
            config_entry = {
                "column": column_name,
                "type": "string",
                "generator": "random",
                "size": 100
            }
            
        config.append(config_entry)
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Generate table configuration from database schema")
    parser.add_argument("table", help="Name of the table to analyze")
    parser.add_argument("--output", default=None, help="Output JSON file (default: <table>_config.json)")
    args = parser.parse_args()
    
    # Set default output filename if not provided
    if not args.output:
        args.output = f"{args.table}_config.json"
    
    try:
        # Load database configuration
        db_config = load_db_config()
        
        # Connect to database
        conn = psycopg2.connect(
            host=db_config['database']['host'],
            port=db_config['database']['port'],
            dbname=db_config['database']['name'],
            user=db_config['database']['user'],
            password=db_config['database']['password']
        )
        
        # Get column information and generate config
        config = get_column_info(conn, args.table)
        
        # Write configuration to file
        with open(args.output, 'w') as f:
            json.dump(config, f, indent=4)
            
        print(f"Configuration file generated: {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
