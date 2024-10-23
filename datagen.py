import argparse
import json
import os
import random
import sqlite3
import string
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ColumnDefinition:
    """Represents a column definition from the configuration"""

    column: str
    type: str
    size: int = None
    generator: str = None
    min: int = None
    max: int = None
    query: str = None


class ConfigurationError(Exception):
    """Custom exception for configuration-related errors"""

    pass


def validate_json_model(file_path: str) -> dict:
    """
    Validates that the provided file is a valid JSON file with the expected structure.

    Args:
        file_path: Path to the JSON model file

    Returns:
        Parsed JSON content

    Raises:
        ConfigurationError: If the file is invalid or doesn't match expected structure
    """
    try:
        if not os.path.exists(file_path):
            raise ConfigurationError(f"Model file not found: {file_path}")

        with open(file_path, "r") as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError as e:
                raise ConfigurationError(f"Invalid JSON file: {str(e)}")

        if not isinstance(config, list):
            raise ConfigurationError("JSON model must be a list of column definitions")

        required_fields = {"column", "type", "generator"}
        for idx, col_def in enumerate(config):
            if not isinstance(col_def, dict):
                raise ConfigurationError(
                    f"Column definition {idx + 1} must be an object"
                )

            missing_fields = required_fields - col_def.keys()
            if missing_fields:
                raise ConfigurationError(
                    f"Column definition {idx + 1} missing required fields: {missing_fields}"
                )
        return file_path

    except Exception as e:
        if not isinstance(e, ConfigurationError):
            raise ConfigurationError(f"Error processing model file: {str(e)}")
        raise


class DataGenerator(ABC):
    """Abstract base class for all data generators"""

    @abstractmethod
    def generate(self) -> Any:
        pass


class RandomStringGenerator(DataGenerator):
    """Generates random string values"""

    def __init__(self, size: int, min_length: int = None, max_length: int = None):
        self.size = size
        self.min_length = min_length or 1
        self.max_length = max_length or size

    def generate(self) -> str:
        length = random.randint(self.min_length, self.max_length)
        return "".join(random.choices(string.ascii_letters, k=length))


class RandomIntGenerator(DataGenerator):
    """Generates random integer values"""

    def __init__(self, min_val: int = 0, max_val: int = 100):
        self.min_val = min_val
        self.max_val = max_val

    def generate(self) -> int:
        if self.min_val is None:
            self.min_val = 0
        if self.max_val is None:
            self.max_val = sys.maxsize
        return random.randint(self.min_val, self.max_val)


class QueryBasedGenerator(DataGenerator):
    """Generates values based on database queries"""

    def __init__(self, connection: sqlite3.Connection, query: str):
        self.connection = connection
        self.query = query
        self._cache = None

    def _load_values(self):
        if self._cache is None:
            cursor = self.connection.cursor()
            cursor.execute(self.query)
            self._cache = [row[0] for row in cursor.fetchall()]

    def generate(self) -> Any:
        self._load_values()
        return random.choice(self._cache) if self._cache else None


class DataGeneratorFactory:
    """Factory class to create appropriate generators based on column definitions"""

    def __init__(self, db_connection: sqlite3.Connection = None):
        self.db_connection = db_connection

    def create_generator(self, column_def: ColumnDefinition) -> DataGenerator:
        if column_def.generator == "random":
            if column_def.type == "string":
                return RandomStringGenerator(
                    size=column_def.size,
                    min_length=column_def.min,
                    max_length=column_def.max,
                )
            elif column_def.type == "int":
                return RandomIntGenerator(
                    min_val=column_def.min, max_val=column_def.max
                )
        elif column_def.generator.startswith("select"):
            if self.db_connection is None:
                raise ValueError(
                    "Database connection required for query-based generator"
                )
            return QueryBasedGenerator(self.db_connection, column_def.generator)

        raise ValueError(f"Unsupported generator type: {column_def.generator}")


class DataGeneratorOrchestrator:
    """Main class to orchestrate data generation"""

    def __init__(self, config_file: str, db_connection: sqlite3.Connection = None):
        self.config = self._load_config(config_file)
        self.factory = DataGeneratorFactory(db_connection)
        self.generators = self._setup_generators()

    def _load_config(self, config_file: str) -> List[ColumnDefinition]:
        with open(config_file, "r") as f:
            config_data = json.load(f)

        return [ColumnDefinition(**col_def) for col_def in config_data]

    def _setup_generators(self) -> Dict[str, DataGenerator]:
        return {
            col_def.column: self.factory.create_generator(col_def)
            for col_def in self.config
        }

    def generate_row(self) -> Dict[str, Any]:
        return {
            column: generator.generate()
            for column, generator in self.generators.items()
        }

    def generate_rows(self, count: int) -> List[Dict[str, Any]]:
        return [self.generate_row() for _ in range(count)]


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate random data based on a JSON model definition"
    )
    parser.add_argument("--model", required=True, help="Path to the JSON model file")
    parser.add_argument(
        "--rows", type=int, default=10, help="Number of rows to generate (default: 10)"
    )
    parser.add_argument(
        "--output",
        default="output.csv",
        help="Output CSV file path (default: output.csv)",
    )
    parser.add_argument(
        "--database",
        help="SQLite database file path (required for query-based generators)",
    )

    return parser.parse_args()


def main():
    """Main execution function"""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Validate and load the model file
        config = validate_json_model(args.model)

        # Set up database connection if provided
        db_connection = None
        if args.database:
            try:
                db_connection = sqlite3.connect(args.database)
            except sqlite3.Error as e:
                raise ConfigurationError(f"Failed to connect to database: {str(e)}")

        # Create orchestrator
        orchestrator = DataGeneratorOrchestrator(config, db_connection)

        # Generate the requested number of rows
        rows = orchestrator.generate_rows(args.rows)

        # Write to CSV
        import csv

        with open(args.output, "w", newline="") as f:
            if not rows:
                print("No data generated")
                return

            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"Successfully generated {len(rows)} rows of data to {args.output}")

    except ConfigurationError as e:
        print(f"Configuration error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    # except Exception as e:
    #     print(f"Unexpected error: {str(e)}", file=sys.stderr)
    #     sys.exit(1)
    # finally:
    #     if db_connection:
    #         db_connection.close()


if __name__ == "__main__":
    main()
