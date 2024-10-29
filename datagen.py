import argparse
import datetime
import json
import os
import random
import string
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import psycopg2
from faker import Faker


@dataclass
class ColumnDefinition:
    """Represents a column definition from the configuration"""

    column: str
    type: str
    size: int = None
    generator: str = None
    faker_method: str = None  # e.g. "name_female", "email", "address"
    min: int = None
    max: int = None
    query: str = None
    sql: str = None  # SQL query for db-lookup generator
    value: Any = None  # Static value for static generator
    values: List[Any] = None  # List of values for list generator
    reference_column: str = None  # For lookup generator to reference another column's value


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

            # Validate Faker configuration
            if col_def.get("generator") == "faker":
                if "faker_method" not in col_def:
                    raise ConfigurationError(
                        f"Column definition {idx + 1} is missing required 'faker_method' for faker generator"
                    )
                # Verify the faker method exists
                faker = Faker()
                if not hasattr(faker, col_def["faker_method"]):
                    raise ConfigurationError(
                        f"Column definition {idx + 1} specifies invalid faker_method: {col_def['faker_method']}"
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

    def __init__(self, size: int = 10, min_length: int = None, max_length: int = None):
        self.size = size or 10  # Default to 10 if size is None
        self.min_length = min_length if min_length is not None else 1
        self.max_length = max_length if max_length is not None else self.size

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


class RandomFloatGenerator(DataGenerator):
    """Generates random float values"""

    def __init__(self, min_val: float = 0.0, max_val: float = 100.0):
        self.min_val = min_val
        self.max_val = max_val

    def generate(self) -> float:
        if self.min_val is None:
            self.min_val = 0.0
        if self.max_val is None:
            self.max_val = sys.float_info.max
        return random.uniform(self.min_val, self.max_val)


class RandomDateTimeGenerator(DataGenerator):
    """Generates random datetime values between specified years"""

    def __init__(self, min_year: int = None, max_year: int = None):
        current_year = datetime.datetime.now().year
        self.start_date = datetime.datetime(
            min_year if min_year else current_year - 1, 1, 1
        )
        self.end_date = datetime.datetime(
            max_year if max_year else current_year, 12, 31, 23, 59, 59
        )

    def generate(self) -> datetime.datetime:
        time_between_dates = self.end_date - self.start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        random_date = self.start_date + datetime.timedelta(days=random_number_of_days)
        return random_date


class FakerGenerator(DataGenerator):
    """Generates fake data using the Faker library"""

    def __init__(self, faker_method: str = "text"):
        self.faker = Faker()
        self.faker_method = faker_method

    def generate(self) -> str:
        if not self.faker_method:
            return self.faker.text()

        faker_function = getattr(self.faker, self.faker_method, None)
        if faker_function is None:
            raise ValueError(f"Invalid Faker method: {self.faker_method}")

        return faker_function()


class ListGenerator(DataGenerator):
    """Generates random values from a predefined list"""

    def __init__(self, values: List[Any]):
        if not values:
            raise ValueError("List generator requires non-empty values list")
        self.values = values

    def generate(self) -> Any:
        return random.choice(self.values)


class ExpressionGenerator(DataGenerator):
    """Generates values by combining values from other columns using an expression template"""

    def __init__(self, expression: str, generators: Dict[str, DataGenerator]):
        self.expression = expression
        self.generators = generators
        self._validate_expression()

    def _validate_expression(self):
        # Extract column names from the expression (anything between {} brackets)
        import re
        column_names = re.findall(r'{(\w+)}', self.expression)
        
        # Verify all referenced columns exist
        for column in column_names:
            if column not in self.generators:
                raise ConfigurationError(f"Referenced column '{column}' not found in expression: {self.expression}")

    def generate(self) -> str:
        # Generate a value for each referenced column
        values = {}
        import re
        for column in re.findall(r'{(\w+)}', self.expression):
            values[column] = str(self.generators[column].generate())
        
        # Replace the placeholders with actual values
        result = self.expression
        for column, value in values.items():
            result = result.replace(f'{{{column}}}', value)
        
        return result


class LookupGenerator(DataGenerator):
    """Generates values by looking up from another column's db-lookup generator"""

    def __init__(self, reference_column: str, generators: Dict[str, DataGenerator]):
        self.reference_column = reference_column
        self.generators = generators
        self._validate_reference()
        self._cache = {}

    def _validate_reference(self):
        # Parse reference in format "column.field"
        try:
            column, field = self.reference_column.split('.')
        except ValueError:
            raise ConfigurationError(f"Invalid reference_column format: {self.reference_column}. Expected format: column.field")

        if column not in self.generators:
            raise ConfigurationError(f"Referenced column '{column}' not found")
        
        referenced_generator = self.generators[column]
        if not isinstance(referenced_generator, QueryBasedGenerator):
            raise ConfigurationError(f"Referenced column '{column}' must be a db-lookup generator")

    def generate(self) -> Any:
        column = self.reference_column.split('.')[0]
        field = self.reference_column.split('.')[1]
        
        # Get the referenced generator
        referenced_generator = self.generators[column]
        
        # Get the full row data from the referenced generator
        if column not in self._cache:
            cursor = referenced_generator.connection.cursor()
            cursor.execute(referenced_generator.query)
            rows = cursor.fetchall()
            cursor.close()
            
            # Get column names from cursor description
            column_names = [desc[0] for desc in cursor.description]
            
            # Find the index of our desired field
            try:
                field_index = column_names.index(field)
            except ValueError:
                raise ConfigurationError(f"Field '{field}' not found in query results for column '{column}'")
            
            self._cache[column] = [row[field_index] for row in rows]
            
        # Return a random value from the cached field values
        return random.choice(self._cache[column])


class StaticGenerator(DataGenerator):
    """Generates a static value"""

    def __init__(self, value: Any):
        self.value = value

    def generate(self) -> Any:
        return self.value


class QueryBasedGenerator(DataGenerator):
    """Generates values based on database queries"""

    def __init__(self, connection, query: str):
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

    def __init__(self, db_connection=None, generators=None):
        self.db_connection = db_connection
        self.generators = generators

    def create_generator(self, column_def: ColumnDefinition) -> DataGenerator:
        if column_def.generator == "faker":
            return FakerGenerator(faker_method=column_def.faker_method)
        elif column_def.generator == "random":
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
            elif column_def.type == "float":
                return RandomFloatGenerator(
                    min_val=column_def.min, max_val=column_def.max
                )
            elif column_def.type == "datetime":
                return RandomDateTimeGenerator(
                    min_year=column_def.min, max_year=column_def.max
                )
        elif column_def.generator == "db-lookup":

            if not column_def.sql:
                raise ValueError(
                    f"SQL query required for db-lookup generator. Invalid column: {column_def.column}"
                )
            if self.db_connection is None:
                raise ValueError("Database connection required for db-lookup generator")
            return QueryBasedGenerator(self.db_connection, column_def.sql)
        elif column_def.generator.startswith("select"):
            if self.db_connection is None:
                raise ValueError(
                    "Database connection required for query-based generator"
                )
            return QueryBasedGenerator(self.db_connection, column_def.generator)
        elif column_def.generator == "static":
            return StaticGenerator(column_def.value)
        elif column_def.generator == "list":
            if not column_def.values:
                raise ValueError(f"Values list required for list generator. Invalid column: {column_def.column}")
            return ListGenerator(column_def.values)
        elif column_def.generator == "lookup":
            if not column_def.reference_column:
                raise ValueError(f"reference_column required for lookup generator. Invalid column: {column_def.column}")
            return LookupGenerator(column_def.reference_column, self.generators)
        elif column_def.generator == "expression":
            if not column_def.value:
                raise ValueError(f"value (expression template) required for expression generator. Invalid column: {column_def.column}")
            return ExpressionGenerator(column_def.value, self.generators)

        raise ValueError(f"Unsupported generator type: {column_def.generator}")


class DataGeneratorOrchestrator:
    """Main class to orchestrate data generation"""

    def __init__(self, config_file: str, db_config_file: str = "db_config.json"):
        self.config = self._load_config(config_file)
        self.db_config = self._load_db_config(db_config_file)
        self.db_connection = self._create_db_connection()
        self.generators = {}  # Initialize empty dict
        self._setup_non_lookup_generators()  # First setup all non-lookup generators
        self._setup_lookup_generators()  # Then setup lookup generators

    def _load_db_config(self, config_file: str) -> dict:
        if not os.path.exists(config_file):
            return None
        with open(config_file, "r") as f:
            return json.load(f)

    def _create_db_connection(self):
        if not self.db_config:
            return None
        try:
            return psycopg2.connect(
                host=self.db_config["database"]["host"],
                port=self.db_config["database"]["port"],
                dbname=self.db_config["database"]["name"],
                user=self.db_config["database"]["user"],
                password=self.db_config["database"]["password"],
            )
        except psycopg2.Error as e:
            raise ConfigurationError(f"Failed to connect to database: {str(e)}")

    def _load_config(self, config_file: str) -> List[ColumnDefinition]:
        with open(config_file, "r") as f:
            config_data = json.load(f)

        return [ColumnDefinition(**col_def) for col_def in config_data]

    def _setup_non_lookup_generators(self):
        """First pass: setup all generators except lookup type"""
        factory = DataGeneratorFactory(self.db_connection, self.generators)
        for col_def in self.config:
            if col_def.generator != "lookup":
                print(f"Setting up generator for column: {col_def.column} (type: {col_def.generator})")
                self.generators[col_def.column] = factory.create_generator(col_def)

    def _setup_lookup_generators(self):
        """Second pass: setup lookup generators now that other generators exist"""
        factory = DataGeneratorFactory(self.db_connection, self.generators)
        for col_def in self.config:
            if col_def.generator == "lookup":
                print(f"Setting up generator for column: {col_def.column} (type: {col_def.generator})")
                self.generators[col_def.column] = factory.create_generator(col_def)

    def generate_row(self) -> Dict[str, Any]:
        # First get all column names from config to maintain order
        row = {col_def.column: None for col_def in self.config}
        
        # Then generate values for each column
        for column, generator in self.generators.items():
            value = generator.generate()
            print(f"Generated value for {column}: {value}")
            row[column] = value
            
        return row

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

    return parser.parse_args()


def main():
    """Main execution function"""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Validate and load the model file
        config = validate_json_model(args.model)

        # Set up database connection if provided
        # Create orchestrator with default db_config.json
        orchestrator = DataGeneratorOrchestrator(config)

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
