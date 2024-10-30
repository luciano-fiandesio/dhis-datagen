# DHIS2 Data Generator

This tool generates mass data for the DHIS2 project.

## Database configuration

A database connection is required to generate data.

The database configuration is stored in `db_config.json`.

Example:

```json
{
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "db",
        "user": "demo",
        "password": "demo"
    }
}
```

## Usage

```bash
poetry install
poetry run python datagen.py --model events.json --rows 1000
```

## Generate a model file from an existing table

```bash
poetry run python generate_table_config.py {table-name}
```

e.g.

```bash
poetry run python generate_table_config.py analytics_enrollment_ezkn8vyzwjr
```

## Generator types

### Static

Generates a static value.

Example:

```json
{
    "column": "year",
    "generator": "static",
    "type": "int",
    "value": 2024
}
```

### Random

Generates a random value.

```json
{
    "column": "year",
    "generator": "random",
    "type": "int",
    "min": 2020,
    "max": 2024
}
```

The generated value is based on the type of the column and the min and max values.

For instance, if the column is of type `int`, the generated value will be an integer between the min and max values.
If the column is of type `string`, the generated value will be a string of length between the min and max values.
If the column is of type `float`, the generated value will be a float between the min and max values.
If the column is of type `datetime`, the generated value will expect a min and max year, and a random date will be generated between the min and max year.

### Faker

Generates a value using a [faker](https://faker.readthedocs.io) method.

Example:

```json
{
    "column": "name",
    "type": "string",
    "generator": "faker",
    "faker_method": "first_name"
}
```

### DB Lookup

Generates a value by looking up a value in the database.

Example:

```json
{
    "column": "name",
    "generator": "db-lookup",
    "sql": "select distinct name from dataelement"
}
```

This generator will cache the values from the lookup query, and select a random  cached value for each row.

### List

Generates a value from a list of predefined values.

Example:

```json
{
    "column": "name",
    "type": "string",
    "generator": "list",
    "values": ["Alice", "Bob", "Charlie"]
}
```

### Lookup

Select a random value from a column in the same table referenced by the `reference_column` property.

Example:

```json
{
    "column": "ou",
    "type": "string",
    "generator": "db-lookup",
    "size": 11,
    "sql": "select distinct organisationunituid, level from analytics_rs_orgunitstructure"
},
{
    "column": "oulevel",
    "type": "int",
    "generator": "lookup",
    "size": 11,
    "reference_column": "ou.level"
}
```

In the above example, the `oulevel` column will select a random value from the `ou.level` column, defined in the `ou` column.

### Expression

Generates a value using a simple expression.

Example:

```json
{
    "column": "id",
    "type": "string",
    "generator": "expression",
    "value": "{dx}-{pe}-{ou}-{co}-{ao}"
}
```

In the above example, the `id` column will be generated using the expression `{dx}-{pe}-{ou}-{co}-{ao}`, which
combines the values from the `dx`, `pe`, `ou`, `co` and `ao` columns.