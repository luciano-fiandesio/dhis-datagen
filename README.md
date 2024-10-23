# DHIS2 Data Generator

This tool generates mass data for the DHIS2 project.

## Database configuration

The database configuration is stored in `db_config.json`.


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
