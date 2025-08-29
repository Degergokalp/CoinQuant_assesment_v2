
import sys, json
from jsonschema import validate, ValidationError
import importlib.resources as pkg_resources
import os, pathlib

def load_schema():
    schema_path = pathlib.Path(__file__).with_name("schema_v2.json")
    with open(schema_path, "r") as f:
        return json.load(f)

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate.py <strategy.json>")
        sys.exit(1)
    schema = load_schema()
    with open(sys.argv[1]) as f:
        data = json.load(f)
    try:
        validate(instance=data, schema=schema)
        print("VALID ✅")
    except ValidationError as e:
        print("INVALID ❌")
        print(e)

if __name__ == "__main__":
    main()
