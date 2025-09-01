
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
        # Additional semantic checks for grouped conditions
        errors = []
        conditions = data.get("conditions", [])
        groups = {}
        for c in conditions:
            gid = c.get("group_id")
            if gid is not None:
                groups.setdefault(gid, []).append(c)

        for gid, conds in groups.items():
            types = [c.get("type") for c in conds]
            actionable = [t for t in types if t in ("entry", "exit")]
            # Exactly one actionable type per group (entry or exit), rest can be filters
            if len(actionable) == 0:
                errors.append(f"Group {gid}: must contain exactly one actionable condition (entry or exit), found none.")
            elif len(set(actionable)) > 1:
                errors.append(f"Group {gid}: cannot mix entry and exit in the same group.")
            elif types.count("entry") > 1 or types.count("exit") > 1:
                errors.append(f"Group {gid}: should contain exactly one entry or exit condition (found multiple).")

            # Consistent timeframe within group (if provided)
            tfs = set([c.get("timeframe") for c in conds if c.get("timeframe")])
            if len(tfs) > 1:
                errors.append(f"Group {gid}: mixed timeframes {sorted(tfs)}, must be consistent.")

        if errors:
            print("INVALID ❌")
            for e in errors:
                print("-", e)
            sys.exit(1)

        print("VALID ✅")
    except ValidationError as e:
        print("INVALID ❌")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
