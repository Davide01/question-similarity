"""Rewrite absolute host paths in mlflow.db to /app (container path)."""

import sqlite3

NEW_ROOT = "/app"

db = sqlite3.connect("mlflow.db")

row = db.execute(
    "SELECT artifact_uri FROM runs WHERE artifact_uri IS NOT NULL LIMIT 1"
).fetchone()

if not row:
    print("No runs found in mlflow.db, nothing to patch.")
    db.close()
    raise SystemExit(0)

uri = row[0]
idx = uri.find("/mlruns/")
if idx < 0:
    print(f"Cannot detect project root from URI: {uri}")
    db.close()
    raise SystemExit(1)

old_root = uri[:idx]

if old_root == NEW_ROOT:
    print("Paths already point to /app, nothing to patch.")
    db.close()
    raise SystemExit(0)

print(f"Patching: '{old_root}' -> '{NEW_ROOT}'")

for table, col in [
    ("runs", "artifact_uri"),
    ("model_versions", "source"),
    ("model_versions", "storage_location"),
    ("experiments", "artifact_location"),
    ("logged_models", "artifact_location"),
    ("workspaces", "default_artifact_root"),
]:
    try:
        cur = db.execute(
            f"UPDATE {table} SET {col} = ? || SUBSTR({col}, LENGTH(?) + 1) "
            f"WHERE {col} LIKE ?",
            (NEW_ROOT, old_root, old_root + "%"),
        )
        print(f"  {table}.{col}: {cur.rowcount} rows updated")
    except Exception as exc:
        print(f"  {table}.{col}: skipped ({exc})")

db.commit()
db.close()
print("Done.")
