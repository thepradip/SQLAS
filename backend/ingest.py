from __future__ import annotations
"""
Data ingestion script — loads CSVs into SQLite with proper schema and indexes.
Run once: python ingest.py

Author: Pradip Tivhale
"""

import sqlite3
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = Path(__file__).resolve().parent / "health.db"


def ingest():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ─── Table 1: health_demographics ───────────────────────────────────────
    cursor.execute("DROP TABLE IF EXISTS health_demographics")
    cursor.execute("""
        CREATE TABLE health_demographics (
            Patient_Number              INTEGER PRIMARY KEY,
            Blood_Pressure_Abnormality  INTEGER NOT NULL,
            Level_of_Hemoglobin         REAL NOT NULL,
            Genetic_Pedigree_Coefficient REAL,
            Age                         INTEGER NOT NULL,
            BMI                         REAL NOT NULL,
            Sex                         INTEGER NOT NULL,
            Pregnancy                   INTEGER NOT NULL DEFAULT 0,
            Smoking                     INTEGER NOT NULL,
            salt_content_in_the_diet    REAL NOT NULL,
            alcohol_consumption_per_day REAL DEFAULT 0,
            Level_of_Stress             INTEGER NOT NULL,
            Chronic_kidney_disease      INTEGER NOT NULL,
            Adrenal_and_thyroid_disorders INTEGER NOT NULL
        )
    """)

    with open(DATA_DIR / "health_dataset_1.csv") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append((
                int(r["Patient_Number"]),
                int(r["Blood_Pressure_Abnormality"]),
                float(r["Level_of_Hemoglobin"]),
                float(r["Genetic_Pedigree_Coefficient"]) if r["Genetic_Pedigree_Coefficient"] else None,
                int(r["Age"]),
                float(r["BMI"]),
                int(r["Sex"]),
                int(float(r["Pregnancy"])) if r["Pregnancy"] else 0,
                int(r["Smoking"]),
                float(r["salt_content_in_the_diet"]),
                float(r["alcohol_consumption_per_day"]) if r["alcohol_consumption_per_day"] else 0,
                int(r["Level_of_Stress"]),
                int(r["Chronic_kidney_disease"]),
                int(r["Adrenal_and_thyroid_disorders"]),
            ))

    cursor.executemany(
        "INSERT INTO health_demographics VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    print(f"  health_demographics: {len(rows)} rows inserted")

    cursor.execute("CREATE INDEX idx_demo_bp ON health_demographics(Blood_Pressure_Abnormality)")
    cursor.execute("CREATE INDEX idx_demo_sex ON health_demographics(Sex)")
    cursor.execute("CREATE INDEX idx_demo_smoking ON health_demographics(Smoking)")
    cursor.execute("CREATE INDEX idx_demo_ckd ON health_demographics(Chronic_kidney_disease)")
    cursor.execute("CREATE INDEX idx_demo_stress ON health_demographics(Level_of_Stress)")
    cursor.execute("CREATE INDEX idx_demo_age ON health_demographics(Age)")

    # ─── Table 2: physical_activity ─────────────────────────────────────────
    cursor.execute("DROP TABLE IF EXISTS physical_activity")
    cursor.execute("""
        CREATE TABLE physical_activity (
            Patient_Number    INTEGER NOT NULL,
            Day_Number        INTEGER NOT NULL,
            Physical_activity INTEGER,
            PRIMARY KEY (Patient_Number, Day_Number),
            FOREIGN KEY (Patient_Number) REFERENCES health_demographics(Patient_Number)
        )
    """)

    with open(DATA_DIR / "health_dataset_2.csv") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            activity = r["Physical_activity"]
            rows.append((
                int(r["Patient_Number"]),
                int(r["Day_Number"]),
                int(float(activity)) if activity else None,
            ))

    cursor.executemany(
        "INSERT INTO physical_activity VALUES (?,?,?)",
        rows,
    )
    print(f"  physical_activity: {len(rows)} rows inserted")

    cursor.execute("CREATE INDEX idx_activity_patient ON physical_activity(Patient_Number)")
    cursor.execute("CREATE INDEX idx_activity_day ON physical_activity(Day_Number)")

    conn.commit()

    for table in ["health_demographics", "physical_activity"]:
        count = cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows verified")

    conn.close()
    print(f"\nDatabase created at: {DB_PATH}")


if __name__ == "__main__":
    print("Ingesting health data into SQLite...")
    ingest()
    print("Done.")
