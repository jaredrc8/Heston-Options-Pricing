"""
Print the structure of files under data/: file names, column names, dtypes,
row counts and null counts. It never prints a data value (DATA_CONTRACT.md
section 3.2), and any error is reported by file name and error type only.

    python scripts/inspect_schema.py data/some_file.csv
    python scripts/inspect_schema.py data/some_folder
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

SUFFIXES = (".csv", ".csv.gz", ".parquet")


def inspect_csv(path):
    """Read in chunks so that large files fit in memory. Returns (dtypes, rows, nulls)."""
    dtypes = {}
    nulls = {}
    rows = 0
    for chunk in pd.read_csv(path, chunksize=200_000, low_memory=False):
        rows += len(chunk)
        for column in chunk.columns:
            dtypes[column] = str(chunk[column].dtype)
            nulls[column] = nulls.get(column, 0) + int(chunk[column].isna().sum())
    return dtypes, rows, nulls


def inspect_parquet(path):
    """Use the Parquet footer only, so no column data is loaded. Returns (dtypes, rows, nulls)."""
    file = pq.ParquetFile(path)
    schema = file.schema_arrow
    dtypes = {name: str(schema.field(name).type) for name in schema.names}
    nulls = {name: 0 for name in schema.names}
    for g in range(file.metadata.num_row_groups):
        group = file.metadata.row_group(g)
        for c in range(group.num_columns):
            column = group.column(c)
            stats = column.statistics
            name = column.path_in_schema
            if name in nulls:
                if stats is not None and stats.has_null_count:
                    nulls[name] += stats.null_count
                else:
                    nulls[name] = None if nulls[name] is None else -1
    return dtypes, file.metadata.num_rows, nulls


def inspect_file(path):
    print(f"\nFILE {path}")
    try:
        if path.name.endswith(".parquet"):
            dtypes, rows, nulls = inspect_parquet(path)
        else:
            dtypes, rows, nulls = inspect_csv(path)
    except Exception as error:
        # Pandas messages can echo row contents, so only the type is shown.
        print(f"  could not read this file ({type(error).__name__})")
        return
    print(f"  rows: {rows}")
    for column in dtypes:
        print(f"  {column!r:32} {dtypes[column]:14} nulls={nulls[column]}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    target = Path(sys.argv[1])
    if target.is_dir():
        files = sorted(p for p in target.rglob("*") if p.is_file() and p.name.endswith(SUFFIXES))
    else:
        files = [target]

    # Large partitioned folders: the first and last file are enough to see the structure.
    print(f"{len(files)} file(s) found")
    if len(files) > 2:
        print("showing the first and last file only")
        files = [files[0], files[-1]]
    for path in files:
        inspect_file(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
