import os
import polars as pl

INPUT_DIR = "../data"
MRCONSO_FILE = os.path.join(INPUT_DIR, "MRCONSO.RRF")
MRSTY_FILE = os.path.join(INPUT_DIR, "MRSTY.RRF")
OUTPUT_JSON_FILE = os.path.join(INPUT_DIR, "syns_by_cui.jsonl")


# https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/
MRCONSO_COLS = [
    "CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF",
    "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE",
    "STR", "SRL", "SUPPRESS", "CVF"
]
# https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.Tf/
MRSTY_COLS = [
    "CUI", "TUI", "STN", "STY", "ATUI", "CVF"
]

if __name__ == "__main__":
    mrconso_df = pl.read_csv(MRCONSO_FILE, separator="|",
                             has_header=False, new_columns=MRCONSO_COLS,
                             encoding="utf8-lossy")
    mrconso_df = mrconso_df.filter(
        (mrconso_df["LAT"] == "ENG") &     # English syns only
        (mrconso_df["SUPPRESS"] == "N"))   # current syns only
    num_mrconso = len(mrconso_df)

    mrsty_df = pl.read_csv(MRSTY_FILE, separator="|",
                           has_header=False, new_columns=MRSTY_COLS,
                           encoding="utf8-lossy")
    mrsty_df = mrsty_df.filter(
        (mrsty_df["STY"] == "Disease or Syndrome") |    # Diseases
        (mrsty_df["STY"] == "Clinical Drug"))           # Drugs
    print(mrsty_df.head())
    num_mrsty = len(mrsty_df)

    conso_sty_df = (
        mrconso_df
        .join(mrsty_df, on="CUI", how="inner")
        .select(["CUI", "STR", "STY"])
        .group_by(["CUI", "STY"]).agg(pl.col("STR").flatten())
    )
    print(conso_sty_df.head())
    num_final = len(conso_sty_df)

    conso_sty_df.write_ndjson(OUTPUT_JSON_FILE)
    print("#-MRCONSO: {:d}, MRSTY: {:d} (filtered), output: {:d}".format(
        num_mrconso, num_mrsty, num_final))
