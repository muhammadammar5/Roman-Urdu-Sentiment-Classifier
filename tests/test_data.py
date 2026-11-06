import pandas as pd

from src.data import clean, load_raw, load_split, prepare, split

ROWS = (
    ["yeh product bohat acha hai {i},Positive,".format(i=i) for i in range(40)]
    + ["service kharab thi bilkul {i},Neative,".format(i=i) for i in range(40)]   # typo label
    + ["delivery kal aaye gi {i},Neutral,".format(i=i) for i in range(40)]
    + ["Yeh product bohat acha hai 0,Positive,"]             # duplicate (case differs)
    + ["same text,Positive,", "same text,Negative,"]         # conflicting labels
    + [",Neutral,"]                                          # missing text
    + ["text with bad label,Maybe,"]
)


def write_csv(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("\n".join(ROWS) + "\n", encoding="utf-8")
    return path


def test_load_raw_has_no_header_and_two_columns(tmp_path):
    df = load_raw(write_csv(tmp_path))
    assert list(df.columns) == ["text", "label"]
    assert len(df) == len(ROWS)


def test_clean_fixes_typos_and_removes_noise(tmp_path):
    df, stats = clean(load_raw(write_csv(tmp_path)), "three")
    assert set(df["label"]) == {"Positive", "Negative", "Neutral"}   # "Neative" fixed
    assert len(df) == 120                                           # dupes, conflicts, bad rows removed
    assert stats["dropped_conflicting_label_texts"] == 1
    assert "same text" not in set(df["text"])


def test_binary_mode_drops_neutral(tmp_path):
    df, _ = clean(load_raw(write_csv(tmp_path)), "binary")
    assert set(df["label"]) == {"Positive", "Negative"}
    assert set(df["label_id"]) == {0, 1}


def test_splits_are_disjoint_and_stratified(tmp_path):
    df, _ = clean(load_raw(write_csv(tmp_path)), "three")
    train, val, test = split(df, seed=1)
    assert len(train) + len(val) + len(test) == len(df)
    keys = [set(part["text"]) for part in (train, val, test)]
    assert not (keys[0] & keys[1]) and not (keys[0] & keys[2]) and not (keys[1] & keys[2])
    assert set(test["label"]) == {"Positive", "Negative", "Neutral"}


def test_prepare_writes_files_and_stats(tmp_path):
    stats = prepare(write_csv(tmp_path), tmp_path / "processed", "three", seed=3)
    assert stats["final_rows"] == 120
    train = load_split("three", "train", tmp_path / "processed")
    assert {"text", "label", "label_id"} <= set(train.columns)
