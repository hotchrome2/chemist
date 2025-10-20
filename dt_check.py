import pandas as pd
import datetime

def find_missing_datetimes(strs, target_date):
    """
    指定した1日を基準に正解DatetimeIndexを生成し、
    文字列リストから作成したdatetimeとの不足・不正を検出する。

    Parameters
    ----------
    strs : list of str
        調査対象の日時文字列リスト。
        例: ["2025-10-20 00:00:00", "2025-10-20 00:00:10", ...]
    target_date : datetime.date
        対象の日付（例: datetime.date(2025, 10, 20)）

    Returns
    -------
    result : dict
        欠落や不正データの情報をまとめた辞書。
        {
            "match": bool,                 # 欠落や不正がなければ True
            "missing_count": int,          # 欠落数
            "missing_index": DatetimeIndex,# 欠落した日時
            "invalid_count": int,          # 不正文字列の数
            "invalid_positions": list[int] # 不正文字列の位置
        }
    """

    # --- 1. 正解のDatetimeIndexを生成（10秒間隔、1日分） ---
    start = pd.Timestamp(datetime.datetime.combine(target_date, datetime.time(0, 0, 0)))
    end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    truth_index = pd.date_range(start=start, end=end, freq="10S")

    # --- 2. 調査対象の日時を変換 ---
    dt_series = pd.to_datetime(strs, errors="coerce")

    # --- 3. 無効文字列（NaT）の記録 ---
    invalid_positions = dt_series[dt_series.isna()].index.tolist()
    invalid_count = len(invalid_positions)

    # --- 4. 有効なdatetimeだけ残す ---
    dt_series = dt_series.dropna()

    # --- 5. Trueフラグを値にしたSeriesを作成 ---
    s_found = pd.Series(True, index=dt_series).sort_index()

    # --- 6. 正解のインデックスに合わせて再インデックス ---
    aligned = s_found.reindex(truth_index)

    # --- 7. 欠落の検出 ---
    is_missing = aligned.isna()
    missing_index = aligned[is_missing].index
    missing_count = int(is_missing.sum())

    # --- 8. 結果まとめ ---
    result = {
        "match": missing_count == 0 and invalid_count == 0,
        "missing_count": missing_count,
        "missing_index": missing_index,
        "invalid_count": invalid_count,
        "invalid_positions": invalid_positions
    }

    return result
