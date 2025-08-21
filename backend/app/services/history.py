import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import tempfile
import subprocess
import json
root_dir = "./static/historical_data"


def handle_historical_download(symbol, timeframe):
    print(f"📥 Fetching historical data: {symbol} - {timeframe}")
    fetch_mt5_data(symbol, timeframe, 'json', './static/historical_data')


TIMEFRAME_LOOKUP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

def fetch_mt5_data(symbol, timeframe_str="M5", output_format="json", folder_name="test_data", max_batches=100):
    timeframe = TIMEFRAME_LOOKUP[timeframe_str]
    full_df = fetch_full_mt5_data(symbol, timeframe, max_batches=30)
   

    # Save
    output_dir = os.path.join(folder_name, symbol)
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{symbol}_{timeframe_str}_{len(full_df)}.{output_format}"
    file_path = os.path.join(output_dir, filename)

    if output_format == "json":
        full_df.to_json(file_path, orient='records', indent=4, date_format='iso')
    elif output_format == "csv":
        full_df.to_csv(file_path, index=False)
    else:
        raise ValueError("Unsupported output format. Use 'json' or 'csv'.")

    print(f"💾 Saved {len(full_df)} records to: {file_path}")
    return full_df


def run_mt5_batch(symbol, timeframe, end_time):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
        output_file = tmp_file.name
    python_path = r"C:\Code\Sandbox\PY\venv\Scripts\python.exe"  # adjust to your environment

    cmd = [
        python_path,
        "C:\\Code\\Sandbox\\PY\\site\\algo\\fetch_batch.py",
        symbol,
        str(timeframe),
        end_time.isoformat(),
        output_file
    ]

    subprocess.run(cmd)

    if not os.path.exists(output_file):
        return None

    with open(output_file, "r") as f:
        result = json.load(f)

    os.remove(output_file)

    if isinstance(result, dict) and "error" in result:
        print("⚠️ Subprocess error:", result)
        return None

    df = pd.DataFrame(result)
    df['time'] = pd.to_datetime(df['time'])
    return df


def fetch_full_mt5_data(symbol, timeframe, max_batches=20):
    all_batches = []
    end_time = datetime.utcnow()
    prev_oldest_time = None

    for i in range(max_batches):
        print(f"\n📦 Batch {i+1} | End time: {end_time}")
        df = run_mt5_batch(symbol, timeframe, end_time)

        if df is None or df.empty:
            print("🛑 No more data or error.")
            break

        oldest_time = df['time'].min()

        if oldest_time == prev_oldest_time:
            print("🛑 Duplicate oldest timestamp detected.")
            break

        prev_oldest_time = oldest_time
        all_batches.append(df)
        end_time = oldest_time - timedelta(seconds=1)
        time.sleep(10)

    if not all_batches:
        print("❌ No data collected.")
        return None

    full_df = pd.concat(all_batches).drop_duplicates(subset='time').sort_values('time')
    print(f"\n✅ Total records: {len(full_df)}")
    return full_df


def get_timestamps(path):
    stat = os.stat(path)
    return {
        "created_at": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    }

def timeFrameFilesAsTree():
    data = []

    for symbol in sorted(os.listdir(root_dir)):
        symbol_path = os.path.join(root_dir, symbol)
        if os.path.isdir(symbol_path):
            timestamps = get_timestamps(symbol_path)
            data.append({
                "id": symbol,
                "parentId": None,
                "name": symbol,
                "type": "folder",
                **timestamps
            })

            for filename in sorted(os.listdir(symbol_path)):
                if filename.endswith(".json"):
                    file_path = os.path.join(symbol_path, filename)
                    timestamps = get_timestamps(file_path)
                    data.append({
                        "id": f"{symbol}/{filename}",
                        "parentId": symbol,
                        "name": filename,
                        "type": "file",
                        **timestamps
                    })

    return data