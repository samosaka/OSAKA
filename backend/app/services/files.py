import os
import json
from datetime import datetime
agents_root_dir = "./static/agents"
backtests_root_dir = "./static/backtests"

def get_timestamps(path):
    stat = os.stat(path)
    return {
        "created_at": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    }

def get_agents():
    data = []

    for symbol in sorted(os.listdir(agents_root_dir)):
        symbol_path = os.path.join(agents_root_dir, symbol)
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



def get_backtests():
    data = []

    for filename in sorted(os.listdir(backtests_root_dir)):
        if filename.endswith(".py"):
            file_path = os.path.join(backtests_root_dir, filename)
            timestamps = get_timestamps(file_path)

            data.append({
                "id": filename,
                "parentId": None,
                "name": filename,
                "type": "file",
                **timestamps
            })

    return data



