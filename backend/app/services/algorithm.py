import os
import json
from datetime import datetime
root_dir = "./static/strategy"

def get_timestamps(path):
    stat = os.stat(path)
    return {
        "created_at": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    }

def build_filelist_data():
    data = []

    for filename in sorted(os.listdir(root_dir)):
        if filename.endswith(".py"):
            file_path = os.path.join(root_dir, filename)
            timestamps = get_timestamps(file_path)

            data.append({
                "id": filename,
                "parentId": None,
                "name": filename,
                "type": "file",
                **timestamps
            })

    return data