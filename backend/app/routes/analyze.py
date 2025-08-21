from flask import Blueprint, request, jsonify
from app.services.algorithm import build_filelist_data
from app.services.files import get_training_results
import json
from pathlib import Path
from typing import Any

analyze_bp = Blueprint('analyze', __name__)

@analyze_bp.route('/analyze/testResults', methods=['GET'])
def testResults():
    return get_training_results()



@analyze_bp.route('/analyze/testResultData', methods=['POST'])
def testResultData():
    data = request.get_json()
    testResult = data.get("testResult")
    return load_backtest_results(testResult)

    # this.testResults = function () {
    #     return $http.get(`${baseURL}/analyze/testResults`);
    # };

    # this.testResultData = function (testResult) {
    #     return $http.post(`${baseURL}/analyze/testResultData`, {
    #         testResult: testResult
    #     });
    # };


def load_backtest_results(folder_name: str) -> dict[str, Any]:
    """
    Load backtest results (metrics, trades, used data) from results/<folder_name>.
    
    Returns a dictionary with keys:
        - metrics
        - trades
        - used_data
    """
    out_dir = Path("static/backtest_results") / folder_name
    if not out_dir.exists():
        raise FileNotFoundError(f"Folder not found: {out_dir.as_posix()}")

    result = {}

    # Metrics
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            result["metrics"] = json.load(f)
    else:
        result["metrics"] = None

    # Trades
    trades_path = out_dir / "trades.json"
    if trades_path.exists():
        with trades_path.open("r", encoding="utf-8") as f:
            result["trades"] = json.load(f)
    else:
        result["trades"] = []

    # Used data
    data_path = out_dir / "used_data.json"
    if data_path.exists():
        with data_path.open("r", encoding="utf-8") as f:
            result["used_data"] = json.load(f)
    else:
        result["used_data"] = []

    return result