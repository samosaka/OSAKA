from flask import Blueprint, request, jsonify
from app.services.history import handle_historical_download, timeFrameFilesAsTree

history_bp = Blueprint('history', __name__)

@history_bp.route('/getHistorical', methods=['POST'])
def get_historical():
    data = request.get_json()
    symbol = data.get("selectedSymbol")
    timeframe = data.get("selectedTimeFrame")
    handle_historical_download(symbol, timeframe)
    return jsonify({"status": "download_started"})

@history_bp.route('/getHistoricalTree', methods=['GET'])
def get_tree():
    return jsonify(timeFrameFilesAsTree())
