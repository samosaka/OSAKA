from flask import Blueprint, jsonify
from app.services.mt5 import get_filtered_symbols

meta_bp = Blueprint('meta', __name__)

@meta_bp.route('/symbols', methods=['GET'])
def get_symbols():
    return jsonify(get_filtered_symbols())
