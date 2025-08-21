from flask import Blueprint
from app.services.algorithm import build_filelist_data

algorithm_bp = Blueprint('algorithm', __name__)

@algorithm_bp.route('/algorithms', methods=['GET'])
def get_symbols():
    return build_filelist_data()