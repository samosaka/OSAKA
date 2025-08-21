from flask import Blueprint

from app.services.files import get_backtests

backtest_bp = Blueprint('backtest', __name__)

@backtest_bp.route('/backtest', methods=['GET'])
def get_backtest1():
    return get_backtests()
