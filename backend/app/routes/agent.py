from flask import Blueprint, jsonify, request

from app.services.agent.train import train_model
from app.services.files import get_agents
from app.services.agent.test import test_model
from app.services.agent.plugin_loader import load_backtest_module
from typing import Any, Callable, Optional

agent_bp = Blueprint('agent', __name__)

@agent_bp.route("/agent/train", methods=["POST"])
def train_agent():
    data = request.get_json()
    data_filename = data.get("dataFile")
    strategy_filename = data.get("strategyFile")
    
    if not data_filename or not strategy_filename:
        return jsonify({"error": "Missing files"}), 400

    symbol = get_symbol_from_filename(data_filename)

    # TODO: Start actual training logic here
    print(f"🎓 Generating with symbol: { symbol }, data: {data_filename}, strategy: {strategy_filename}")

    train_out = train_model(
        strategy_name=strategy_filename,
        data_filename=data_filename,
        config={"symbol":symbol,"window_size":50,"risk_percentage":0.02,"reward_ratio":2.0,"seed":42},
        strategies_dir="static\\strategy",
        data_dir="static\\historical_data\\"+symbol,
        out_root="static\\agents",
        train_steps=300_000
    )


    return jsonify({"message": "Generation started"})

@agent_bp.route("/agent/test", methods=["POST"])
def test_agent():
    data = request.get_json()
    data_filename = data.get("dataFile")
    agent_name = data.get("agentFile")
    backtest_filname = data.get("backtestFile")
    
    if not data_filename or not agent_name:
        return jsonify({"error": "Missing files"}), 400

    symbol = get_symbol_from_filename(data_filename)

    # TODO: Start actual training logic here
    print(f"🎓 Testing with symbol: { symbol }, data: {data_filename}, agent: {agent_name}")

    function_name = "run_detailed_backtest"

    mod = load_backtest_module('static\\backtests\\' + backtest_filname, fn_name=function_name)
    backtest_fn: Callable[..., Any] = getattr(mod, function_name)


    test_output = test_model(
        run_dir= agent_name,
        data_filename=data_filename,
        data_dir="static\\historical_data\\" + symbol,
        backtest_fn=backtest_fn
    )


    return jsonify({"message": "Generation started"})

@agent_bp.route('/agent', methods=['GET'])
def get_agents1():
    return get_agents()


def get_symbol_from_filename(filename: str) -> str:
    """
    Extracts the trading symbol from a filename like 'AUDUSD_MN1_79'.
    The symbol is the part before the first underscore.
    """
    base = filename.split("/")[-1]      # remove folder path if any
    name = base.split(".")[0]           # remove extension if any
    return name.split("_")[0]