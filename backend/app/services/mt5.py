import MetaTrader5 as mt5

def get_filtered_symbols():
    if not mt5.initialize():
        return {"error": f"MT5 init failed: {mt5.last_error()}"}
    
    symbols = mt5.symbols_get()
    mt5.shutdown()

    return [{"id": s.name, "name": s.name} for s in symbols if "USD" in s.name or "EUR" in s.name]
