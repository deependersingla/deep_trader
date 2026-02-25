
import MetaTrader5 as mt5
import time

# Credentials (should be shared ideally, but keeping standalone as per structure)
ACCOUNT = 40938581
PASSWORD = "Root_-123$"
SERVER = "Deriv-Demo"

def error_handler(msg):
    """Handles the capturing of error messages"""
    print(f"Server Error: {msg}")

def reply_handler(msg):
    """Handles of server replies"""
    # MT5 prints directly usually, but we can wrap
    print(f"Server Response: {msg}")

def create_contract(symbol, sec_type='STK', exch='NSE', prim_exch='NSE', curr='INR'):
    """
    In MT5, a 'contract' is mostly just the symbol name.
    However, to replicate the signature, we accept all params.
    Returns the symbol string which is the key identifier in MT5.
    """
    # MT5 symbols might differ (e.g., "EURUSD" vs "KSCL").
    # For replication, we return the symbol string.
    return symbol

def create_order(order_type, quantity, action):
    """
    Create an Order dictionary structure compatible with MT5.
    
    order_type - 'MKT', 'LMT' (Mapped to MT5 types)
    quantity - Float (MT5 volume)
    action - 'BUY' or 'SELL'
    """
    mt5_type = mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL
    
    # Check for Limit order mapping if needed, simplified for MKT here as per original usage
    if order_type == 'LMT':
        mt5_type = mt5.ORDER_TYPE_BUY_LIMIT if action == 'BUY' else mt5.ORDER_TYPE_SELL_LIMIT
        
    order = {
        "action": mt5.TRADE_ACTION_DEAL,
        "volume": float(quantity),
        "type": mt5_type,
        # "price" will be filled at execution time
    }
    return order

def place_order(symbol, order_dict):
    """
    Helper to execute the order using MT5.
    """
    # Get current price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Symbol {symbol} not found")
        return

    price = tick.ask if order_dict["type"] == mt5.ORDER_TYPE_BUY else tick.bid
    
    request = order_dict.copy()
    request["symbol"] = symbol
    request["price"] = price
    request["deviation"] = 20
    request["magic"] = 123456
    request["comment"] = "mt_api_contract"
    request["type_time"] = mt5.ORDER_TIME_GTC
    request["type_filling"] = mt5.ORDER_FILLING_IOC
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order Failed: {result.retcode} - {result.comment}")
    else:
        print(f"Order Executed: {result.order}")

if __name__ == "__main__":
    # Connect
    if not mt5.initialize():
        print("initialize() failed")
        quit()
        
    if not mt5.login(ACCOUNT, password=PASSWORD, server=SERVER):
        print("login() failed")
        mt5.shutdown()
        quit()
        
    print("Connected to MT5")

    # Create a contract (Symbol)
    # Using EURUSD for demo test instead of KSCL
    test_symbol = "EURUSD" 
    goog_contract = create_contract(test_symbol, 'STK', 'NSE', 'NSE', 'INR')

    # Go long 0.01 lot (min size often)
    goog_order = create_order('MKT', 0.01, 'BUY')

    # Place Order
    print(f"Placing order for {test_symbol}...")
    place_order(goog_contract, goog_order)

    # Disconnect
    mt5.shutdown()
