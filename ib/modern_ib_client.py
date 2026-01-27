import datetime
import time
import threading
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class IBDataApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.finished = False

    def error(self, reqId, errorCode, errorString):
        print(f"Error: {reqId} {errorCode} {errorString}")

    def historicalData(self, reqId, bar):
        # Callback for historical data
        self.data.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume, bar.wap, bar.barCount])

    def historicalDataEnd(self, reqId, start, end):
        print(f"Historical Data Download Finished for ReqId: {reqId}")
        self.finished = True

def download_historical_data(symbol, duration="2 D", bar_size="1 min", what_to_show="TRADES"):
    app = IBDataApp()
    # Connect to TWS or IB Gateway
    # standard port is 7497 (paper) or 7496 (live)
    app.connect("127.0.0.1", 7497, clientId=1)

    # Start the socket in a thread
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()

    time.sleep(1) # Allow connection time

    if not app.isConnected():
        print("Failed to connect to TWS/Gateway. Make sure it is open and accepting connections.")
        return None

    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK" # Or IND, depends on symbol
    contract.exchange = "NSE" # From original code
    contract.currency = "INR"

    print(f"Requesting data for {symbol}...")
    app.reqHistoricalData(reqId=1, 
                          contract=contract, 
                          endDateTime="", 
                          durationStr=duration, 
                          barSizeSetting=bar_size, 
                          whatToShow=what_to_show, 
                          useRTH=1, 
                          formatDate=1, 
                          keepUpToDate=False, 
                          chartOptions=[])

    # Wait for completion
    timeout = 30
    start_time = time.time()
    while not app.finished:
        time.sleep(1)
        if time.time() - start_time > timeout:
            print("Timeout waiting for data.")
            break

    app.disconnect()
    
    if app.data:
        df = pd.DataFrame(app.data, columns=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume', 'WAP', 'Count'])
        return df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    symbols = ['NIFTY50'] # Original used NIFTY50 as IND, others as STK. 
    # Logic to distinguish can be added.
    
    for sym in symbols:
        df = download_historical_data(sym)
        if not df.empty:
            filename = f"{sym}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved to {filename}")
        else:
            print(f"No data for {sym}")
