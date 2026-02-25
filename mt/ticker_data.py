
import MetaTrader5 as mt5
import time
from mt.mt_api_contract import ACCOUNT, PASSWORD, SERVER

def error_handler(msg):
    print(msg)

class MT_TickerData:
    def __init__(self, symbol="EURUSD"):
        self.symbol = symbol
        self.connect()

    def connect(self):
        if not mt5.initialize():
            print("initialize() failed")
            return
            
        if not mt5.login(ACCOUNT, password=PASSWORD, server=SERVER):
            print("login() failed")
            return
            
        print("Connected to MT5")
        
        # Ensure symbol is selected
        if not mt5.symbol_select(self.symbol, True):
            print(f"Failed to select symbol {self.symbol}")

    def run(self):
        print(f"Requesting Tick Data for {self.symbol}...")
        try:
            while True:
                tick = mt5.symbol_info_tick(self.symbol)
                if tick:
                    # Imitate the print format of original ticker_data.py
                    # Original prints: FIELD value
                    print(f"LAST {tick.last}")
                    print(f"BID {tick.bid}")
                    print(f"ASK {tick.ask}")
                    print(f"VOLUME {tick.volume}")
                    print(f"TIME {tick.time}")
                else:
                    print("No tick data")
                
                # Original had sleep(10000000) inside main but handled callbacks. 
                # Here we poll. To see "updates", we need a loop.
                time.sleep(1) 
                
        except KeyboardInterrupt:
            print("Stopping...")
            mt5.shutdown()

if __name__ == "__main__":
    # Original used KSCL, we use EURUSD for demo availability
    app = MT_TickerData(symbol="EURUSD")
    app.run()
