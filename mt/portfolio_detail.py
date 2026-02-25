
import MetaTrader5 as mt5
import time
from mt.mt_api_contract import ACCOUNT, PASSWORD, SERVER

class MT_Portfolio:
    def __init__(self):
        self.connect()
        
    def connect(self):
        if not mt5.initialize():
            print("initialize() failed")
            return
            
        if not mt5.login(ACCOUNT, password=PASSWORD, server=SERVER):
            print("login() failed")
            return
            
        print("Connected to MT5")

    def get_account_updates(self):
        """
        Equivalent to reqAccountUpdates. 
        Fetches account info and positions.
        """
        while True:
            # 1. Account Info
            account_info = mt5.account_info()
            if account_info:
                print("\n=== Account Info ===")
                print(f"Balance: {account_info.balance}")
                print(f"Equity: {account_info.equity}")
                print(f"Margin: {account_info.margin}")
                print(f"Profit: {account_info.profit}")
            else:
                print("Failed to get account info")

            # 2. Positions (Portfolio)
            positions = mt5.positions_get()
            if positions:
                print("\n=== Positions ===")
                for pos in positions:
                    print(pos)
            else:
                print("\nNo positions currently open.")
                
            time.sleep(10)

if __name__ == '__main__':
    bot = MT_Portfolio()
    try:
        bot.get_account_updates()
    except KeyboardInterrupt:
        print("Stopping...")
        mt5.shutdown()
