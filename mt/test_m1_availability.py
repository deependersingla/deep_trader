
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mt.mt_api_contract import ACCOUNT, PASSWORD, SERVER

def test_m1_availability():
    """Test different time ranges to find maximum M1 data availability"""
    
    # Connect
    if not mt5.initialize():
        print("❌ initialize() failed")
        return
        
    if not mt5.login(ACCOUNT, password=PASSWORD, server=SERVER):
        print("❌ login() failed")
        mt5.shutdown()
        return
    
    print("✅ Connected to MT5")
    print(f"   Account: {ACCOUNT}, Server: {SERVER}\n")
    
    symbol = 'EURUSD'
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Failed to select {symbol}")
        mt5.shutdown()
        return
    
    timeframe = mt5.TIMEFRAME_M1
    now = datetime.now()
    
    # Test different time ranges
    test_ranges = [
        ("1 jour", 1),
        ("3 jours", 3),
        ("1 semaine", 7),
        ("2 semaines", 14),
        ("1 mois", 30),
        ("2 mois", 60),
        ("3 mois", 90),
        ("6 mois", 180),
        ("1 an", 365),
    ]
    
    print("🔍 Test de disponibilité des données M1 EURUSD:\n")
    print(f"{'Période':<15} {'Barres':<12} {'Statut':<10} {'Plage de dates'}")
    print("-" * 80)
    
    max_available = None
    
    for period_name, days in test_ranges:
        start_date = now - timedelta(days=days)
        
        # Try to get data
        rates = mt5.copy_rates_from(symbol, timeframe, start_date, 100000)
        
        if rates is not None and len(rates) > 0:
            first_date = datetime.fromtimestamp(rates[0][0])
            last_date = datetime.fromtimestamp(rates[-1][0])
            status = "✅ OK"
            max_available = (period_name, days, len(rates), first_date, last_date)
            print(f"{period_name:<15} {len(rates):<12,} {status:<10} {first_date.strftime('%Y-%m-%d')} → {last_date.strftime('%Y-%m-%d')}")
        else:
            status = "❌ Échec"
            error = mt5.last_error()
            print(f"{period_name:<15} {'0':<12} {status:<10} Error: {error}")
    
    print("\n" + "=" * 80)
    
    if max_available:
        period, days, bars, first, last = max_available
        print(f"\n📊 RÉSULTAT:")
        print(f"   Durée maximale disponible: {period} ({days} jours)")
        print(f"   Nombre de barres M1: {bars:,}")
        print(f"   Période couverte: {first.strftime('%Y-%m-%d %H:%M')} → {last.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Taille estimée du fichier: {bars * 75 / (1024*1024):.2f} MB")
        
        # Calculate actual days covered
        actual_days = (last - first).days
        print(f"   Jours réels couverts: {actual_days} jours")
        print(f"   Barres par jour (moyenne): {bars / max(actual_days, 1):.0f}")
    else:
        print("\n❌ Aucune donnée M1 disponible sur ce serveur")
    
    mt5.shutdown()

if __name__ == "__main__":
    test_m1_availability()
