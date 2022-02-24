
import time
import queue
import argparse
from data import HistoricCSVDataHandler
from strategy import BuyAndHoldStrategy
from portfolio import NaivePortfolio
from execution import SimulatedExecutionHandler


def main(csv_dir, symbols, start_date):
    events = queue.Queue()
    bars = HistoricCSVDataHandler(events, csv_dir, symbols.split(","))
    strategy = BuyAndHoldStrategy(bars, events)
    port = NaivePortfolio(bars, events, start_date)
    broker = SimulatedExecutionHandler(events)

    while True:
        # Update the bars (specific backtest code, as opposed to live trading)
        if bars.continue_backtest == True:
            bars.update_bars()
        else:
            break
        
        # Handle the events
        while True:
            try:
                event = events.get(False)
            except queue.Empty:
                break
            else:
                if event is not None:
                    if event.type == 'MARKET':
                        strategy.calculate_signals(event)
                        port.update_timeindex(event)

                    elif event.type == 'SIGNAL':
                        port.update_signal(event)

                    elif event.type == 'ORDER':
                        broker.execute_order(event)

                    elif event.type == 'FILL':
                        port.update_fill(event)

        # 10-Minute heartbeat
        #time.sleep(10*60)
    
    port.create_equity_curve_dataframe()
    stats = port.output_summary_stats()
    print(stats)
    return port

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="quick start backtest")
    parser.add_argument("csv_dir", type=str, help="csv file path")
    parser.add_argument("symbols", type=str, help="symbols to backtest")
    parser.add_argument("start_date", type=str, help="start time to backtest")
    args = parser.parse_args()
    main(args.csv_dir, args.symbols, args.start_date)