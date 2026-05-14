from core import (simulate_intraday, signal_and_backtest, perf_stats)
import matplotlib.pyplot as plt

def run_synthetic():
    df = simulate_intraday(n_days=12)
    df = signal_and_backtest(df, z_window=60, z_thresh=1.0, round_trip_cost=0.0005)

    stats = perf_stats(df['strategy_ret'], freq='min')
    cum = (1 + df['strategy_ret']).cumprod() - 1

    print("Sample rows:")
    print(df[['mid','ret','imbalance','imb_z','signal','pos','strategy_ret']].head(12).to_string())

    print("\nPerformance (synthetic):")
    for k,v in stats.items():
        print(f"  {k}: {v:.6f}")

    # Plot cumulative returns
    plt.figure(figsize=(10,4))
    plt.plot(cum.index, cum.values)
    plt.title("Project1 (Synthetic): Intraday strategy cumulative return")
    plt.xlabel("Time")
    plt.ylabel("Cumulative return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_synthetic()