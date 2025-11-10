#!/usr/bin/env python
import argparse
import pandas as pd
from neuralprophet import load
from pathlib import Path

# NOTE: Pin your NeuralProphet version same as training to avoid incompatibilities.
# If you ran into the PyTorch 2.6 "weights_only" change, ensure your env matches training,
# or re-save the model with your current versions to keep things stable.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--horizon-hours", type=int, default=48)
    ap.add_argument("--freq", default="H")
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--history-csv", default="data/current.csv",
                    help="Your latest history with columns ['ds','y'] or your target schema.")
    args = ap.parse_args()

    # Load saved model (no fitting)
    m = load(args.model_path)

    # Load your latest history (whatever you used at train time)
    # Must have 'ds' and the trained target column name.
    df = pd.read_csv(args.history_csv)
    # If needed: df['ds'] = pd.to_datetime(df['ds'], utc=True).dt.tz_convert("America/Montreal").dt.tz_localize(None)

    # Make future DF and predict
    future = m.make_future_dataframe(
        df, periods=args.horizon_hours, n_historic_predictions=False, freq=args.freq)
    fcst = m.predict(future)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    if args.output_path.endswith(".csv"):
        fcst.to_csv(args.output_path, index=False)
    else:
        try:
            fcst.to_parquet(args.output_path, index=False)
        except Exception:
            fcst.to_csv(args.output_path.replace(
                ".parquet", ".csv"), index=False)


if __name__ == "__main__":
    main()
