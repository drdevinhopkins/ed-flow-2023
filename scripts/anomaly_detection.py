import dropbox
from dotenv import load_dotenv
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
np.float_ = np.float64
from prophet import Prophet
import argparse
from anomaly_refresh import validate_source, validate_ranges, publish_file

def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Refresh and verify anomaly reference ranges")
    parser.add_argument("--hourly-only", action="store_true", help="Skip daily inflow forecasting")
    args = parser.parse_args()

    run(args)


def run(args):

    allData = pd.read_csv(
        'https://www.dropbox.com/scl/fi/s83jig4zews1xz7vhezui/allDataWithCalculatedColumns.csv?rlkey=9mm4zwaugxyj2r4ooyd39y4nl&dl=1')

    allData.ds = pd.to_datetime(allData.ds)
    print(allData.tail(1))
    print('length of allData: '+str(len(allData)))

    validate_source(allData)
    df = allData.copy()
    df.isna().sum().sum()
    df.ds = pd.to_datetime(df.ds)

    output = pd.DataFrame()
    FIRST_RUN = True

    for column in tqdm(df.columns.to_list()):
        if column in ['ds']:
            continue
        try:
            print('working on '+column)

            m = Prophet(interval_width=0.95)
            m.fit(df[['ds', column]].rename(columns={column: 'y'}))
            # Keep a full extra day of headroom beyond the 24h forecast horizon.
            # The anomaly ranges refresh daily, so 48h could become one hour short
            # around the refresh boundary before the new file finished publishing.
            future = m.make_future_dataframe(periods=24*3, freq='h')
            forecast = m.predict(future.tail(24*14))
            if FIRST_RUN:
                output['ds'] = forecast['ds']

            for forecast_column in ['yhat', 'yhat_lower', 'yhat_upper']:
                kwargs = {column+'_'+forecast_column: forecast[forecast_column]}
                output = output.assign(**kwargs)
            FIRST_RUN = False
        except Exception as exc:
            print(f'{column} failed: {type(exc).__name__}')

    validate_ranges(output)
    output.to_csv('anomaly_detection_ranges.csv', index=False)
    output.to_excel('anomaly_detection_ranges.xlsx', index_label="index")

    dbx = dropbox.Dropbox(
        oauth2_refresh_token=os.environ["DROPBOX_REFRESH_TOKEN"],
        app_key=os.environ["DROPBOX_APP_KEY"],
        app_secret=os.environ["DROPBOX_APP_SECRET"],
        timeout=120,
    )
    publish_file(dbx, 'anomaly_detection_ranges.csv', verify_ranges=True)
    publish_file(dbx, 'anomaly_detection_ranges.xlsx')
    if args.hourly_only:
        return

    print(allData.tail(5))


    daily_inflow = pd.read_csv(
        'https://www.dropbox.com/scl/fi/e61kmtweg90tq5hpqrnsf/daily_inflow.csv?rlkey=15r15klj0s0005fvhvlt2iuqt&raw=1')
    daily_inflow.ds = pd.to_datetime(daily_inflow.ds)
    daily_inflow = daily_inflow.tail(365*3)

    column = 'Daily_Inflow_Total'
    m = Prophet(interval_width=0.95, weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                seasonality_mode='additive', changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, holidays_prior_scale=10.0)
    m.add_country_holidays(country_name='CA')
    m.fit(daily_inflow[['ds', column]].rename(columns={column: 'y'}))
    future = m.make_future_dataframe(periods=30, freq='D')
    forecast = m.predict(future.tail(30+14))

    daily_inflow_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    daily_inflow_forecast.to_csv('daily_inflow_forecast.csv', index=False)
    daily_inflow_forecast.to_excel(
        'daily_inflow_forecast.xlsx', index_label="index")
    publish_file(dbx, 'daily_inflow_forecast.csv')
    publish_file(dbx, 'daily_inflow_forecast.xlsx')
    print(daily_inflow_forecast.tail(5))


if __name__ == "__main__":
    main()
