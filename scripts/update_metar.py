import os
import requests
import pandas as pd
from utils import upload
import dropbox
from dotenv import load_dotenv


load_dotenv()

metar_df = pd.read_csv(
    'https://www.dropbox.com/scl/fi/7b390c7zu7lg2nug9r21e/full_metar_data.csv?rlkey=ob25xfgvuqth42lruczhszoz3&raw=1')
metar_df.valid = pd.to_datetime(metar_df.valid, errors='coerce')
print("METAR data shape:", metar_df.shape)


# get the most recent date in metar_df
most_recent_date = metar_df['valid'].max()
print("Most recent date in METAR data:", most_recent_date)

# get the day before the most recent date
day_before_most_recent = most_recent_date - pd.Timedelta(days=1)
print("Day before most recent date:", day_before_most_recent)

# get the day after the most recent date
day_after_most_recent = most_recent_date + pd.Timedelta(days=1)
print("Day after most recent date:", day_after_most_recent)

base_url = (
    "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
    "network=CA_QC_ASOS"
    "&station=CYUL"
    "&data=all"
    "&year1={y1}&month1={m1}&day1={d1}"
    "&year2={y2}&month2={m2}&day2={d2}"
    "&tz=America%2FNew_York"
    "&format=onlycomma"
    "&latlon=no"
    "&elev=no"
    "&missing=M"
    "&trace=T"
    "&direct=no"
    "&report_type=3"
)

# Example usage:
url = base_url.format(
    y1=day_before_most_recent.year, m1=day_before_most_recent.month, d1=day_before_most_recent.day,
    y2=day_after_most_recent.year, m2=day_after_most_recent.month, d2=day_after_most_recent.day
)

print(url)


recent_metar_df = pd.read_csv(url)
print("Recent METAR data shape:", recent_metar_df.shape)

# merge recent_metar_df with metar_df, avoiding duplicates
metar_df = pd.concat([metar_df, recent_metar_df]).drop_duplicates(
    subset=['valid']).reset_index(drop=True)
print("Updated METAR dataframe shape:", metar_df.shape)

metar_df.to_csv('full_metar_data.csv', index=False)

dropbox_app_key = os.environ.get("DROPBOX_APP_KEY")
dropbox_app_secret = os.environ.get("DROPBOX_APP_SECRET")
dropbox_refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")

# exchange the authorization code for an access token:
token_url = "https://api.dropboxapi.com/oauth2/token"
params = {
    "grant_type": "refresh_token",
    "refresh_token": dropbox_refresh_token,
    "client_id": dropbox_app_key,
    "client_secret": dropbox_app_secret
}
r = requests.post(token_url, data=params)
# print(r.text)

dropbox_access_token = r.json()['access_token']

dbx = dropbox.Dropbox(dropbox_access_token)

upload(dbx, 'full_metar_data.csv', 'metar', '',
            'full_metar_data.csv', overwrite=True)
