import os
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from utils import upload
import dropbox

load_dotenv()

# make a call to this api https://www.shiftadmin.com/vjgh/org_scheduled_shifts with a body json of start_date and end_date in the format YYYY-MM-DD
# we need to use basic auth with the username and password stored in environment variables SHIFTADMIN_USER and SHIFTADMIN_PASS

SHIFTADMIN_USER = os.getenv("SHIFTADMIN_USER")
SHIFTADMIN_PASS = os.getenv("SHIFTADMIN_PASS")


def fetch_shifts(start_date, end_date):
    url = "https://www.shiftadmin.com/vjgh/org_scheduled_shifts"
    body = {
        "start_date": start_date,
        "end_date": end_date
    }
    response = requests.post(url, json=body, auth=(
        SHIFTADMIN_USER, SHIFTADMIN_PASS))
    response.raise_for_status()
    return response.json()


all_shifts_from_dropbox = pd.read_csv(
    'https://www.dropbox.com/scl/fi/yeyr2a7pj6nry8i2q3m0c/all_shifts.csv?rlkey=q1su2h8fqxfnlu7t1l2qe1w0q&raw=1')
all_shifts_from_dropbox['shift_start'] = pd.to_datetime(
    all_shifts_from_dropbox['shift_start'], errors='coerce')
all_shifts_from_dropbox['shift_end'] = pd.to_datetime(
    all_shifts_from_dropbox['shift_end'], errors='coerce')

# fetch todays shifts
today = datetime.now().date()
lastweek = today - timedelta(days=7)
nextweek = today + timedelta(days=7)
shifts = fetch_shifts(str(lastweek), str(nextweek))
shifts_df = pd.DataFrame(shifts)
shifts_df['shift_start'] = pd.to_datetime(
    shifts_df['shift_start'], errors='coerce')
shifts_df['shift_end'] = pd.to_datetime(
    shifts_df['shift_end'], errors='coerce')

# merge with all_shifts_from_dropbox
merged_shifts_df = pd.concat([all_shifts_from_dropbox, shifts_df]).drop_duplicates(
    subset='scheduled_shift_id', keep='last').reset_index(drop=True)

merged_shifts_df.to_csv("all_shifts.csv", index=False)
print(merged_shifts_df.info())

time_index = pd.date_range(start=merged_shifts_df['shift_start'].min(
), end=merged_shifts_df['shift_end'].max(), freq='h')[:-1]
hourly_shifts_df = pd.DataFrame(index=time_index)
hourly_shifts_df.index.name = 'ds'
for _, shift in merged_shifts_df.iterrows():
    shift_start = pd.to_datetime(shift['shift_start']).ceil('h')
    shift_end = pd.to_datetime(shift['shift_end']).ceil('h')
    shift_hours = pd.date_range(
        start=shift_start, end=shift_end, freq='h')[:-1]
    for hour in shift_hours:
        if hour in hourly_shifts_df.index:
            hourly_shifts_df.at[hour, shift['shift_short_name']
                                ] = shift['first_name']+shift['last_name']
hourly_shifts_df.reset_index(inplace=True)
print(hourly_shifts_df.info())

hourly_shifts_df.to_csv("hourly_shifts.csv", index=False)


dropbox_app_key = os.environ.get("DROPBOX_APP_KEY")
dropbox_app_secret = os.environ.get("DROPBOX_APP_SECRET")
dropbox_refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")

# exchange the authorization code for an access token:
TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"
params = {
    "grant_type": "refresh_token",
    "refresh_token": dropbox_refresh_token,
    "client_id": dropbox_app_key,
    "client_secret": dropbox_app_secret
}
r = requests.post(TOKEN_URL, data=params)
# print(r.text)

dropbox_access_token = r.json()['access_token']

dbx = dropbox.Dropbox(dropbox_access_token)

upload(dbx, 'all_shifts.csv', 'shiftadmin', '',
            'all_shifts.csv', overwrite=True)

upload(dbx, 'hourly_shifts.csv', 'shiftadmin', '',
            'hourly_shifts.csv', overwrite=True)
