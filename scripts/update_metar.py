import io
import os

import dropbox
import pandas as pd
import requests
from dotenv import load_dotenv

from utils import upload


load_dotenv()

dropbox_app_key = os.environ.get("DROPBOX_APP_KEY")
dropbox_app_secret = os.environ.get("DROPBOX_APP_SECRET")
dropbox_refresh_token = os.environ.get("DROPBOX_REFRESH_TOKEN")

token_url = "https://api.dropboxapi.com/oauth2/token"
params = {
    "grant_type": "refresh_token",
    "refresh_token": dropbox_refresh_token,
    "client_id": dropbox_app_key,
    "client_secret": dropbox_app_secret,
}
token_response = requests.post(token_url, data=params, timeout=30)
token_response.raise_for_status()
dropbox_access_token = token_response.json()["access_token"]
dbx = dropbox.Dropbox(dropbox_access_token)

# Read the canonical METAR history through the authenticated Dropbox API.
# This avoids transient HTML/error responses from the public share URL being
# handed to pandas as if they were CSV.
_, metar_response = dbx.files_download("/metar/full_metar_data.csv")
metar_df = pd.read_csv(io.BytesIO(metar_response.content))
metar_df["valid"] = pd.to_datetime(
    metar_df["valid"], format="mixed", errors="coerce"
)
metar_df = metar_df.dropna(subset=["valid"]).copy()
print("METAR data shape:", metar_df.shape)

most_recent_date = metar_df["valid"].max()
if pd.isna(most_recent_date):
    raise ValueError("METAR history contains no valid timestamps.")
print("Most recent date in METAR data:", most_recent_date)

# Always overlap the existing history by one day so revised observations are
# picked up. If the history is stale, fetch all missing dates through today in
# bounded chunks rather than advancing only one week per hourly run.
fetch_start = most_recent_date.normalize() - pd.Timedelta(days=1)
fetch_end = (
    pd.Timestamp.now(tz="America/Montreal")
    .normalize()
    .tz_localize(None)
    + pd.Timedelta(days=1)
)
print("METAR catch-up range:", fetch_start, "to", fetch_end)

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

recent_chunks = []
chunk_start = fetch_start
chunk_number = 0
while chunk_start < fetch_end:
    chunk_end = min(chunk_start + pd.Timedelta(days=30), fetch_end)
    chunk_number += 1
    url = base_url.format(
        y1=chunk_start.year,
        m1=chunk_start.month,
        d1=chunk_start.day,
        y2=chunk_end.year,
        m2=chunk_end.month,
        d2=chunk_end.day,
    )
    print(
        f"Fetching METAR chunk {chunk_number}: "
        f"{chunk_start.date()} to {chunk_end.date()}"
    )
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    chunk_df = pd.read_csv(io.StringIO(response.text))
    chunk_df["valid"] = pd.to_datetime(
        chunk_df["valid"], format="mixed", errors="coerce"
    )
    chunk_df = chunk_df.dropna(subset=["valid"])
    print(f"METAR chunk {chunk_number} rows:", len(chunk_df))
    recent_chunks.append(chunk_df)

    if chunk_end >= fetch_end:
        break
    # Reuse the boundary date intentionally; duplicates are removed below.
    chunk_start = chunk_end

if recent_chunks:
    recent_metar_df = pd.concat(recent_chunks, ignore_index=True)
    metar_df = (
        pd.concat([metar_df, recent_metar_df], ignore_index=True)
        .drop_duplicates(subset=["valid"], keep="last")
        .sort_values("valid")
        .reset_index(drop=True)
    )

print("Updated METAR dataframe shape:", metar_df.shape)
print("Updated most recent METAR timestamp:", metar_df["valid"].max())

metar_df.to_csv("full_metar_data.csv", index=False)

# Reuse the authenticated Dropbox client created above.
upload(
    dbx,
    "full_metar_data.csv",
    "metar",
    "",
    "full_metar_data.csv",
    overwrite=True,
)
