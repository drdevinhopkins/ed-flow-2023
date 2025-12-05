import os
import requests
from utils import upload
import dropbox
from dotenv import load_dotenv
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd

load_dotenv()

hist_weather_df = pd.read_csv('https://www.dropbox.com/scl/fi/gmhwwld9z9yychg4r0yuk/weather.csv?rlkey=66c78m90aviamr0x0uu72pfr8&raw=1')
hist_weather_df.ds = pd.to_datetime(hist_weather_df.ds)


# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": 45.5088,
	"longitude": -73.5878,
	"hourly": ["temperature_2m", "precipitation_probability", "precipitation", "rain", "snowfall", "snow_depth", "cloud_cover_high", "cloud_cover_mid", "cloud_cover_low", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "apparent_temperature", "dew_point_2m", "relative_humidity_2m", "weather_code", "pressure_msl", "surface_pressure"],
	"timezone": "America/New_York",
	"past_days": 3,
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_precipitation_probability = hourly.Variables(1).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
hourly_rain = hourly.Variables(3).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(4).ValuesAsNumpy()
hourly_snow_depth = hourly.Variables(5).ValuesAsNumpy()
hourly_cloud_cover_high = hourly.Variables(6).ValuesAsNumpy()
hourly_cloud_cover_mid = hourly.Variables(7).ValuesAsNumpy()
hourly_cloud_cover_low = hourly.Variables(8).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(9).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(10).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(11).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(12).ValuesAsNumpy()
hourly_apparent_temperature = hourly.Variables(13).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(14).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(15).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(16).ValuesAsNumpy()
hourly_pressure_msl = hourly.Variables(17).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(18).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["precipitation_probability"] = hourly_precipitation_probability
hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain
hourly_data["snowfall"] = hourly_snowfall
hourly_data["snow_depth"] = hourly_snow_depth
hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
hourly_data["apparent_temperature"] = hourly_apparent_temperature
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["weather_code"] = hourly_weather_code
hourly_data["pressure_msl"] = hourly_pressure_msl
hourly_data["surface_pressure"] = hourly_surface_pressure

hourly_dataframe = pd.DataFrame(data = hourly_data)

hourly_dataframe = hourly_dataframe.rename(columns={'date': 'ds'})
hourly_dataframe['ds'] = hourly_dataframe['ds'].dt.tz_localize(None)

new_weather_df = pd.concat([hist_weather_df, hourly_dataframe], ignore_index=True)

new_weather_df = new_weather_df.drop_duplicates(subset=['ds'], keep='last')

new_weather_df.to_csv('weather.csv', index=False)


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

upload(dbx, 'weather.csv', 'weather', '',
            'weather.csv', overwrite=True)
