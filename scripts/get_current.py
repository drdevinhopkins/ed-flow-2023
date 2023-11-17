import requests
import camelot
import datetime
import os
import pandas as pd
from deta import Deta
from dotenv import load_dotenv

import constants

load_dotenv()

deta = Deta(os.environ.get("DETA_PROJECT_KEY"))

data = deta.Drive("data")

url = 'https://www.dropbox.com/s/ckijmipu33z3feg/HourlyReport.pdf?dl=1'
r = requests.get(url, allow_redirects=True)
open('hourlyreport.pdf', 'wb').write(r.content)

tables = camelot.read_pdf('hourlyreport.pdf', flavor='stream', pages='1', columns=['41, 57, 75, 97.5, 115.5, 138, 160, 187, 202, 220, 238, 256, 279, 301, 323, 342, 360, 381, 400, 425, 443, 465, 483, 501, 519, 537, 559.5, 577.5, 600, 627.5, 652, 667, 690, 706.5, 730, 753, 778, 798, 822, 839.5, 855.5, 881, 906, 930, 956, 978'])

df = tables[0].df.loc[2:].reset_index(drop=True)
df.columns = column_names
df.dateflg = df.dateflg[df.dateflg.str.strip() != '']
df.dateflg = df.dateflg.ffill()
for column in df.columns.tolist():
    if column in ['dateflg']:
        continue
    df[column] = df[column].astype('float').astype('int')
df["ds"] = pd.to_datetime(
    df["dateflg"] + " " + (df["timeflg"] - 1).astype(str) + ":00", format='mixed') + datetime.timedelta(hours=1)
df = df.set_index('ds').reset_index().drop(['dateflg','timeflg'], axis=1)

df.to_csv('current.csv', index=False)

data.put(name='current.csv', path='current.csv')

