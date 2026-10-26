# Regional ED pressure collection

`scripts/update_regional_ed_pressure.py` archives observed Montréal peer-ED pressure,
excluding JGH. Collection tries these official sources in order:

1. The MSSS rolling seven-day CSV.
2. Données Québec `datastore_search` for resource
   `b256f87f-40ec-4c79-bdba-a23e9c50e741`.
3. The same resource's official CSV export at
   `https://www.donneesquebec.ca/recherche/datastore/dump/b256f87f-40ec-4c79-bdba-a23e9c50e741?format=csv`.
4. The paginated official Québec.ca emergency-status page.

The datastore/export can contain only the latest snapshot, rather than seven days.
It cannot recover missed hours that the upstream source no longer supplies. Existing
archive history is retained when new observations are merged.

The datastore's `Heure_de_l'extraction_(image)` can be a clock time only. It is combined
with the date of `Mise_a_jour`, rolling back one day when the extraction clock precedes
a publication after midnight. Undated rows are excluded; the collector's current date
is never substituted. ISO dates retain year-month-day order.

Québec.ca pages must contain an observation timestamp. Access-challenge pages and
failed requests abort collection, including failures after earlier pages succeeded.
No partial result from such a failed pagination run is published. The existing
four-hour source-age check still runs before archive writes or Dropbox uploads.

To validate without publishing:

```sh
python tests/test_regional_ed_pressure.py
python scripts/update_regional_ed_pressure.py --no-dropbox --output /tmp/regional-ed-pressure.csv
```

These changes do not alter scheduling, Dropbox destinations, or forecast feature flags.
