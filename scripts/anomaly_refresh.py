"""Validation and verified publishing for the anomaly refresh job."""
from io import BytesIO
from pathlib import Path

import pandas as pd

from validate_forecast_inputs import LOCAL_TZ, check_anomaly_ranges, check_ed_data


def now_local():
    return pd.Timestamp.now(tz=LOCAL_TZ).tz_localize(None).floor('min')


def require_pass(results):
    for result in results:
        print(f'[{"PASS" if result.ok else "FAIL"}] {result.name}: {result.detail}')
    failures = [result.name for result in results if not result.ok]
    if failures:
        raise ValueError('Anomaly refresh validation failed: ' + ', '.join(failures))


def validate_source(frame, *, now=None):
    require_pass(check_ed_data(
        frame, now_local=now if now is not None else now_local(),
        max_age_hours=3, continuity_hours=24,
    )[0])


def validate_ranges(frame, *, now=None):
    # Daily refreshes must retain a full day of headroom beyond the 24h forecast.
    current = now if now is not None else now_local()
    require_pass(check_anomaly_ranges(
        frame, forecast_start=current.floor('h') + pd.Timedelta(hours=1), horizon=48,
    ))


def publish_file(dbx, filename, *, verify_ranges=False):
    """Fail on upload errors and verify the exact bytes stored at the destination."""
    from dropbox.files import WriteMode

    data = Path(filename).read_bytes()
    if verify_ranges:
        validate_ranges(pd.read_csv(BytesIO(data)))
    path = '/' + Path(filename).name
    dbx.files_upload(data, path, mode=WriteMode.overwrite, mute=True)
    _, response = dbx.files_download(path)
    try:
        published = response.content
        if published != data:
            raise RuntimeError(f'Published file differs from generated file: {path}')
        if verify_ranges:
            validate_ranges(pd.read_csv(BytesIO(published)))
    finally:
        response.close()
    print(f'Published and verified {path}')
