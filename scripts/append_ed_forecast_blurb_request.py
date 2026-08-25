#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from generate_ed_forecast_blurb import COLUMNS, append_row, get_dbx

REQUIRED_KEYS = set(COLUMNS)


def load_request(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise RuntimeError('Request must be a JSON object')

    missing = REQUIRED_KEYS - set(payload)
    extra = set(payload) - REQUIRED_KEYS
    if missing or extra:
        raise RuntimeError(
            f'Request keys must exactly match CSV columns; missing={sorted(missing)} extra={sorted(extra)}'
        )

    row = {column: str(payload[column]).strip() for column in COLUMNS}
    if not row['generated_at_local']:
        raise RuntimeError('generated_at_local is required')
    if not row['forecast_data_time_local']:
        raise RuntimeError('forecast_data_time_local is required')
    if not row['blurb']:
        raise RuntimeError('blurb is required')
    if not row['source_status']:
        raise RuntimeError('source_status is required')
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Append an exact pre-generated ED forecast blurb request to Dropbox.'
    )
    parser.add_argument(
        '--request',
        default='.github/ed_forecast_blurb_request.json',
        help='Path to request JSON committed by the blurb author',
    )
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    row = load_request(Path(args.request))

    with open('ed_forecast_blurb_row.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(row)

    print(row['blurb'])
    print('oncall_recommendation=' + row['oncall_recommendation'])
    print('source_status=' + row['source_status'])

    if args.dry_run:
        print('Dry run: request validated; Dropbox not modified.')
        return 0

    dbx = get_dbx()
    appended = append_row(dbx, row)
    print('append_result=' + ('appended' if appended else 'duplicate_skipped'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
