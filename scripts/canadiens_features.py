#!/usr/bin/env python3
"""Experimental NHL schedule covariates; never imports into production routes."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

LOCAL_TZ = 'America/Montreal'
API = 'https://api-web.nhle.com/v1/club-schedule-season/MTL/{season}'


def download_schedule(first: int, last: int) -> dict:
    """Fetch public schedules, discarding outcomes, scores and actual end times."""
    if last < first:
        raise ValueError('last season must follow first season')
    games = []
    sources = []
    for year in range(first, last + 1):
        season = year * 10000 + year + 1
        url = API.format(season=season)
        with urlopen(Request(url, headers={'User-Agent': 'ed-flow-research/1.0'}), timeout=45) as response:
            payload = json.load(response)
        if int(payload.get('currentSeason', -1)) != season:
            raise ValueError(f'NHL returned an unexpected season for {url}')
        sources.append(url)
        for game in payload['games']:
            if game['gameType'] not in (2, 3):  # exclude preseason
                continue
            if game.get('gameScheduleState') != 'OK':
                raise ValueError(f'Unresolved schedule state for game {game["id"]}')
            games.append(dict(game_id=game['id'], season=season,
                              start_utc=game['startTimeUTC'],
                              home=int(game['homeTeam']['abbrev'] == 'MTL'),
                              playoffs=int(game['gameType'] == 3)))
    if len({g['game_id'] for g in games}) != len(games):
        raise ValueError('Duplicate NHL game IDs')
    return dict(schema_version=1, team='MTL',
                fetched_at=datetime.now(timezone.utc).isoformat(),
                coverage_start=f'{first}-07-01', coverage_end=f'{last + 1}-07-01',
                schedule_vintage='retrospective_latest', sources=sources,
                games=sorted(games, key=lambda game: game['start_utc']))


def schedule_features(stamps, schedule: dict, duration_hours: float = 3.0) -> pd.DataFrame:
    """Point-in-time windows around scheduled starts, using a fixed game duration.

    Naive timestamps are Montreal wall time. Latest historical schedules are NOT
    evidence of the schedule known at an old issuance time: vintage is explicit.
    """
    if duration_hours <= 0 or not np.isfinite(duration_hours):
        raise ValueError('duration_hours must be positive and finite')
    local = pd.DatetimeIndex(pd.to_datetime(stamps))
    if local.tz is None:
        local = local.tz_localize(LOCAL_TZ, ambiguous='raise', nonexistent='raise')
    else:
        local = local.tz_convert(LOCAL_TZ)
    naive = local.tz_localize(None)
    start = pd.Timestamp(schedule['coverage_start'])
    end = pd.Timestamp(schedule['coverage_end'])
    if naive.isna().any() or ((naive < start) | (naive >= end)).any():
        raise ValueError('Timestamps are invalid or outside verified schedule coverage')
    names = ['game_day', 'home_game_day', 'playoff_day', 'pre_2h',
             'scheduled_game', 'post_3h', 'home_during', 'playoff_during',
             'scheduled_hours_remaining_today']
    out = pd.DataFrame(0.0, index=range(len(local)), columns=['sports_' + n for n in names])
    for game in schedule['games']:
        tip = pd.Timestamp(game['start_utc'])
        if tip.tzinfo is None:
            raise ValueError('Game start must have an explicit UTC offset')
        tip = tip.tz_convert(LOCAL_TZ)
        delta = np.asarray((local - tip).total_seconds() / 3600)
        same_day = np.asarray(naive.normalize() == tip.tz_localize(None).normalize())
        during = (delta >= 0) & (delta < duration_hours)
        for name, mask in [('game_day', same_day), ('home_game_day', same_day & bool(game['home'])),
                           ('playoff_day', same_day & bool(game['playoffs'])),
                           ('pre_2h', (delta >= -2) & (delta < 0)),
                           ('scheduled_game', during),
                           ('post_3h', (delta >= duration_hours) & (delta < duration_hours + 3)),
                           ('home_during', during & bool(game['home'])),
                           ('playoff_during', during & bool(game['playoffs']))]:
            out['sports_' + name] = np.maximum(out['sports_' + name], mask.astype(float))
        # Exposure still ahead at the current cutoff; clips late games at local midnight.
        midnight = (naive.normalize() + pd.Timedelta(days=1)).tz_localize(LOCAL_TZ)
        finish = tip + pd.Timedelta(hours=duration_hours)
        remaining = np.maximum(0, (np.minimum(midnight.asi8, finish.value)
                                   - np.maximum(local.asi8, tip.value)) / 3.6e12)
        out['sports_scheduled_hours_remaining_today'] += remaining
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-season', type=int, default=2021)
    parser.add_argument('--last-season', type=int, default=2025)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    schedule = download_schedule(args.first_season, args.last_season)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(schedule, indent=2) + '\n')
    print(f'Saved {len(schedule["games"])} regular-season/playoff games to {args.output}')


if __name__ == '__main__':
    main()
