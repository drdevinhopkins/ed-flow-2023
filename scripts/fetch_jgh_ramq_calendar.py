#!/usr/bin/env python3
"""Fetch the official JGH RAMQ establishment holiday calendar.

RAMQ lets each establishment move the standard 13 professional holidays. This utility
uses the official RAMQ calendar page and selects Montréal (06) and establishment 0011X
(Hôpital général juif). It discovers controls by their option text rather than relying on
fragile ASP.NET-generated element ids.

Diagnostics are persisted at each stage so the scraper can be maintained if RAMQ changes
its legacy ASP.NET form.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

RAMQ_URL = (
    "https://www4.prod.ramq.gouv.qc.ca/ETA/SN/SNC_CalenFerie/"
    "SNC2_CnsulCalenFerieEtab_iut/CnsulCalenFerieEtab.aspx"
)
REGION_CODE = "06"
ESTABLISHMENT_CODE = "0011X"


def _norm(text: str) -> str:
    return " ".join((text or "").replace("\xa0", " ").split())


async def _bounded_postback_wait(page) -> None:
    """Allow an ASP.NET AutoPostBack/navigation to settle without waiting for networkidle."""
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=15_000)
    except PlaywrightTimeoutError:
        pass
    await page.wait_for_timeout(2_000)


async def _select_by_option_text(page, needle: str) -> dict[str, str]:
    """Select the first <select> containing an option whose text contains needle."""
    selects = page.locator("select")
    for i in range(await selects.count()):
        select = selects.nth(i)
        options = select.locator("option")
        for j in range(await options.count()):
            option = options.nth(j)
            text = _norm(await option.inner_text())
            value = await option.get_attribute("value") or ""
            if needle.casefold() in text.casefold():
                name = await select.get_attribute("name") or ""
                element_id = await select.get_attribute("id") or ""
                await select.select_option(value=value)
                await _bounded_postback_wait(page)
                return {
                    "select_index": str(i),
                    "select_name": name,
                    "select_id": element_id,
                    "option_text": text,
                    "option_value": value,
                }
    raise RuntimeError(f"Could not find option containing {needle!r}")


async def _describe_selects(page) -> list[dict[str, object]]:
    described: list[dict[str, object]] = []
    selects = page.locator("select")
    for i in range(await selects.count()):
        select = selects.nth(i)
        options = []
        option_nodes = select.locator("option")
        for j in range(await option_nodes.count()):
            option = option_nodes.nth(j)
            options.append(
                {
                    "text": _norm(await option.inner_text()),
                    "value": await option.get_attribute("value") or "",
                    "selected": bool(await option.evaluate("el => el.selected")),
                }
            )
        described.append(
            {
                "index": i,
                "name": await select.get_attribute("name") or "",
                "id": await select.get_attribute("id") or "",
                "options": options,
            }
        )
    return described


async def _snapshot(page, output_dir: Path, diagnostic: dict[str, object], stage: str) -> None:
    body_text = _norm(await page.locator("body").inner_text())
    diagnostic["stage"] = stage
    diagnostic["url"] = page.url
    diagnostic["date_candidates"] = {
        "dmy_slash": sorted(set(re.findall(r"\b\d{1,2}/\d{1,2}/\d{4}\b", body_text))),
        "ymd_dash": sorted(set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", body_text))),
        "dmy_dash": sorted(set(re.findall(r"\b\d{1,2}-\d{1,2}-\d{4}\b", body_text))),
    }
    (output_dir / "ramq-jgh-diagnostic.json").write_text(
        json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "ramq-jgh-page.txt").write_text(body_text + "\n", encoding="utf-8")
    (output_dir / "ramq-jgh-page.html").write_text(await page.content(), encoding="utf-8")
    try:
        await page.screenshot(path=str(output_dir / "ramq-jgh-page.png"), full_page=True)
    except Exception:
        pass


async def fetch(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic: dict[str, object] = {}
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(locale="fr-CA")
        try:
            await page.goto(RAMQ_URL, wait_until="domcontentloaded", timeout=60_000)
            await page.wait_for_timeout(1_500)
            diagnostic["initial_selects"] = await _describe_selects(page)
            await _snapshot(page, output_dir, diagnostic, "initial")

            diagnostic["region_selection"] = await _select_by_option_text(page, REGION_CODE)
            diagnostic["after_region_selects"] = await _describe_selects(page)
            await _snapshot(page, output_dir, diagnostic, "region_selected")

            diagnostic["establishment_selection"] = await _select_by_option_text(
                page, ESTABLISHMENT_CODE
            )
            diagnostic["after_establishment_selects"] = await _describe_selects(page)
            await _snapshot(page, output_dir, diagnostic, "establishment_selected")
        except Exception as exc:
            diagnostic["error"] = f"{type(exc).__name__}: {exc}"
            try:
                await _snapshot(page, output_dir, diagnostic, "error")
            finally:
                await browser.close()
            raise
        await browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output-ramq"))
    args = parser.parse_args()
    asyncio.run(fetch(args.output_dir))
