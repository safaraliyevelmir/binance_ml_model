# pip install aiohttp pandas pyarrow
import asyncio
import io
import os
import zipfile

import aiohttp
import pandas as pd

CHUNK_SIZE = 256 * 1024  # 256 KB per chunk
MAX_RETRIES = 5
CONCURRENCY = 3  # low for slow connections
SYMBOL = "SOLUSDT"

def years_month_generator():
    for year in range(2021, 2026):
        for month in range(1, 13):
            if (year, month) < (2021, 8):
                continue  # SOLUSDT launch period — illiquid, non-representative
            yield year, month


async def fetch_with_resume(session: aiohttp.ClientSession, url: str, tmp_path: str) -> bytes | None:
    """Download url into tmp_path in chunks, resuming if the file already exists partially."""
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)

    for attempt in range(1, MAX_RETRIES + 1):
        downloaded = os.path.getsize(tmp_path) if os.path.exists(tmp_path) else 0
        headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

        try:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=None, sock_read=60)) as r:
                if r.status == 416:
                    # Range not satisfiable — file already fully downloaded
                    break
                if r.status not in (200, 206):
                    print(f"  HTTP {r.status}, skipping.")
                    return None

                mode = "ab" if downloaded and r.status == 206 else "wb"
                with open(tmp_path, mode) as f:
                    async for chunk in r.content.iter_chunked(CHUNK_SIZE):
                        f.write(chunk)
            break

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}. Retrying...")
            await asyncio.sleep(2 ** attempt)
    else:
        print(f"All retries exhausted for {url}")
        return None

    with open(tmp_path, "rb") as f:
        return f.read()


async def download_binance_agg_trades(session: aiohttp.ClientSession, symbol: str, year: int, month: int):
    out_path = f"data/raw/{symbol}_{year}_{month:02d}.parquet"
    if os.path.exists(out_path):
        print(f"Already exists, skipping: {year}-{month:02d}")
        return pd.read_parquet(out_path)

    url = (
        f"https://data.binance.vision/data/futures/um/monthly/aggTrades/"
        f"{symbol}/{symbol}-aggTrades-{year}-{month:02d}.zip"
    )
    tmp_path = f"data/raw/.tmp_{symbol}_{year}_{month:02d}.zip"
    print(f"Downloading: {year}-{month:02d}")

    content = await fetch_with_resume(session, url, tmp_path)
    if content is None:
        return None

    column_names = [
        'agg_trade_id', 'price', 'quantity', 'first_trade_id', 
        'last_trade_id', 'timestamp', 'is_buyer_maker'
    ]

    try:
        z = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        print(f"Corrupt zip for {year}-{month:02d}, removing tmp file.")
        os.remove(tmp_path)
        return None

    df = pd.read_csv(
        z.open(z.namelist()[0]),
    )
    if "timestamp" not in df.columns and "transact_time" not in df.columns:
        df = pd.read_csv(
            z.open(z.namelist()[0]),
            names=column_names,
            dtype={'price': 'float32', 'quantity': 'float32'},
        )
    else:
        df = pd.read_csv(
            z.open(z.namelist()[0]),
            dtype={'price': 'float32', 'quantity': 'float32'},
        )    

    df.rename(columns={'transact_time': 'timestamp'}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['dollar_volume'] = df['price'] * df['quantity']

    os.makedirs("data/raw", exist_ok=True)
    df.to_parquet(out_path)
    os.remove(tmp_path)
    print(f"Saved {year}-{month:02d}: {len(df):,} trades, {df['dollar_volume'].sum()/1e9:.1f}B USDT")
    return df


async def main():
    pairs = list(years_month_generator())

    connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(CONCURRENCY)

        async def bounded(y, m):
            async with sem:
                return await download_binance_agg_trades(session, SYMBOL, y, m)

        await asyncio.gather(*[bounded(y, m) for y, m in pairs])


if __name__ == "__main__":
    asyncio.run(main())
