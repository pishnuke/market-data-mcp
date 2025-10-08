# market-data-mcp
Market & Options Data MCP ("market-data") — FastAPI server

A deployable, minimal, vendor-pluggable service that exposes endpoints used by an LLM
or any client to fetch OHLCV, option chains, Greeks, basic corporate events, and to
assemble a training dataset with alignment/caching. Uses yfinance as a default
provider so it runs out-of-the-box; swap in Polygon/IEX/etc by implementing the
DataProvider interface below.

Run locally:
```
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000
```

Docker:
```
  docker build -t market-data-mcp:latest .
  docker run -p 8000:8000 --name market-data-mcp market-data-mcp:latest
```

Example curl:
```
  curl "http://localhost:8000/get_ohlcv?symbol=AAPL&timeframe=1d&start=2024-01-01&end=2024-06-30"
  curl -X POST http://localhost:8000/make_dataset -H 'Content-Type: application/json' -d '{
    "symbols":["AAPL","NVDA"],
    "features":["ohlcv(1d,120d)","rv_park(5d)","ret_1d","iv30"],
    "horizon":"1d","window":"180d","align":"market_close"
  }'
```

### Notes
- Image will publish to `ghcr.io/pishnuke/market-data-mcp:edge` on `master`, plus a `sha-<short>` tag, and to `ghcr.io/pishnuke/market-data-mcp:<tag>` when you push a Git tag like `v0.1.0`.
- Ensure your repo is **public** or that consumers have permission to pull from GHCR. For private repos, consumers need a token.
- The GHCR repository name is lowercase; if your GitHub org/repo has uppercase, GHCR normalizes it.

Point your MCP client at `mcp.json` (or the running URL) and call tools like `/get_ohlcv`.
