# @psyb0t/predictalot

An OpenClaw/MCP plugin that connects your agent to a self-hosted
[predictalot](https://github.com/psyb0t/docker-predictalot) forecasting API
over the [Model Context Protocol](https://modelcontextprotocol.io).

predictalot already serves a Streamable-HTTP MCP endpoint at `/mcp`. This
package is a thin stdio↔HTTP bridge (via
[`mcp-remote`](https://www.npmjs.com/package/mcp-remote)) for MCP clients that
speak local stdio servers — it forwards everything to your running predictalot
instance and authenticates with your bearer token when the server requires one.

> predictalot is **self-hosted**. This plugin does not ship the forecasting
> engine — it connects to a predictalot server that **you** run. See the
> [predictalot repo](https://github.com/psyb0t/docker-predictalot) to stand
> one up.

## Tools

The MCP surface mirrors predictalot's foundation-model timeseries API — one
named tool per (forecast type, model) cell across 5 zero-shot forecasters
(`chronos-2`, `timesfm-2.5`, `moirai-2`, `toto-1`, `sundial-base-128m`), plus
a per-type weighted ensemble tool and a per-type model-listing tool. Forecast
types: `univariate`, `multivariate`, `covariates_past`, `covariates_future`,
`covariates_both`, `samples`. Tabular ML (train/forecast on your own
engineered features) is HTTP-only and not exposed over MCP.

## Configuration

| Env var | Required | Description |
|---|---|---|
| `PREDICTALOT_URL` | yes | Base URL of your running predictalot server, e.g. `http://localhost:8080`. The bridge appends `/mcp`. |
| `PREDICTALOT_AUTH_TOKENS` | no | Bearer token — only if the predictalot server was started with `PREDICTALOT_AUTH_TOKENS` set. |

## Install

Install it into your OpenClaw agent from ClawHub:

```bash
openclaw plugins install clawhub:@psyb0t/predictalot
```

Then set `PREDICTALOT_URL` (and `PREDICTALOT_AUTH_TOKENS` if your server uses
auth) in the plugin's environment.

## Native remote MCP (no install)

If your MCP client already supports **remote** Streamable-HTTP servers, you
don't need this bridge — point the client straight at
`$PREDICTALOT_URL/mcp` with an `Authorization: Bearer <token>` header.

## License

MIT. See [LICENSE](LICENSE).
