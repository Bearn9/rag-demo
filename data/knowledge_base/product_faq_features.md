# Product FAQ: Features and Data Connectors

Answers to common questions about what the Bearn Analytics platform can do.

## Data connectors

Bearn Analytics natively connects to the following data sources: PostgreSQL, Snowflake, BigQuery, MySQL, and Amazon Redshift. Custom connectors for other sources can be built using the platform's REST API.

## Data refresh rates

How often your connected data refreshes depends on your plan tier:

- **Starter and Team plans:** data refreshes every **15 minutes**.
- **Enterprise plan:** **real-time** streaming refresh via change-data-capture, where supported by the source database.

Manual "refresh now" is available on all plans for on-demand updates outside the scheduled interval.

## API rate limits

The Bearn API enforces the following rate limits per API key:

- **Standard tier (Starter/Team):** 100 requests per minute.
- **Enterprise tier:** 1,000 requests per minute, with the option to request a further increase from your account manager.

Requests beyond the limit receive an HTTP 429 response with a `Retry-After` header.

## Dashboards and sharing

Dashboards support drag-and-drop chart building, scheduled email/Slack digests, and public share links (Team plan and above). Row-level security for shared dashboards is available on the Enterprise plan.
