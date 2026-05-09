# secrets/

**Not committed to git.** Local-only directory for OAuth client
JSONs, service-account keys, and any other credential files that
must never reach version control.

The `.gitignore` covers `client_secret_*.json`, `*.pem`, `*.key`,
`credentials.json`, and the `secrets/` directory itself.

For environment variables (API keys, etc.) use `.env` at the
project root instead — also gitignored.
