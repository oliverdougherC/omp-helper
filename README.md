# omp-helper (omp-providers)

A single-file Python helper (stdlib + PyYAML) for configuring **custom model
providers** in [omp](https://omp.sh), which has no native custom-provider UI.
Custom providers live in `~/.omp/agent/models.yml` under the `providers:` root
key; this tool detects your omp install (or installs it), probes OpenAI-compatible
endpoints for their model rosters, and writes a valid `models.yml` — plus
one-file export/import to clone your provider config onto another machine.

Works on Linux, macOS, and WSL. The agent directory is resolved with
`omp config path`, so named profiles, `PI_CODING_AGENT_DIR`, and Linux XDG
layouts are honored automatically.

## Install omp (if missing)

    curl -fsSL https://omp.sh/install | sh

Or let the tool do it: `python3 omp_helper.py install-omp`

## Usage

    python3 omp_helper.py                # interactive guided menu
    python3 omp_helper.py status         # omp + models.yml detection
    python3 omp_helper.py list           # providers currently defined
    python3 omp_helper.py add            # guided wizard
    python3 omp_helper.py add --id my-llm --base-url http://host:8000/v1 \
        --api openai-completions --auth-env MY_LLM_KEY --discover \
        --model qwen3.8 --context 262144 --max-tokens 32768 --reasoning --vision
    python3 omp_helper.py doctor --probe # validate schema + reach endpoints
    python3 omp_helper.py remove my-llm
    python3 omp_helper.py export ~/omp-providers.yml   # clone to another machine
    python3 omp_helper.py import ~/omp-providers.yml   # merge them back in

`add` flags: `--keyless` (local server, `auth: none`), `--api-key <literal>`
(stored in the file), `--auth-header` (Bearer injection for gateways),
`--developer-role/--no-developer-role` (compat), `--overwrite`.

## Cloning to a new installation

    machine A:  python3 omp_helper.py export my-providers.yml
    machine B:  python3 omp_helper.py import my-providers.yml

Prefer `--auth-env` (environment-variable name) over `--api-key` (literal) for
keyed providers so secrets don't travel in the exported file.

## What it does NOT manage

- OAuth/`/login` credentials — stored in `~/.omp/agent/agent.db`
- `disabledProviders`, model roles — `~/.omp/agent/config.yml`
- Built-in providers (anthropic, openai, …) — bundled with the omp binary

## Schema notes (verified against omp docs)

- Written keys follow "Model and Provider Configuration" (`omp://models.md`):
  `baseUrl`, `api`, `apiKey` (env-var-name-or-literal), `auth: none`,
  `authHeader`, `compat.supportsDeveloperRole`, `models[]` with `id`, `name`,
  `reasoning`, `input`, `contextWindow`, `maxTokens`, `cost`.
- Keyless providers use `auth: none` (documented form), not a literal
  `apiKey: none`.
- Every write backs up the previous file (`models.yml.bak-<timestamp>`).
- `doctor` cross-checks with `omp models`: a schema error makes the registry
  skip the custom file silently, so this catches it early.

## Requirements

Python ≥ 3.10, PyYAML (`python3 -m pip install pyyaml`).
