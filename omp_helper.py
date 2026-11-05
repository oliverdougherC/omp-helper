#!/usr/bin/env python3
"""omp-providers — configure custom OpenAI-compatible providers for omp.

omp has no native "custom provider" UI: providers are declared in
`~/.omp/agent/models.yml` under a `providers:` root key (see the omp docs:
"Model and Provider Configuration"). This helper detects your omp install,
walks you through adding an OpenAI/Anthropic-compatible endpoint, probes the
endpoint for its model list, and writes a valid models.yml — plus one-file
export/import so you can clone your provider config onto another machine.

Usage:
    python3 omp_helper.py                      # interactive menu
    python3 omp_helper.py status               # omp + config detection
    python3 omp_helper.py list                 # providers in models.yml
    python3 omp_helper.py add [opts]           # add a provider (wizard if no opts)
    python3 omp_helper.py remove <id>          # remove a provider
    python3 omp_helper.py doctor               # validate models.yml + probe endpoints
    python3 omp_helper.py export <path>        # copy models.yml out (transferable)
    python3 omp_helper.py import <path>        # merge a models.yml in
    python3 omp_helper.py install-omp          # run the official installer

Key flags for `add` (non-interactive when given):
    --id ID --base-url URL [--api openai-completions|openai-responses|anthropic-messages]
    [--auth-env VAR | --api-key LITERAL | --keyless] [--auth-header]
    [--discover] [--model ID] [--context N] [--max-tokens N] [--reasoning] [--vision]
    [--developer-role BOOL] [--overwrite]
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write(
        "omp-providers needs PyYAML:  python3 -m pip install pyyaml\n"
    )
    sys.exit(2)

PROBE_TIMEOUT = 8.0
DEFAULT_MODELS_YML_NAME = "models.yml"
VALID_APIS = ("openai-completions", "openai-responses", "anthropic-messages")
INSTALL_CMD = "curl -fsSL https://omp.sh/install | sh"


# --------------------------------------------------------------------------
# omp / config detection
# --------------------------------------------------------------------------

def omp_binary() -> Path | None:
    p = shutil.which("omp")
    if p:
        return Path(p)
    for cand in (Path.home() / ".local/bin/omp", Path("/usr/local/bin/omp"),
                 Path.home() / ".bun/bin/omp"):
        if cand.exists():
            return cand
    return None


def omp_version(binary: Path) -> str | None:
    try:
        out = subprocess.run([str(binary), "--version"], capture_output=True,
                             text=True, timeout=10)
        line = (out.stdout or out.stderr).strip().splitlines()
        return line[0] if line else None
    except Exception:
        return None


def agent_dir() -> Path:
    """Resolve the active agent directory. Prefer `omp config path`, which
    honors PI_CODING_AGENT_DIR, named profiles, and XDG layouts on Linux;
    fall back to the documented default when omp isn't installed yet."""
    b = omp_binary()
    if b:
        try:
            out = subprocess.run([str(b), "config", "path"], capture_output=True,
                                 text=True, timeout=10)
            line = (out.stdout or "").strip()
            if out.returncode == 0 and line:
                return Path(line).expanduser()
        except Exception:  # noqa: BLE001 - older builds may lack the subcommand
            pass
    override = os.environ.get("PI_CODING_AGENT_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".omp" / "agent"


def models_yml_path(create_parent: bool = False) -> Path:
    d = agent_dir()
    if create_parent:
        d.mkdir(parents=True, exist_ok=True)
    for name in ("models.yml", "models.yaml"):
        p = d / name
        if p.exists():
            return p
    return d / DEFAULT_MODELS_YML_NAME


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        sys.exit(f"error: {path} is not a mapping at the root")
    return data


def save_config(path: Path, cfg: dict) -> Path:
    backup = None
    if path.exists():
        backup = path.with_name(path.name + ".bak-" + time.strftime("%Y%m%d-%H%M%S"))
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# omp custom providers — managed by omp-providers (edits preserved).\n"
              "# Docs: 'Model and Provider Configuration' (models.yml). Only the\n"
              "# `providers:` root key configures model behavior.\n")
    body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, indent=2)
    path.write_text(header + body)
    return backup


# --------------------------------------------------------------------------
# endpoint discovery (GET .../models)
# --------------------------------------------------------------------------

def candidate_model_urls(base_url: str, api: str) -> list[str]:
    base = base_url.rstrip("/")
    urls = []
    if api == "anthropic-messages":
        # Anthropic wire: models live at {root}/v1/models; if the user gave a
        # root ending in /v1, probe both spellings.
        if base.endswith("/v1"):
            urls = [base + "/models"]
        else:
            urls = [base + "/v1/models"]
    else:
        if base.endswith("/v1"):
            urls = [base + "/models", base + "/v1/models"]
        else:
            urls = [base + "/v1/models", base + "/models"]
    return urls


def probe_models(base_url: str, api: str, key: str | None,
                 auth_header: bool) -> tuple[list[dict], str | None, str | None]:
    """Return (models, normalized_base_url, error). Each model dict uses omp's
    models.yml schema (id, name, reasoning, input, contextWindow, maxTokens)."""
    last_err = None
    for url in candidate_model_urls(base_url, api):
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        if api == "anthropic-messages":
            req.add_header("anthropic-version", "2023-06-01")
            if key and key.lower() != "none":
                req.add_header("x-api-key", key)
        elif key and key.lower() != "none":
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as e:  # noqa: BLE001 - report any probe failure
            last_err = (last_err + f"\n{' ' * 10}{url}: {e}") if last_err else f"{url}: {e}"
            continue
        rows = payload.get("data") if isinstance(payload, dict) else None
        if rows is None and isinstance(payload, dict):
            rows = payload.get("models")
        if not isinstance(rows, list):
            last_err = f"{url}: unexpected payload shape"
            continue
        models = [normalize_row(r) for r in rows if isinstance(r, dict) and r.get("id")]
        return models, url[: -len("/models")], None
    return [], None, last_err


def advertised_efforts(r: dict) -> list[str] | None:
    """Effort ladder the row states about itself (omp's advertisedReasoning
    spellings plus the `thinking` block)."""
    thinking = r.get("thinking") if isinstance(r.get("thinking"), dict) else None
    for src in ((thinking or {}).get("efforts"),
                r.get("supported_reasoning_efforts"),
                r.get("supported_reasoning_levels")):
        if isinstance(src, list) and src:
            return [str(e) for e in src]
    return None


def infer_thinking_format(model_id: str) -> dict | None:
    """Wire shaping for known thinking families, mirroring the catalog rules
    omp applies to built-in rows (e.g. llama.cpp.kdl `thinkingFormat`)."""
    mid = model_id.lower()
    if "qwen" in mid:
        return {"thinkingFormat": "qwen-chat-template",
                "qwenTemplateReasoningEffort": True}
    return None


def normalize_row(r: dict) -> dict:
    m = {"id": r["id"], "name": r.get("name") or r["id"]}
    limits = r.get("limits") if isinstance(r.get("limits"), dict) else {}
    ctx = first_num(r.get("max_model_len"), r.get("context_length"),
                    r.get("context_window"), limits.get("max_input_tokens"))
    out = first_num(r.get("max_output_tokens"), r.get("max_completion_tokens"),
                    limits.get("max_output_tokens"),
                    (r.get("top_provider") or {}).get("max_completion_tokens")
                    if isinstance(r.get("top_provider"), dict) else None)
    if ctx:
        m["contextWindow"] = ctx
    if out:
        m["maxTokens"] = out

    efforts = advertised_efforts(r)
    supported = r.get("supported_parameters") or []
    reasoning = r.get("reasoning")
    if reasoning is None:
        # Absent flag: efforts advertised (or thinking block present) imply
        # a reasoning-capable server; a `reasoning: false` row still wins.
        reasoning = bool(efforts) or isinstance(r.get("thinking"), dict) \
            or (isinstance(supported, list)
                and any("reasoning" in str(s) for s in supported))
    m["reasoning"] = bool(reasoning)

    thinking = r.get("thinking") if isinstance(r.get("thinking"), dict) else None
    if m["reasoning"]:
        if thinking and (thinking.get("mode") or thinking.get("efforts")):
            block = {}
            if thinking.get("mode"):
                block["mode"] = thinking["mode"]
            if isinstance(thinking.get("efforts"), list) and thinking["efforts"]:
                block["efforts"] = [str(e) for e in thinking["efforts"]]
            for k in ("requiresEffort", "effortMap", "defaultLevel"):
                if thinking.get(k) is not None:
                    block[k] = thinking[k]
            m["thinking"] = block
        elif efforts:
            m["thinking"] = {"mode": "effort", "efforts": efforts}
    elif efforts:
        m["thinking"] = {"mode": "effort", "efforts": efforts}

    compat = {}
    if efforts:
        compat["supportsReasoningEffort"] = True
        compat.update(infer_thinking_format(r["id"]))
    if compat:
        m["compat"] = compat

    arch = r.get("architecture") if isinstance(r.get("architecture"), dict) else {}
    modality = (arch.get("input_modalities") or r.get("input_modalities")
                or r.get("modality") or r.get("input"))
    vision = "image" in str(arch) or (isinstance(modality, list)
                                      and ("image" in modality or "vision" in modality))
    m["input"] = ["text", "image"] if vision else ["text"]
    return m


def first_num(*vals):
    for v in vals:
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return int(v)
    return None


# --------------------------------------------------------------------------
# small prompt helpers
# --------------------------------------------------------------------------

def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        v = input(f"{prompt}{suffix}: ").strip()
        if v:
            return v
        if default is not None:
            return default
        print("  (required)")


def ask_bool(prompt: str, default: bool) -> bool:
    d = "Y/n" if default else "y/N"
    v = input(f"{prompt} [{d}]: ").strip().lower()
    if not v:
        return default
    return v in ("y", "yes")


def ask_int(prompt: str, default: int) -> int:
    while True:
        v = ask(prompt, str(default))
        try:
            return int(v)
        except ValueError:
            print("  please enter a number")


# --------------------------------------------------------------------------
# provider construction
# --------------------------------------------------------------------------

def build_provider(args) -> tuple[str, dict]:
    iid = args.id or ask("Provider id (used in selectors like my-llm/model-a)")
    if " " in iid or "/" in iid:
        sys.exit(f"error: provider id '{iid}' must not contain spaces or '/'")

    api = args.api
    base_url = args.base_url
    key_literal: str | None = None
    env_var: str | None = None
    keyless = args.keyless
    auth_header = args.auth_header

    if not base_url:
        print("\nOpenAI-compatible servers usually end in /v1 (omp sends requests to it directly).")
        base_url = ask("Base URL", "http://127.0.0.1:1234/v1")
    if not api:
        if not sys.stdin.isatty():
            api = "openai-completions"
        else:
            print("\nWire API?")
            print("  1) openai-completions  (most local servers: vLLM, llama.cpp, LM Studio, oMLX)")
            print("  2) openai-responses    (OpenAI Responses-compatible servers)")
            print("  3) anthropic-messages  (Anthropic /v1/messages proxies)")
            api = {"1": "openai-completions", "2": "openai-responses",
                   "3": "anthropic-messages"}.get(
                ask("Choice", "1"), "openai-completions")
            if api not in VALID_APIS:
                api = "openai-completions"

    if not (keyless or args.auth_env or args.api_key is not None):
        if sys.stdin.isatty():
            print("\nAuthentication?")
            print("  1) keyless (local/unauthenticated server)")
            print("  2) API key from an environment variable (recommended: portable across machines)")
            print("  3) API key literal (stored in models.yml — travels with the file)")
            choice = ask("Choice", "1")
            if choice == "2":
                env_var = ask("Environment variable name", "MY_PROVIDER_API_KEY")
            elif choice == "3":
                key_literal = getpass.getpass("API key (input hidden): ") or ask("API key", "")
            else:
                keyless = True
        else:
            keyless = True
    if args.auth_env:
        env_var = args.auth_env
    if args.api_key is not None and not env_var and not keyless:
        key_literal = args.api_key

    # The value omp stores: env-var NAME beats literal (omp resolves it as
    # env-var-name-or-literal). `none` keeps prior configs working.
    api_key_value = env_var or key_literal

    provider: dict = {"baseUrl": base_url.rstrip("/"), "api": api}
    if keyless:
        provider["auth"] = "none"
    elif api_key_value:
        provider["apiKey"] = api_key_value
    if auth_header:
        provider["authHeader"] = True
    if args.no_developer_role or (args.developer_role is False):
        provider.setdefault("compat", {})["supportsDeveloperRole"] = False
    elif args.developer_role is True:
        provider.setdefault("compat", {})["supportsDeveloperRole"] = True

    discovered: list[dict] = []
    err = None
    show_key = api_key_value if not keyless else "none"
    if args.discover or (args.model is None and sys.stdin.isatty()):
        print(f"\nProbing {provider['baseUrl']} for models …")
        discovered, norm, err = probe_models(provider["baseUrl"], api, show_key, auth_header)
        if norm and norm != provider["baseUrl"]:
            print(f"  (models endpoint is {norm}; keeping baseUrl as configured)")
        if err:
            print(f"  discovery failed: {err}")
            if not (args.model or []):
                if sys.stdin.isatty() and ask_bool("Continue anyway and add models manually?", True):
                    pass
                else:
                    sys.exit("error: no models discovered and none provided")
        else:
            print(f"  found {len(discovered)} model(s)")

    models: list[dict] = []
    explicit_ids = args.model or []
    if explicit_ids:
        by_id = {m["id"]: m for m in discovered}
        for mid in explicit_ids:
            base = dict(by_id.get(mid, {"id": mid, "name": mid,
                                        "input": ["text"]}))
            if by_id.get(mid) is None:
                # no discovery for this id: don't bake in a silent default
                base["reasoning"] = args.reasoning
                levels = [lv for group in (args.thinking or []) for lv in group]
                if levels:
                    base["thinking"] = {"mode": "effort", "efforts": levels}
            if args.context:
                base["contextWindow"] = args.context
            if args.max_tokens:
                base["maxTokens"] = args.max_tokens
            if args.reasoning:
                base["reasoning"] = True
            if args.vision:
                base["input"] = ["text", "image"]
            models.append(clean_model(base))
    elif discovered and sys.stdin.isatty():
        print("\nDiscovered models:")
        for i, m in enumerate(discovered, 1):
            flags = []
            if m.get("reasoning"):
                flags.append("reasoning")
            if "image" in m.get("input", []):
                flags.append("vision")
            ctx = m.get("contextWindow")
            print(f"  {i:3}) {m['id']}" + (f"  ({', '.join(flags)}, ctx {ctx})" if flags or ctx else ""))
        sel = ask("Include which? (numbers/comma, blank=all, 0=none)", "all")
        chosen = discovered
        if sel.strip() in ("0", "none"):
            chosen = []
        elif sel not in ("", "all", "ALL"):
            idx = [int(s) for s in sel.replace(" ", "").split(",") if s]
            chosen = [discovered[i - 1] for i in idx if 1 <= i <= len(discovered)]
        if ask_bool("Edit context window / output cap / capabilities for each?", False):
            for m in chosen:
                m["contextWindow"] = ask_int(f"  contextWindow for {m['id']}",
                                             m.get("contextWindow", 128000))
                m["maxTokens"] = ask_int(f"  maxTokens for {m['id']}",
                                         m.get("maxTokens", 8192))
                prompt_thinking(m)
                m["input"] = ["text", "image"] if ask_bool(
                    f"  {m['id']} accepts images?", "image" in m.get("input", [])) else ["text"]
        models = [clean_model(m) for m in chosen]
    elif not explicit_ids:
        sys.exit("error: non-interactive add needs --model (and optionally --discover to fill limits)")

    if models:
        provider["models"] = models
    return iid, provider


def clean_model(m: dict) -> dict:
    out = {"id": m["id"]}
    if m.get("name") and m["name"] != m["id"]:
        out["name"] = m["name"]
    out["reasoning"] = bool(m.get("reasoning"))
    if m.get("thinking"):
        out["thinking"] = m["thinking"]
    out["input"] = m.get("input") or ["text"]
    if m.get("contextWindow"):
        out["contextWindow"] = int(m["contextWindow"])
    if m.get("maxTokens"):
        out["maxTokens"] = int(m["maxTokens"])
    if m.get("compat"):
        out["compat"] = m["compat"]
    out["cost"] = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    return out


EFFORT_LADDER = ["minimal", "low", "medium", "high", "xhigh", "max"]


def prompt_thinking(m: dict) -> None:
    """Ask for the reasoning surface when it wasn't advertised, so a manual
    entry doesn't silently flatten to `reasoning: false`."""
    m["reasoning"] = ask_bool(f"  {m['id']} reasons/thinks?", m.get("reasoning", False))
    if not m["reasoning"]:
        m.pop("thinking", None)
        return
    preset = (m.get("thinking") or {}).get("efforts")
    if preset:
        sel = ask(f"  thinking levels for {m['id']}", ",".join(preset))
    else:
        sel = ask(f"  thinking levels for {m['id']}"
                  f" (comma from {'/'.join(EFFORT_LADDER)}, blank=single toggle)",
                  "")
    if not sel.strip():
        m.pop("thinking", None)
        return
    efforts = [e.strip() for e in sel.replace(" ", "").split(",")
               if e.strip() in EFFORT_LADDER]
    m["thinking"] = {"mode": "effort", "efforts": efforts}
    fmt = infer_thinking_format(m["id"])
    if fmt and ask_bool(f"  {m['id']} carries thinking in the chat template"
                        " (qwen-style reasoning_content)?", True):
        m.setdefault("compat", {}).update(fmt)
    if ask_bool(f"  {m['id']} accepts a reasoning_effort request field?",
                bool(m.get("compat", {}).get("supportsReasoningEffort"))):
        m.setdefault("compat", {})["supportsReasoningEffort"] = True


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_status(_args) -> int:
    b = omp_binary()
    if b:
        print(f"omp:           {b}  ({omp_version(b) or 'version unknown'})")
    else:
        print(f"omp:           NOT FOUND — install with:\n                 {INSTALL_CMD}")
    path = models_yml_path()
    cfg = load_config(path)
    provs = (cfg.get("providers") or {}) if isinstance(cfg.get("providers"), dict) else {}
    print(f"agent dir:     {agent_dir()}")
    print(f"models.yml:    {path}  ({'exists' if path.exists() else 'not created yet'})")
    print(f"providers:     {', '.join(provs) if provs else '(none defined)'}")
    print("built-in providers (anthropic, openai, …) ship with omp; models.yml only\n"
          "declares custom ones and overrides.")
    return 0


def cmd_list(_args) -> int:
    path = models_yml_path()
    cfg = load_config(path)
    provs = cfg.get("providers") or {}
    if not provs:
        print(f"{path}: no custom providers defined")
        return 0
    print(f"# {path}")
    for iid, p in provs.items():
        p = p or {}
        n = len(p.get("models") or [])
        auth = "keyless" if p.get("auth") == "none" else (
            f"env:{p['apiKey']}" if p.get("apiKey") and str(p["apiKey"]).isidentifier()
            else ("literal-key" if p.get("apiKey") else "unset"))
        print(f"  {iid:20} {p.get('api', '?'):20} {p.get('baseUrl', '?')}  [{auth}, {n} model(s)]")
    return 0


def cmd_add(args) -> int:
    path = models_yml_path(create_parent=True)
    cfg = load_config(path)
    provs = cfg.setdefault("providers", {})
    iid, provider = build_provider(args)
    if iid in provs and not args.overwrite:
        if sys.stdin.isatty():
            if not ask_bool(f"provider '{iid}' already exists — overwrite?", False):
                sys.exit("aborted")
        else:
            sys.exit(f"error: provider '{iid}' exists (use --overwrite)")
    provs[iid] = provider
    backup = save_config(path, cfg)
    if backup:
        print(f"backup: {backup}")
    print(f"wrote provider '{iid}' to {path}")
    print(f"  use it with:  omp --model {iid}/{(provider.get('models') or [{}])[0].get('id', '<model>')}")
    return 0


def cmd_remove(args) -> int:
    path = models_yml_path()
    cfg = load_config(path)
    provs = cfg.get("providers") or {}
    if args.id not in provs:
        sys.exit(f"error: '{args.id}' not in {path}")
    del provs[args.id]
    backup = save_config(path, cfg)
    print(f"removed '{args.id}'  (backup: {backup})")
    return 0


def cmd_doctor(args) -> int:
    path = models_yml_path()
    if not path.exists():
        print(f"{path}: missing — nothing to validate")
        return 0
    try:
        cfg = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        print(f"YAML error in {path}: {e}")
        return 1
    provs = (cfg or {}).get("providers") or {}
    if not isinstance(provs, dict):
        print("error: root `providers:` must be a mapping")
        return 1
    bad = 0
    for iid, p in provs.items():
        p = p or {}
        problems = []
        if not p.get("baseUrl"):
            problems.append("missing baseUrl")
        api = p.get("api")
        models = p.get("models") or []
        if models and api not in VALID_APIS:
            problems.append(f"provider-level api must be one of {VALID_APIS} (got {api!r})")
        if p.get("auth") not in (None, "none", "oauth", "apiKey"):
            problems.append(f"invalid auth: {p.get('auth')!r}")
        for m in models:
            if not m.get("id"):
                problems.append("model entry without id")
            cw, mt = m.get("contextWindow"), m.get("maxTokens")
            if cw is not None and cw <= 0:
                problems.append(f"{m.get('id')}: contextWindow must be positive")
            if mt is not None and cw is not None and mt > cw:
                problems.append(f"{m.get('id')}: maxTokens exceeds contextWindow")
        status = "OK " if not problems else "BAD"
        print(f"  [{status}] {iid}")
        for pr in problems:
            print(f"         - {pr}")
            bad += 1
        if args.probe:
            key = p.get("apiKey")
            key_val = os.environ.get(key, key) if isinstance(key, str) else None
            models_found, norm, err = probe_models(
                p.get("baseUrl", ""), api or "openai-completions",
                key_val, bool(p.get("authHeader")))
            if err:
                print(f"         probe: unreachable ({err})")
            else:
                have = {m.get("id") for m in models}
                disc = {m["id"] for m in models_found}
                print(f"         probe: {norm} — {len(models_found)} model(s)"
                      + (f", missing from config: {', '.join(sorted(disc - have))}"
                         if disc - have else ""))
    if bad:
        print(f"\n{bad} problem(s) found; omp skips the file on schema errors — fix before launching")
        return 1
    print("\nmodels.yml is structurally valid")
    b = omp_binary()
    if b and not args.no_cross_check:
        r = subprocess.run([str(b), "models"], capture_output=True, text=True, timeout=60)
        ok = r.returncode == 0
        print(f"omp models cross-check: {'OK' if ok else 'FAILED'}")
        if not ok:
            print((r.stderr or r.stdout)[:2000])
            return 1
    return 0


def cmd_export(args) -> int:
    src = models_yml_path()
    if not src.exists():
        sys.exit(f"error: {src} does not exist")
    dst = Path(args.path).expanduser()
    shutil.copy2(src, dst)
    print(f"exported {src} -> {dst}")
    print("On the target machine: run `omp_helper.py import <that file>`\n"
          "(it merges providers, backs up any existing file). OAuth logins and\n"
          "disabledProviders live elsewhere (agent.db / config.yml).")
    return 0


def cmd_import(args) -> int:
    src = Path(args.path).expanduser()
    if not src.exists():
        sys.exit(f"error: {src} not found")
    incoming = yaml.safe_load(src.read_text()) or {}
    provs_in = incoming.get("providers")
    if not isinstance(provs_in, dict):
        sys.exit(f"error: {src} has no `providers:` mapping")
    dst = models_yml_path(create_parent=True)
    cfg = load_config(dst)
    provs = cfg.setdefault("providers", {})
    added = updated = 0
    for iid, p in provs_in.items():
        if iid in provs:
            updated += 1
        else:
            added += 1
        provs[iid] = p
    backup = save_config(dst, cfg)
    print(f"imported into {dst}: {added} new, {updated} overwritten (backup: {backup})")
    return 0


def cmd_install_omp(args) -> int:
    if omp_binary() and not args.force:
        print(f"omp already installed: {omp_binary()} ({omp_version(omp_binary())})")
        return 0
    print(f"Running: {INSTALL_CMD}")
    r = subprocess.run(["sh", "-c", INSTALL_CMD])
    if r.returncode != 0:
        return r.returncode
    b = omp_binary()
    print(f"installed: {b} ({omp_version(b) if b else 'restart your shell / check PATH'})")
    return 0


# --------------------------------------------------------------------------
# interactive menu
# --------------------------------------------------------------------------

def menu() -> int:
    print("omp-providers — custom provider configurator\n")
    cmd_status(None)
    print("\nWhat do you want to do?")
    print("  1) add a custom OpenAI-compatible provider (guided)")
    print("  2) list providers")
    print("  3) validate + probe endpoints (doctor)")
    print("  4) export models.yml for another machine")
    print("  5) import a models.yml")
    print("  6) remove a provider")
    c = ask("Choice", "1")
    ns = argparse.Namespace(id=None, base_url=None, api=None, auth_env=None,
                            api_key=None, keyless=False, auth_header=False,
                            discover=True, model=None, thinking=None,
                            context=None,
                            max_tokens=None, reasoning=False, vision=False,
                            developer_role=None, no_developer_role=False,
                            overwrite=False)
    if c == "1":
        return cmd_add(ns)
    if c == "2":
        return cmd_list(None)
    if c == "3":
        ns.probe = True
        ns.no_cross_check = False
        return cmd_doctor(ns)
    if c == "4":
        ns.path = ask("Export to path", "~/omp-providers.yml")
        return cmd_export(ns)
    if c == "5":
        ns.path = ask("Import from path", "~/omp-providers.yml")
        return cmd_import(ns)
    if c == "6":
        ns.id = ask("Provider id to remove")
        return cmd_remove(ns)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="omp-providers", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("status", help="detect omp install + models.yml")
    sub.add_parser("list", help="list custom providers")

    p_add = sub.add_parser("add", help="add a custom provider")
    p_add.add_argument("--id")
    p_add.add_argument("--base-url")
    p_add.add_argument("--api", choices=VALID_APIS)
    p_add.add_argument("--auth-env", help="environment variable holding the API key")
    p_add.add_argument("--api-key", help="literal API key stored in models.yml")
    p_add.add_argument("--keyless", action="store_true", help="no credentials (auth: none)")
    p_add.add_argument("--auth-header", action="store_true",
                       help="send Authorization: Bearer <key> (gateways that need it)")
    p_add.add_argument("--discover", action="store_true", help="probe the endpoint's /models")
    p_add.add_argument("--model", action="append", help="model id to include (repeatable)")
    p_add.add_argument("--thinking", action="append", nargs="+", metavar="LEVEL",
                       help="thinking effort levels, e.g. --thinking low medium xhigh"
                            " (flat reasoning toggle if omitted)")
    p_add.add_argument("--context", type=int)
    p_add.add_argument("--max-tokens", type=int)
    p_add.add_argument("--reasoning", action="store_true")
    p_add.add_argument("--vision", action="store_true")
    p_add.add_argument("--developer-role", dest="developer_role", action="store_true",
                       default=None)
    p_add.add_argument("--no-developer-role", dest="no_developer_role", action="store_true")
    p_add.add_argument("--overwrite", action="store_true")

    p_rm = sub.add_parser("remove", help="remove a provider")
    p_rm.add_argument("id")

    p_doc = sub.add_parser("doctor", help="validate models.yml and probe endpoints")
    p_doc.add_argument("--probe", action="store_true", help="also GET /models each provider")
    p_doc.add_argument("--no-cross-check", action="store_true",
                       help="skip `omp models` validation")

    p_ex = sub.add_parser("export", help="copy models.yml to a transfer path")
    p_ex.add_argument("path")
    p_im = sub.add_parser("import", help="merge a models.yml into this install")
    p_im.add_argument("path")

    p_in = sub.add_parser("install-omp", help="run the official omp installer")
    p_in.add_argument("--force", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "add":
        return cmd_add(args)
    if args.cmd == "remove":
        return cmd_remove(args)
    if args.cmd == "doctor":
        return cmd_doctor(args)
    if args.cmd == "export":
        return cmd_export(args)
    if args.cmd == "import":
        return cmd_import(args)
    if args.cmd == "install-omp":
        return cmd_install_omp(args)
    if not sys.stdin.isatty():
        ap.print_help()
        return 2
    return menu()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\naborted")
        sys.exit(130)
