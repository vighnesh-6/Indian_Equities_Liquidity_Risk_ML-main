"""Load credentials exclusively from the checkout's root .env.local."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
APPROVED_TARGET = (
    "super-flower-30495255", "br-winter-thunder-azx88rt3", "ep-misty-shadow-azik30ky",
    "ep-misty-shadow-azik30ky-pooler.c-3.ap-southeast-1.aws.neon.tech", "neondb",
)


class ConfigurationError(ValueError):
    """Configuration is absent, ambiguous or outside the approved target."""


@dataclass(frozen=True)
class Settings:
    project_name: str
    project_id: str
    branch_name: str
    branch_id: str
    endpoint_id: str
    hostname: str
    database: str
    connect_timeout_seconds: int
    statement_timeout_ms: int
    lock_timeout_ms: int
    username: str
    password: str = field(repr=False)


def load_settings(project_root: Path = ROOT) -> Settings:
    try:
        config = json.loads((Path(__file__).with_name("config.json")).read_text(encoding="utf-8"))
        lines = (project_root / ".env.local").read_text(encoding="utf-8-sig").splitlines()
    except (OSError, ValueError):
        raise ConfigurationError("Production configuration could not be loaded") from None
    if tuple(config.get(key) for key in ["project_id", "branch_id", "endpoint_id", "hostname", "database"]) != APPROVED_TARGET:
        raise ConfigurationError("Configuration differs from the verified Liquidity Risk target")
    if any(type(config.get(key)) is not int or config[key] <= 0 for key in ["connect_timeout_seconds", "statement_timeout_ms", "lock_timeout_ms"]):
        raise ConfigurationError("Production timeouts must be positive integers")
    values = []
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if separator and key.strip() == "LIQUIDITY_NEON_DATABASE_URL":
            value = value.strip()
            if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
                value = value[1:-1]
            values.append(value)
    if len(values) != 1 or not values[0]:
        raise ConfigurationError("Exactly one nonempty Liquidity Risk database value is required")
    try:
        uri = urlsplit(values[0])
        parameters = parse_qsl(uri.query, keep_blank_values=True)
        valid = (
            uri.scheme in {"postgres", "postgresql"}
            and uri.hostname == config["hostname"]
            and uri.port in {None, 5432}
            and unquote(uri.path.lstrip("/")) == config["database"]
            and uri.username and uri.password and not uri.fragment
            and all(key in {"sslmode", "channel_binding"} for key, _ in parameters)
            and len({key for key, _ in parameters}) == len(parameters)
        )
        if not valid:
            raise ValueError
        return Settings(**config, username=unquote(uri.username), password=unquote(uri.password))
    except (ValueError, TypeError, KeyError):
        raise ConfigurationError("Local database URL does not match the approved Liquidity Risk target") from None
