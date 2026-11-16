"""Scheduled and manual daily entry point, sharing the catch-up collector."""

from .catchup import run_cli


if __name__ == "__main__":
    run_cli("daily")
