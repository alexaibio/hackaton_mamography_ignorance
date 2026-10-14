import argparse
from pathlib import Path

from mammography.config import get_config, get_settings
from mammography.fhir.client import fetch_resources
from mammography.fhir.flatten import flatten_resources
from mammography.logger import setup_logger

RESOURCES = ("Patient", "Communication", "Appointment")


def cmd_fetch(args: argparse.Namespace) -> None:
    log = setup_logger("fetch")
    settings = get_settings()
    raw_dir = settings.data_folder / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for resource_type in RESOURCES:
        out_path = raw_dir / f"{resource_type}.ndjson"
        if out_path.exists() and not args.force:
            log.info("{} already exists, skipping", out_path.name)
            continue
        log.info("fetching {} from {}", resource_type, settings.fhir_base_url)
        written = fetch_resources(resource_type, out_path, limit=args.limit)
        log.info("wrote {} {} resources to {}", written, resource_type, out_path)


def cmd_flatten(args: argparse.Namespace) -> None:
    log = setup_logger("flatten")
    settings = get_settings()
    log.info("flattening {} -> {}", settings.raw_dir, settings.interim_dir)
    flatten_resources()


def main() -> None:
    parser = argparse.ArgumentParser(prog="mammo")
    sub = parser.add_subparsers(dest="command", required=True)

    p_fetch = sub.add_parser("fetch", help="download FHIR resources to data/raw")
    p_fetch.add_argument("--limit", type=int, default=None, help="stop after N resources")
    p_fetch.add_argument("--force", action="store_true", help="re-download existing files")
    p_fetch.set_defaults(func=cmd_fetch)

    p_flatten = sub.add_parser("flatten", help="data/raw ndjson -> data/interim csv")
    p_flatten.set_defaults(func=cmd_flatten)

    args = parser.parse_args()
    args.func(args)


