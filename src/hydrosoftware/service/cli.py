"""Command-line launcher for the local HydroSoftware API."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the HydroSoftware local API.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Project workspace containing portable data paths.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reload", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit(
            "HydroSoftware only permits loopback hosts. Use 127.0.0.1."
        )
    if args.workers < 1:
        raise SystemExit("--workers must be at least one.")
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "The local-service dependencies are not installed. "
            "Install the project with the 'service' optional dependency."
        ) from exc

    workspace = args.workspace.resolve()
    if args.reload:
        raise SystemExit(
            "Automatic reload is intentionally disabled for the configuration-driven "
            "launcher. Restart the service after changing backend code."
        )

    from hydrosoftware.service.api import create_app

    uvicorn.run(
        create_app(workspace_root=workspace, max_workers=args.workers),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
