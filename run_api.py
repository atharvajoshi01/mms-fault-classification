#!/usr/bin/env python
"""
Run the FastAPI prediction server.

Usage:
    python run_api.py              # Run on default port 8000
    python run_api.py --port 8080  # Run on custom port
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run MMS Fault Classification API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         MMS Fault Classification - FastAPI Server            ║
╠══════════════════════════════════════════════════════════════╣
║  API Docs:  http://{args.host}:{args.port}/docs                        ║
║  Health:    http://{args.host}:{args.port}/health                      ║
║  Predict:   POST http://{args.host}:{args.port}/predict                ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
