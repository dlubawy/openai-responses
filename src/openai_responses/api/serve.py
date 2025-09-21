import argparse
import logging
import os

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Responses API server")
    parser.add_argument(
        "--checkpoint",
        metavar="FILE",
        type=str,
        help="Path to the SafeTensors checkpoint",
        default="~/model",
        required=False,
    )
    parser.add_argument(
        "--port",
        metavar="PORT",
        type=int,
        default=8000,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--workers",
        metavar="WORKERS",
        type=int,
        default=1,
        help="Workers to process",
    )
    parser.add_argument(
        "--inference-backend",
        metavar="BACKEND",
        type=str,
        help="Inference backend to use",
        # default to metal on macOS, triton on other platforms
        default="metal" if __import__("platform").system() == "Darwin" else "triton",
    )
    parser.add_argument(
        "--verbosity",
        metavar="VERBOSITY",
        type=str,
        help="Verbosity level to use",
        default="WARN",
        choices=list(logging.getLevelNamesMapping().keys()),
    )
    parser.add_argument(
        "--log-level",
        metavar="LOG_LEVEL",
        type=str,
        help="Log level to use",
        default="DEBUG",
        choices=list(logging.getLevelNamesMapping().keys()),
    )
    args = parser.parse_args()

    os.environ.update(
        {
            "OPENAI_RESPONSES_INFERENCE_BACKEND": args.inference_backend,
            "OPENAI_RESPONSES_CHECKPOINT": args.checkpoint,
            "OPENAI_RESPONSES_LOG_LEVEL": str(logging.getLevelName(args.log_level)),
            "OPENAI_RESPONSES_VERBOSITY": str(logging.getLevelName(args.verbosity)),
        }
    )
    uvicorn.run(
        "openai_responses.api.api_server:create_api_server",
        port=args.port,
        workers=args.workers,
        factory=True,
    )


if __name__ == "__main__":
    main()
