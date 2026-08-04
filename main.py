#!/usr/bin/env python3
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.bootstrap import bootstrap
from interaction.cli import InteractiveCLI

WORKSPACE = os.environ.get("AGENT_WORKSPACE") or os.getcwd()

async def main():
    harness = await bootstrap(
        workspace=WORKSPACE,
        skills_dir="./skills",
        store_path="./agent_state.db",
        max_steps=200,
    )

    cli = InteractiveCLI(harness=harness, workspace=WORKSPACE)
    try:
        await cli.run()
    finally:
        await harness.shutdown()


if __name__ == "__main__":
    asyncio.run(main())