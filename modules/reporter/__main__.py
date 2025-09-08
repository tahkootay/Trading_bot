#!/usr/bin/env python3
"""
Entry point for running the reporter module as a package.
"""

import asyncio
from .main import main

if __name__ == "__main__":
    asyncio.run(main())