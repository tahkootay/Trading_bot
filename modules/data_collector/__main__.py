#!/usr/bin/env python3
"""
Entry point for data collector module when run with python -m
"""

import asyncio
from .main import main

if __name__ == "__main__":
    asyncio.run(main())