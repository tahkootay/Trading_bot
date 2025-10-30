#!/usr/bin/env python3
"""
Data Collector Module CLI Entry Point

This module runs the data collection system from the command line.
"""

import asyncio
from .main import main

if __name__ == "__main__":
    asyncio.run(main())