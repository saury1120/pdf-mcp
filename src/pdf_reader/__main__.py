"""
PDF Reader Server 启动脚本
"""
import asyncio
from .server import main

if __name__ == "__main__":
    asyncio.run(main())
