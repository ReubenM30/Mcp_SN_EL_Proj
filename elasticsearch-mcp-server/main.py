# main.py
import sys
import os

# ✅ Add 'src' to the Python path so Python can find your package
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# ✅ Import and run your server main
from elasticsearch_mcp_server.server import main

if __name__ == "__main__":
    main()
