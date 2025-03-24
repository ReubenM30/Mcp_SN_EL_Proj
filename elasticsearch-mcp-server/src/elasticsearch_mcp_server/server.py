#!/usr/bin/env python3
import logging
from fastmcp import FastMCP
from .tools.index import IndexTools
from .tools.document import DocumentTools
from .tools.cluster import ClusterTools

class ElasticsearchMCPServer:
    def __init__(self):
        self.name = "elasticsearch_mcp_server"
        self.mcp = FastMCP(self.name)

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.name)
        self.logger.info("✅ MCP Server initialized.")  # Debug log

        # Initialize tools
        self._register_tools()

    def _register_tools(self):
        """Register all MCP tools."""
        self.logger.info("🔧 Registering tools...")  # Debug log
        index_tools = IndexTools(self.logger)
        document_tools = DocumentTools(self.logger)
        cluster_tools = ClusterTools(self.logger)

        index_tools.register_tools(self.mcp)
        document_tools.register_tools(self.mcp)
        cluster_tools.register_tools(self.mcp)
        self.logger.info("✅ All tools registered.")

    def run(self):
        """Run the MCP server."""
        self.logger.info("🚀 Running the MCP server...")
        self.mcp.run()

def main():
    server = ElasticsearchMCPServer()
    server.run()
