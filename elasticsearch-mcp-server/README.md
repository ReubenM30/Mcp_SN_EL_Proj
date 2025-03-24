# Elasticsearch MCP Server

[![smithery badge](https://smithery.ai/badge/elasticsearch-mcp-server)](https://smithery.ai/server/elasticsearch-mcp-server)

## Overview

A Model Context Protocol (MCP) server implementation that provides Elasticsearch interaction. This server enables searching documents, analyzing indices, and managing cluster through a set of tools.

<a href="https://glama.ai/mcp/servers/b3po3delex"><img width="380" height="200" src="https://glama.ai/mcp/servers/b3po3delex/badge" alt="Elasticsearch MCP Server" /></a>

## Demo

https://github.com/user-attachments/assets/f7409e31-fac4-4321-9c94-b0ff2ea7ff15

## Features

### Index Operations

- `list_indices`: List all indices in the Elasticsearch cluster.
- `get_mapping`: Retrieve the mapping configuration for a specific index.
- `get_settings`: Get the settings configuration for a specific index.

### Document Operations

- `search_documents`: Search documents in an index using Elasticsearch Query DSL.

### Cluster Operations

- `get_cluster_health`: Get health status of the cluster.
- `get_cluster_stats`: Get statistical information about the cluster.


## Start Elasticsearch Cluster

Start the Elasticsearch cluster using Docker Compose:

```bash
docker-compose up -d
```

This will start a 3-node Elasticsearch cluster and Kibana. Default Elasticsearch username `elastic`, password `test123`.

You can access Kibana from http://localhost:5601.

## Usage with Claude Desktop

### Option 1: Installing via Smithery

To install Elasticsearch Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/elasticsearch-mcp-server):

```bash
npx -y @smithery/cli install elasticsearch-mcp-server --client claude
```

### Option 2: Using uvx

Using `uvx` will automatically install the package from PyPI, no need to clone the repository locally. Add the following configuration to Claude Desktop's config file `claude_desktop_config.json`.

```json
{
  "mcpServers": {
    "elasticsearch-mcp-server": {
      "command": "uvx",
      "args": [
        "elasticsearch-mcp-server"
      ],
      "env": {
        "ELASTIC_HOST": "https://localhost:9200",
        "ELASTIC_USERNAME": "elastic",
        "ELASTIC_PASSWORD": "test123"
      }
    }
  }
}
```

### Option 3: Using uv with local development

Using `uv` requires cloning the repository locally and specifying the path to the source code. Add the following configuration to Claude Desktop's config file `claude_desktop_config.json`.

```json
{
  "mcpServers": {
    "elasticsearch": {
      "command": "uv",
      "args": [
        "--directory",
        "path/to/src/elasticsearch_mcp_server",
        "run",
        "elasticsearch-mcp-server"
      ],
      "env": {
        "ELASTIC_HOST": "https://localhost:9200",
        "ELASTIC_USERNAME": "elastic",
        "ELASTIC_PASSWORD": "test123"
      }
    }
  }
}
```

- On macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- On Windows: `%APPDATA%/Claude/claude_desktop_config.json`

Restart Claude Desktop to load the new MCP server.

Now you can interact with your Elasticsearch cluster through Claude using natural language commands like:
- "List all indices in the cluster"
- "How old is the student Bob?"
- "Show me the cluster health status"

## License

This project is licensed under the Apache License Version 2.0 - see the [LICENSE](LICENSE) file for details.
