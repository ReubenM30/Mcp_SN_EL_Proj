import os
import requests
from mcp.server.fastmcp import FastMCP

# Initialize MCP
mcp = FastMCP("ServiceNow")

# Configuration for ServiceNow API (use environment variables for security)
INSTANCE_URL = "https://dev268377.service-now.com/"
USER = "admin"
PASSWORD = "A72D^ksF$oFc"

# Headers for the API requests
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}

@mcp.tool()
def create_chg(short_description: str, description: str, priority: str) -> dict:
    """Create a change request in ServiceNow"""
    url = f"{INSTANCE_URL}/api/now/table/change_request"
    data = {
        "short_description": short_description,
        "description": description,
        "priority": priority
    }
    response = requests.post(url, json=data, auth=(USER, PASSWORD), headers=HEADERS)

    if response.status_code != 201:
        return {"error": f"Failed to create change request: {response.status_code}", "details": response.text}

    return response.json()

@mcp.tool()
def get_change(change_id: str) -> dict:
    """Retrieve a change request from ServiceNow"""
    url = f"{INSTANCE_URL}/api/now/table/change_request/{change_id}"
    response = requests.get(url, auth=(USER, PASSWORD), headers=HEADERS)

    if response.status_code != 200:
        return {"error": f"Failed to retrieve change request: {response.status_code}", "details": response.text}

    return response.json()

@mcp.tool()
def create_incident(short_description: str, description: str, urgency: str) -> dict:
    """Create an incident in ServiceNow"""
    url = f"{INSTANCE_URL}/api/now/table/incident"
    data = {
        "short_description": short_description,
        "description": description,
        "urgency": urgency
    }
    response = requests.post(url, json=data, auth=(USER, PASSWORD), headers=HEADERS)

    # Handle potential errors
    if response.status_code != 201:
        return {"error": f"Failed to create incident: {response.status_code}", "details": response.text}

    return response.json()

@mcp.tool()
def get_incident(incident_id: str) -> dict:
    """Retrieve an incident from ServiceNow"""
    url = f"{INSTANCE_URL}/api/now/table/incident/{incident_id}"
    response = requests.get(url, auth=(USER, PASSWORD), headers=HEADERS)
    

    # Handle potential errors
    if response.status_code != 200:
        return {"error": f"Failed to retrieve incident: {response.status_code}", "details": response.text}

    return response.json()

if __name__ == "__main__":
    mcp.run(transport="stdio")
