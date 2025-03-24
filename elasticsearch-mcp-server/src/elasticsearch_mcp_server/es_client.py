import logging
import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
import warnings

class ElasticsearchClient:
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.es_client = self._create_elasticsearch_client()

    def _get_es_config(self):
        """Get Elasticsearch configuration from environment variables."""
        # Load environment variables from .env file
        load_dotenv()
        config = {
            "host": os.getenv("ELASTIC_HOST"),
            "username": os.getenv("ELASTIC_USERNAME"),
            "password": os.getenv("ELASTIC_PASSWORD")
        }
        
        if not all([config["username"], config["password"]]):
            self.logger.error("Missing required Elasticsearch configuration. Please check environment variables:")
            self.logger.error("ELASTIC_USERNAME and ELASTIC_PASSWORD are required")
            raise ValueError("Missing required Elasticsearch configuration")
        
        return config

    def _create_elasticsearch_client(self) -> Elasticsearch:
        """Create and return an Elasticsearch client using configuration from environment."""
        config = self._get_es_config()

        # Disable SSL warnings
        warnings.filterwarnings("ignore", message=".*TLS with verify_certs=False is insecure.*",)

        return Elasticsearch(
            config["host"],
            basic_auth=(config["username"], config["password"]),
            verify_certs=False
        )
