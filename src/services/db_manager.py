
import pymongo
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv


class DBManager:
    """
    MongoDB manager for local and remote (Atlas) connections.
    """

    def __init__(self):
        load_dotenv()
        self.host = os.environ.get("DB_HOST", "localhost")
        self.port = int(os.environ.get("DB_PORT", 27017))
        self.user = os.environ.get("DB_USER", None)
        self.password = os.environ.get("DB_PASSWORD", None)
        self.db_name = os.environ.get("DB_NAME", "critical-path-data")
        self.cluster_name = os.environ.get("DB_CLUSTER_NAME", None)

        self.client = pymongo.MongoClient(f"mongodb://{self.host}:{self.port}/")
        self.db = self.client[self.db_name]

        if self.user and self.password and self.cluster_name:
            self.server_client = pymongo.MongoClient(
                f"mongodb+srv://{self.user}:{self.password}@{self.cluster_name}/?retryWrites=true&w=majority"
            )
        else:
            self.server_client = None

    def get_previous_parameters(
        self,
        collection: str,
        condition: Optional[Dict[str, Any]] = None,
        server: bool = True
    ) -> List[Any]:
        """
        Fetches previous parameters from the specified collection.
        """
        if condition is None:
            condition = {}

        if server and self.server_client:
            items = list(self.server_client[self.db_name][collection].find(condition))
        else:
            items = list(self.db[collection].find(condition))

        if collection != 'reports-parsec_freqmine_optimized':
            return [item['parameters'] for item in items]
        return list(items)

    def insert_to_db(self, collection: str, document: Dict[str, Any]) -> None:
        """
        Inserts a document into the specified collection in the local DB.
        """
        self.db[collection].insert_one(document)
