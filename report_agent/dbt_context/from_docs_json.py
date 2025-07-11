import json
import requests
from pathlib import Path
from typing import Dict, Any

def load_manifest(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch and parse the dbt `manifest.json` from your docs site.
    """
    base = cfg["dbt_docs"]["base_url"].rstrip("/")
    url  = f"{base}/manifest.json"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def load_catalog(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch and parse the dbt `catalog.json` from your docs site.
    """
    base = cfg["dbt_docs"]["base_url"].rstrip("/")
    url  = f"{base}/catalog.json"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def get_model_node(manifest: dict, model_name: str) -> dict:
    """
    Look up the single model node in the manifest by reading
    manifest['metadata']['project_name'] instead of hard-coding it.
    """
    project = manifest["metadata"]["project_name"]
    node_key = f"model.{project}.{model_name}"
    try:
        return manifest["nodes"][node_key]
    except KeyError:
        raise KeyError(
            f"Model '{model_name}' not found under project '{project}'. "
            f"Available keys: {list(manifest['nodes'].keys())[:5]}…"
        )

def get_column_metadata(node: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    From a manifest node, return a map:
      column_name -> { description, data_type, tests, tags… }
    """
    cols = node.get("columns", {})
    return {
        col_name: {
            "description": info.get("description"),      
            "data_type":   info.get("data_type"),       
            "constraints": info.get("constraints", []),  
            "tags":        info.get("tags", []),         
            "meta":        info.get("meta", {}),        
        }
        for col_name, info in cols.items()
    }