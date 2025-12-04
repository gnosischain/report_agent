import json
import requests
from pathlib import Path
from typing import Dict, Any, List

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


def _extract_domain(tags: List[str]) -> str:
    """Extract primary domain from tags."""
    domains = ["execution", "p2p", "tokens", "probelab", "consensus", "bridges", "contracts"]
    for tag in tags:
        if tag in domains:
            return tag
    return "other"


def build_model_catalog(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a comprehensive catalog of all available models that the LLM can use
    to decide which models to query for context.
    
    Returns a dictionary mapping model names to their metadata.
    """
    try:
        manifest = load_manifest(cfg)
    except Exception:
        # If manifest can't be loaded, return empty catalog
        return {}
    
    project = manifest.get("metadata", {}).get("project_name", "cerebro")
    catalog = {}
    
    nodes = manifest.get("nodes", {})
    for node_key, node in nodes.items():
        if not node_key.startswith(f"model.{project}."):
            continue
        
        model_name = node_key.split(".", 2)[-1]
        tags = node.get("tags", [])
        
        # Include all production models (not just API ones, for flexibility)
        if "production" not in tags:
            continue
        
        # Get column info using shared utility
        col_meta = get_column_metadata(node)
        columns = list(col_meta.keys())
        
        # Extract simplified column details for catalog
        column_info = {
            col_name: {
                "data_type": info.get("data_type", ""),
                "description": info.get("description", ""),
            }
            for col_name, info in col_meta.items()
        }
        
        # Infer kind
        has_date = "date" in columns
        has_value = "value" in columns
        has_label = "label" in columns
        
        catalog[model_name] = {
            "name": model_name,
            "description": node.get("description", ""),
            "tags": tags,
            "domain": _extract_domain(tags),
            "kind": "time_series" if has_date else "snapshot",
            "columns": columns,  # columns is already a list
            "column_details": column_info,
            "has_date": has_date,
            "has_value": has_value,
            "has_label": has_label,
        }
    
    return catalog


def find_related_models(
    model_name: str,
    catalog: dict,
    max_related: int = 3
) -> list:
    """
    Find related models using simple heuristics:
    - Same domain (strongest signal)
    - Shared tags (excluding generic ones)
    - Name similarity (common words)
    
    Returns list of model names, ordered by relevance.
    """
    if model_name not in catalog:
        return []
    
    current = catalog[model_name]
    current_domain = current.get("domain")
    current_tags = set(current.get("tags", [])) - {"production", "api", "tier0", "tier1"}
    current_name_words = set(model_name.split("_"))
    
    scores = {}
    
    for name, info in catalog.items():
        if name == model_name:
            continue
        
        score = 0
        
        # Same domain (strong signal)
        if info.get("domain") == current_domain:
            score += 10
        
        # Shared tags
        other_tags = set(info.get("tags", [])) - {"production", "api", "tier0", "tier1"}
        shared_tags = current_tags & other_tags
        score += len(shared_tags) * 2
        
        # Name similarity (common meaningful words)
        other_name_words = set(name.split("_"))
        common_words = current_name_words & other_name_words
        for word in common_words:
            if len(word) > 3:  # Only count meaningful words
                score += 1
        
        if score > 0:
            scores[name] = score
    
    # Return top N by score
    sorted_related = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_related[:max_related]]


def save_catalog_to_file(catalog: Dict[str, Any], filepath: str):
    """Save catalog as JSON for the model to read."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)