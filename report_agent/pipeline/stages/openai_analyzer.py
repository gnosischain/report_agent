# report_agent/pipeline/stages/openai_analyzer.py
"""
OpenAI Code Interpreter implementation of LLMAnalyzer.

This is the primary LLM analyzer implementation, using OpenAI's
Responses API with Code Interpreter to analyze metrics.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

import httpx
from openai import OpenAI

from report_agent.pipeline.models import AnalysisContext, RawAnalysis
from report_agent.pipeline.stages.llm_analyzer import LLMAnalyzer
from report_agent.pipeline.exceptions import AnalysisError

log = logging.getLogger(__name__)


class OpenAIAnalyzer(LLMAnalyzer):
    """
    OpenAI Code Interpreter implementation of LLMAnalyzer.
    
    Uses the OpenAI Responses API with Code Interpreter to:
    - Upload context files (CSV, schema, meta, docs, catalog)
    - Run analysis with Python code execution
    - Parse structured output and narrative
    - Track artifacts for later download
    
    Usage:
        analyzer = OpenAIAnalyzer(api_key="...", model_name="gpt-4.1")
        result = analyzer.analyze(context)
        plots = analyzer.download_artifacts("reports/plots")
    """
    
    def __init__(self, api_key: str, model_name: str = "gpt-4.1"):
        self.api_key = api_key
        self.model_name = model_name
        
        # Disable automatic retries to save credits
        self.client = OpenAI(
            api_key=api_key,
            max_retries=0,
            http_client=httpx.Client(
                timeout=httpx.Timeout(300.0, connect=10.0),  # 5 min total, 10s connect
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            ),
        )
        
        self._last_artifacts: Optional[dict] = None
    
    def analyze(self, context: AnalysisContext) -> RawAnalysis:
        """
        Run OpenAI Code Interpreter analysis.
        
        Args:
            context: AnalysisContext from the ContextBuilder stage
            
        Returns:
            RawAnalysis containing the narrative and structured output
            
        Raises:
            AnalysisError: If the API call fails
        """
        model = context.model_name
        
        # Upload files
        file_ids = self._upload_files(context)
        
        # Build tools config
        tools = [
            {
                "type": "code_interpreter",
                "container": {
                    "type": "auto",
                    "file_ids": file_ids,
                },
            }
        ]
        
        instructions = (
            "You are a data analyst. Always use the python tool to load the attached files. "
            "Follow the task specification given in the input."
        )
        
        try:
            log.debug(f"Calling OpenAI API for model '{model}'")
            resp = self.client.responses.create(
                model=self.model_name,
                tools=tools,
                tool_choice="required",
                max_tool_calls=8,
                parallel_tool_calls=False,
                instructions=instructions,
                input=context.prompt,
                temperature=0.2,
            )
        except Exception as e:
            error_msg = str(e)
            log.error(f"API call failed for {model}: {error_msg}")
            raise AnalysisError(f"API call failed: {error_msg}", model)
        
        # Extract artifacts
        try:
            self._last_artifacts = self._extract_container_artifacts(resp)
        except Exception:
            self._last_artifacts = None
        
        # Parse response
        narrative, structured = self._parse_structured_output(resp)
        
        log.info(f"Analysis complete for model '{model}'")
        
        return RawAnalysis(
            model_name=model,
            narrative=narrative,
            structured=structured,
            artifacts=self._last_artifacts,
        )
    
    def _upload_files(self, context: AnalysisContext) -> List[str]:
        """Upload all context files to OpenAI and return file IDs."""
        file_ids: List[str] = []
        
        # Required files
        with open(context.csv_path, "rb") as f:
            file_ids.append(self.client.files.create(file=f, purpose="assistants").id)
        
        with open(context.schema_path, "rb") as f:
            file_ids.append(self.client.files.create(file=f, purpose="assistants").id)
        
        with open(context.meta_path, "rb") as f:
            file_ids.append(self.client.files.create(file=f, purpose="assistants").id)
        
        # Optional files
        if context.docs_path and context.docs_path.exists():
            with open(context.docs_path, "rb") as f:
                file_ids.append(self.client.files.create(file=f, purpose="assistants").id)
        
        if context.catalog_path and context.catalog_path.exists():
            with open(context.catalog_path, "rb") as f:
                file_ids.append(self.client.files.create(file=f, purpose="assistants").id)
        
        return file_ids
    
    def _extract_container_artifacts(self, resp) -> dict | None:
        """
        Walk the Responses API output and collect any container file citations.
        Returns a dict: {"container_ids": [...], "files": [{"container_id","file_id","filename"}]}
        """
        artifacts = {"container_ids": [], "files": []}
        seen_containers = set()

        output = getattr(resp, "output", None)
        if isinstance(output, list):
            for item in output:
                # Dict-shape
                if isinstance(item, dict) and item.get("type") == "message":
                    for content in (item.get("content") or []):
                        anns = content.get("annotations") or []
                        for a in anns:
                            if a.get("type") == "container_file_citation":
                                cid = a.get("container_id")
                                fid = a.get("file_id")
                                fname = a.get("filename")
                                if cid and cid not in seen_containers:
                                    seen_containers.add(cid)
                                if cid and fid:
                                    artifacts["files"].append({
                                        "container_id": cid,
                                        "file_id": fid,
                                        "filename": fname,
                                    })
                # Attr-shape
                elif hasattr(item, "content"):
                    for content in (item.content or []):
                        anns = getattr(content, "annotations", None) or []
                        for a in anns:
                            if getattr(a, "type", None) == "container_file_citation":
                                cid = getattr(a, "container_id", None)
                                fid = getattr(a, "file_id", None)
                                fname = getattr(a, "filename", None)
                                if cid and cid not in seen_containers:
                                    seen_containers.add(cid)
                                if cid and fid:
                                    artifacts["files"].append({
                                        "container_id": cid,
                                        "file_id": fid,
                                        "filename": fname,
                                    })

        artifacts["container_ids"] = list(seen_containers)
        return artifacts
    
    def _parse_structured_output(self, resp) -> tuple[str, dict]:
        """
        Extract JSON and narrative from LLM response.
        
        Returns:
            tuple: (narrative_text, structured_dict)
        """
        # Try to get text from response
        text = getattr(resp, "output_text", None)
        if not text or not isinstance(text, str):
            # Try alternative extraction methods
            try:
                parts: List[str] = []
                output = getattr(resp, "output", None)
                if isinstance(output, list):
                    for item in output:
                        if isinstance(item, dict) and item.get("type") == "message":
                            for c in item.get("content", []) or []:
                                if c.get("type") in ("output_text", "text"):
                                    parts.append(c.get("text", ""))
                        elif hasattr(item, "content"):
                            for c in (item.content or []):
                                if getattr(c, "type", None) in ("output_text", "text"):
                                    parts.append(getattr(c, "text", ""))
                if parts:
                    text = "\n".join(p for p in parts if p)
            except Exception:
                pass
        
        if not text:
            text = str(resp)
        
        # Try to extract JSON block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                structured = json.loads(json_match.group(1))
                narrative = text.replace(json_match.group(0), "").strip()
                return narrative, structured
            except json.JSONDecodeError:
                pass
        
        # Fallback: try to find JSON anywhere (without code block markers)
        json_match = re.search(r'\{[^{}]*"significance"[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                structured = json.loads(json_match.group(0))
                narrative = text.replace(json_match.group(0), "").strip()
                return narrative, structured
            except json.JSONDecodeError:
                pass
        
        # No structured output found - return text as narrative
        return text, {}
    
    def get_last_artifacts(self) -> dict | None:
        """Return artifacts collected from the most recent analysis."""
        return self._last_artifacts
    
    def _get_base_url(self) -> str:
        """
        Return a usable base URL string for raw HTTP calls.
        """
        env_base = os.getenv("OPENAI_BASE_URL")
        if env_base:
            return env_base.rstrip("/")

        base = getattr(self.client, "base_url", None)
        if base is None:
            return "https://api.openai.com/v1"
        base_str = str(base)
        return base_str.rstrip("/")
    
    def download_artifacts(
        self,
        output_dir: str = "reports/plots",
        include_extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".csv", ".json", ".md"),
    ) -> List[str]:
        """
        Download container-generated files from the last analysis to local disk.

        Returns a list of saved file paths (or 'ERROR:<filename>:<exc>' strings on failure).
        """
        arts = self.get_last_artifacts()
        if not arts or not arts.get("files"):
            return []

        os.makedirs(output_dir, exist_ok=True)
        saved: List[str] = []

        base_url = self._get_base_url()
        api_key = os.getenv("OPENAI_API_KEY") or self.api_key or ""
        org = os.getenv("OPENAI_ORG_ID") or os.getenv("OPENAI_ORGANIZATION")

        # Try to discover SDK method names (versions differ)
        sdk_cf = getattr(self.client, "container_files", None)
        sdk_retrieve = getattr(sdk_cf, "retrieve_content", None) if sdk_cf else None
        sdk_content = getattr(sdk_cf, "content", None) if sdk_cf else None

        for f in arts["files"]:
            cid = f.get("container_id")
            fid = f.get("file_id")
            fname = f.get("filename") or (fid + ".bin")

            # Filter by extension if provided
            if include_extensions:
                try:
                    low = fname.lower()
                    if not any(low.endswith(ext) for ext in include_extensions):
                        continue
                except Exception:
                    pass

            out_path = Path(output_dir) / fname
            try:
                content_bytes = None

                # 1) Prefer SDK method (if present in this version)
                try:
                    if callable(sdk_retrieve):
                        resp = sdk_retrieve(container_id=cid, file_id=fid)
                        content_bytes = (
                            getattr(resp, "read", None)
                            and resp.read()
                            or getattr(resp, "content", None)
                        )
                    elif callable(sdk_content):
                        resp = sdk_content(container_id=cid, file_id=fid)
                        content_bytes = (
                            getattr(resp, "read", None)
                            and resp.read()
                            or getattr(resp, "content", None)
                        )
                except Exception:
                    content_bytes = None  # fall through to HTTP

                # 2) Raw HTTP fallback
                if content_bytes is None:
                    url = f"{base_url}/containers/{cid}/files/{fid}/content"
                    headers = {"Authorization": f"Bearer {api_key}"}
                    if org:
                        headers["OpenAI-Organization"] = org
                    with httpx.Client(timeout=60.0) as http:
                        r = http.get(url, headers=headers)
                        r.raise_for_status()
                        content_bytes = r.content

                Path(out_path).write_bytes(content_bytes)
                saved.append(str(out_path))
            except Exception as e:
                saved.append(f"ERROR:{fname}:{e}")

        return saved
