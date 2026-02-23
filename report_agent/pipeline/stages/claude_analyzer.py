# report_agent/pipeline/stages/claude_analyzer.py
"""
Anthropic Claude Code Execution implementation of LLMAnalyzer.

Uses the Anthropic Messages API with Code Execution tool to analyze metrics.
Files are uploaded via the Files API (beta) and referenced as container_upload
content blocks.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

import anthropic

from report_agent.pipeline.models import AnalysisContext, RawAnalysis
from report_agent.pipeline.stages.llm_analyzer import LLMAnalyzer
from report_agent.pipeline.exceptions import AnalysisError
from report_agent.utils.cost_tracker import get_cost_tracker

log = logging.getLogger(__name__)

_FILES_BETA = "files-api-2025-04-14"
_CODE_EXEC_TOOL = {"type": "code_execution_20250825", "name": "code_execution"}
_MAX_PAUSE_RETRIES = 8


class ClaudeAnalyzer(LLMAnalyzer):
    """
    Anthropic Claude implementation of LLMAnalyzer.

    Uses the Messages API with code execution tool to:
    - Upload context files (CSV, schema, meta, docs, catalog) via Files API
    - Run analysis with Python/Bash code execution in a sandboxed container
    - Parse structured output and narrative
    - Track artifacts for later download
    """

    def __init__(self, api_key: str, model_name: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model_name = model_name

        self.client = anthropic.Anthropic(
            api_key=api_key,
            max_retries=0,
        )

        self._last_artifacts: Optional[dict] = None

    def _record_usage(self, resp, metric_name: str) -> None:
        """Record API usage to cost tracker."""
        usage = getattr(resp, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0

            cache_created = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            if cache_created or cache_read:
                log.debug(
                    f"Cache stats [{metric_name}]: "
                    f"created={cache_created:,}, read={cache_read:,} "
                    f"(of {input_tokens:,} input tokens)"
                )

            tracker = get_cost_tracker()
            tracker.record_usage(
                category="per_metric",
                model=self.model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                metric_name=metric_name,
            )
        else:
            log.debug(f"No usage data in response for {metric_name}")

    def analyze(self, context: AnalysisContext) -> RawAnalysis:
        """
        Run Claude Code Execution analysis.

        Uploads context files, sends the prompt with code execution enabled,
        handles pause_turn for long-running analyses, and parses the result.
        """
        model = context.model_name

        # Upload files via Files API
        file_ids = self._upload_files(context)

        # Build message content: text prompt + file references.
        # The last block gets cache_control so the entire user message is
        # cached across pause_turn continuations (huge input-token savings).
        content: list = [{"type": "text", "text": context.prompt}]
        for fid in file_ids:
            content.append({"type": "container_upload", "file_id": fid})
        if content:
            content[-1]["cache_control"] = {"type": "ephemeral"}

        try:
            log.debug(f"Calling Anthropic API for model '{model}'")
            resp = self.client.beta.messages.create(
                model=self.model_name,
                betas=[_FILES_BETA],
                max_tokens=16384,
                messages=[{"role": "user", "content": content}],
                tools=[_CODE_EXEC_TOOL],
            )
        except Exception as e:
            error_msg = str(e)
            log.error(f"API call failed for {model}: {error_msg}")
            raise AnalysisError(f"API call failed: {error_msg}", model)

        # Handle pause_turn for long-running code execution
        resp = self._handle_pause_turn(resp, content)

        # Debug: log response structure so we can diagnose parsing issues
        self._debug_log_response(resp)

        # Track API usage/cost
        self._record_usage(resp, model)

        # Extract artifacts (generated files like plots)
        try:
            self._last_artifacts = self._extract_artifacts(resp)
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

    def _handle_pause_turn(self, resp, original_content: list):
        """
        Handle pause_turn stop reason for long-running code execution.

        When the API pauses a long turn, we send the response back to let
        Claude continue where it left off, reusing the same container.
        """
        messages = [{"role": "user", "content": original_content}]

        for i in range(_MAX_PAUSE_RETRIES):
            if getattr(resp, "stop_reason", None) != "pause_turn":
                break

            log.debug(f"Received pause_turn (attempt {i + 1}), continuing...")

            # Append the partial assistant response so Claude can continue
            messages.append({"role": "assistant", "content": resp.content})

            container_id = resp.container.id if hasattr(resp, "container") else None

            try:
                kwargs = dict(
                    model=self.model_name,
                    betas=[_FILES_BETA],
                    max_tokens=16384,
                    messages=messages,
                    tools=[_CODE_EXEC_TOOL],
                )
                if container_id:
                    kwargs["container"] = container_id

                resp = self.client.beta.messages.create(**kwargs)

                # Accumulate usage across continuations
                self._record_usage(resp, "pause_turn_continuation")
            except Exception as e:
                log.warning(f"pause_turn continuation failed: {e}")
                break

        return resp

    def _upload_files(self, context: AnalysisContext) -> List[str]:
        """Upload all context files to Anthropic via Files API and return file IDs."""
        file_ids: List[str] = []

        for path in context.get_all_file_paths():
            if path and path.exists():
                try:
                    with open(path, "rb") as f:
                        file_obj = self.client.beta.files.upload(
                            file=f,
                            betas=[_FILES_BETA],
                        )
                    file_ids.append(file_obj.id)
                except Exception as e:
                    log.warning(f"Failed to upload {path.name}: {e}")

        return file_ids

    def _debug_log_response(self, resp) -> None:
        """Log the response structure for debugging (only at DEBUG level)."""
        if not log.isEnabledFor(logging.DEBUG):
            return

        content = resp.content or []
        log.debug(f"Response has {len(content)} content blocks, stop_reason={getattr(resp, 'stop_reason', '?')}")
        for i, item in enumerate(content):
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text = getattr(item, "text", "")
                preview = text[:120].replace("\n", "\\n")
                log.debug(f"  [{i}] text ({len(text)} chars): {preview}...")
            else:
                log.debug(f"  [{i}] {item_type}")

    def _extract_artifacts(self, resp) -> dict | None:
        """
        Walk the response content and collect file IDs from code execution results.

        Returns a dict: {"files": [{"file_id", "filename"}]}
        """
        artifacts: list = []

        for item in (resp.content or []):
            if getattr(item, "type", None) == "bash_code_execution_tool_result":
                content_item = getattr(item, "content", None)
                if content_item is None:
                    continue

                result_content = getattr(content_item, "content", None)
                if not result_content:
                    continue

                for file_entry in result_content:
                    fid = getattr(file_entry, "file_id", None)
                    if fid:
                        artifacts.append({
                            "file_id": fid,
                            "filename": getattr(file_entry, "filename", None),
                        })

        return {"files": artifacts} if artifacts else None

    def _parse_structured_output(self, resp) -> tuple[str, dict]:
        """
        Extract JSON and narrative from Claude's response.

        Strategy:
        1. Collect ALL text blocks from the response (we need them to find
           the JSON no matter where Claude placed it).
        2. Extract the structured JSON.
        3. For the narrative, only use text that appears AFTER the last
           tool-result block, and strip any transitional preamble.
        """
        content = resp.content or []

        # --- Step 1: Gather all text (for JSON extraction) ---
        all_text_parts: List[str] = []
        for item in content:
            if getattr(item, "type", None) == "text":
                all_text_parts.append(getattr(item, "text", ""))
        all_text = "\n".join(p for p in all_text_parts if p)

        # --- Step 2: Extract structured JSON from anywhere in the response ---
        structured = {}
        json_block_text = ""

        # Try fenced ```json block first
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', all_text, re.DOTALL)
        if json_match:
            try:
                structured = json.loads(json_match.group(1))
                json_block_text = json_match.group(0)
            except json.JSONDecodeError:
                pass

        # Fallback: bare JSON with "significance" key
        if not structured:
            json_match = re.search(r'\{[^{}]*"significance"[^{}]*\}', all_text, re.DOTALL)
            if json_match:
                try:
                    structured = json.loads(json_match.group(0))
                    json_block_text = json_match.group(0)
                except json.JSONDecodeError:
                    pass

        # --- Step 3: Build narrative from final text only ---
        last_tool_idx = -1
        for i, item in enumerate(content):
            item_type = getattr(item, "type", None)
            if item_type in (
                "bash_code_execution_tool_result",
                "text_editor_code_execution_tool_result",
                "server_tool_use",
            ):
                last_tool_idx = i

        final_parts: List[str] = []
        for i, item in enumerate(content):
            if getattr(item, "type", None) == "text" and i > last_tool_idx:
                final_parts.append(getattr(item, "text", ""))

        narrative = "\n".join(p for p in final_parts if p)

        # Remove the JSON block from the narrative if it's there
        if json_block_text and json_block_text in narrative:
            narrative = narrative.replace(json_block_text, "")

        # Strip Claude's transitional preamble lines
        narrative = self._strip_preamble(narrative).strip()

        if not narrative and not structured:
            narrative = all_text or str(resp)

        return narrative, structured

    @staticmethod
    def _strip_preamble(text: str) -> str:
        """
        Remove Claude's transitional lead-in sentences that sometimes
        precede the actual analysis, e.g.:
          "Now let me provide my analysis:"
          "Here is my assessment:"
          "Based on my analysis, here are my findings:"
        """
        lines = text.split("\n")
        cleaned: List[str] = []
        past_preamble = False

        for line in lines:
            stripped = line.strip().lower()
            if not past_preamble and not stripped:
                continue
            if not past_preamble and re.match(
                r'^(now\s+)?'
                r'(let\s+me\s+|here\s+(is|are)\s+|based\s+on\s+)'
                r'.{0,80}'
                r'(analysis|assessment|findings|report|results|summary)\s*[:.]?\s*$',
                stripped,
            ):
                continue
            past_preamble = True
            cleaned.append(line)

        return "\n".join(cleaned)

    def get_last_artifacts(self) -> dict | None:
        """Return artifacts collected from the most recent analysis."""
        return self._last_artifacts

    def download_artifacts(
        self,
        output_dir: str = "reports/plots",
        include_extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".csv", ".json", ".md"),
    ) -> List[str]:
        """
        Download container-generated files from the last analysis to local disk.

        Uses the Anthropic Files API to retrieve file metadata and content.
        Returns a list of saved file paths (or 'ERROR:<filename>:<exc>' on failure).
        """
        arts = self.get_last_artifacts()
        if not arts or not arts.get("files"):
            return []

        os.makedirs(output_dir, exist_ok=True)
        saved: List[str] = []

        for f in arts["files"]:
            fid = f.get("file_id")
            fname = f.get("filename")

            if not fid:
                continue

            # Filter by extension
            if include_extensions and fname:
                low = fname.lower()
                if not any(low.endswith(ext) for ext in include_extensions):
                    continue

            try:
                # Get metadata for filename if we don't have it
                if not fname:
                    try:
                        meta = self.client.beta.files.retrieve_metadata(
                            file_id=fid,
                            betas=[_FILES_BETA],
                        )
                        fname = getattr(meta, "filename", None) or f"{fid}.bin"
                    except Exception:
                        fname = f"{fid}.bin"

                    # Re-check extension filter with resolved filename
                    if include_extensions:
                        low = fname.lower()
                        if not any(low.endswith(ext) for ext in include_extensions):
                            continue

                out_path = Path(output_dir) / fname

                file_content = self.client.beta.files.download(
                    file_id=fid,
                    betas=[_FILES_BETA],
                )

                # The SDK returns a streaming response; write it to file
                if hasattr(file_content, "write_to_file"):
                    file_content.write_to_file(str(out_path))
                elif hasattr(file_content, "read"):
                    out_path.write_bytes(file_content.read())
                else:
                    out_path.write_bytes(bytes(file_content))

                saved.append(str(out_path))
            except Exception as e:
                saved.append(f"ERROR:{fname or fid}:{e}")

        return saved
