"""A scripted author/reviewer for composed pipeline tests; no production fallback model."""

from __future__ import annotations

import json
from typing import Any

from causal.shared.gateway import GatewayImage, GatewayResultV1


class PostAnalysisGateway:
    def __init__(self) -> None:
        self.author_calls = 0
        self.review_calls = 0
        self.preview_hashes: list[str] = []

    def invoke(
        self,
        envelope: Any,
        prompt: str,
        schema: dict[str, object],
        *,
        images: tuple[GatewayImage, ...] = (),
    ) -> GatewayResultV1:
        import hashlib

        payload = envelope.payload
        if envelope.task_kind == "post_analysis_review":
            self.review_calls += 1
            assert images, "review must receive the actual final page previews"
            assert all(
                image.mime_type == "image/png" and image.data.startswith(b"\x89PNG")
                for image in images
            )
            self.preview_hashes = [hashlib.sha256(image.data).hexdigest() for image in images]
            body = {
                "verdict": "pass",
                "issues": [],
                "decision_summary": "The scripted review received all final page previews.",
            }
        else:
            assert envelope.task_kind == "post_analysis_author"
            assert not images
            self.author_calls += 1
            if not payload["visuals"]:
                body = {
                    "tool": "render_visual",
                    "arguments": {
                        "table_id": "primary",
                        "kind": "table",
                        "title": "Frozen primary estimates and supplied uncertainty",
                        "caption": "Values and uncertainty methods are copied from the primary analysis.",
                    },
                    "decision_summary": "Display the exact primary numerical table.",
                }
            elif not payload["current_report"]:
                statements = []
                coverage = {}
                for evidence_id in payload["required_evidence"]:
                    row = payload["evidence"][evidence_id]
                    status = row.get("execution_status", row.get("status", "supplied"))
                    explanation = f"{evidence_id}: execution status {status}. "
                    if status != "complete" and status != "supplied":
                        explanation += "This result does not establish that the diagnostic passed. "
                    reason = (
                        row.get("unavailable_reason") or row.get("error_code") or row.get("reason")
                    )
                    if reason:
                        explanation += f"Recorded explanation: {reason}. "
                    explanation += (
                        "Causal interpretation remains conditional on the approved design."
                    )
                    coverage[evidence_id] = explanation
                    statements.append(
                        {
                            "text": explanation,
                            "citations": [{"evidence_id": evidence_id, "selector": ""}],
                        }
                    )
                body = {
                    "tool": "write_report",
                    "arguments": {
                        "title": "Evidence and causal limitations",
                        "sections": [
                            {
                                "title": "Analysis evidence",
                                "statements": statements,
                                "visual_ids": list(payload["visuals"]),
                            }
                        ],
                        "coverage": coverage,
                    },
                    "decision_summary": "Account for each requested result with an exact source citation.",
                }
            else:
                body = {
                    "tool": "submit",
                    "arguments": {},
                    "decision_summary": "Submit this exact report revision for preview review.",
                }
        return GatewayResultV1(
            text=json.dumps(body), parsed=body, token_usage={}, attempts=1, seed=1
        )
