"""
GraphRAG Extension for Agent Zero
Extension Point: message_loop_prompts_after

Injects graph-derived context into the agent's prompt extras when:
  1. GRAPH_RAG_ENABLED=true
  2. Neo4j is reachable

NO core patches. Uses only the upstream Extensions Framework.
"""

import os
import logging
from python.helpers.extension import Extension

logger = logging.getLogger(__name__)

# Lazy import to avoid hard dependency on the graphrag package
_graphrag_available = None


def _check_graphrag():
    """Check if graphrag_agent_zero package is importable."""
    global _graphrag_available
    if _graphrag_available is None:
        try:
            from graphrag_agent_zero.extension_hook import is_enabled, is_neo4j_available  # noqa: F401
            _graphrag_available = True
        except ImportError:
            _graphrag_available = False
            logger.info("graphrag_agent_zero package not installed — GraphRAG disabled")
    return _graphrag_available


class GraphRAGExtension(Extension):
    """
    Inject graph-derived context into the agent's prompt.

    Follows the same pattern as _50_recall_memories.py:
    - Reads loop_data.extras_persistent
    - Adds a "graphrag" key with enriched context
    - Agent Zero's prompt system picks it up automatically
    """

    async def execute(self, loop_data=None, **kwargs):
        # Gate 1: feature flag
        if os.getenv("GRAPH_RAG_ENABLED", "false").lower() != "true":
            return

        # Gate 2: package available
        if not _check_graphrag():
            return

        # Gate 3: Neo4j reachable
        from graphrag_agent_zero.extension_hook import (
            is_neo4j_available,
            enhance_retrieval,
        )

        if not is_neo4j_available():
            logger.debug("GraphRAG enabled but Neo4j unavailable — skipping")
            return

        # Get the user message from loop_data
        if loop_data is None:
            return

        user_msg = ""
        if hasattr(loop_data, "user_message") and loop_data.user_message:
            user_msg = loop_data.user_message.output_text()

        if not user_msg:
            return

        try:
            # Call the GraphRAG retrieval pipeline
            result = enhance_retrieval(
                query=user_msg,
                vector_results=[],  # Agent Zero's own memory handles vector search
            )

            # Inject into prompt extras if graph-derived content was found
            if result.get("graph_derived") and result.get("text"):
                extras = loop_data.extras_persistent
                extras["graphrag"] = (
                    "## Graph Knowledge (GraphRAG)\n"
                    + result["text"]
                )

                if result.get("entities"):
                    entity_list = ", ".join(result["entities"][:20])
                    extras["graphrag"] += f"\n\nRelated entities: {entity_list}"

                logger.debug(
                    f"GraphRAG injected {len(result.get('entities', []))} entities, "
                    f"latency={result.get('latency_ms', 0):.0f}ms"
                )
        except Exception as e:
            # Never crash the agent — graceful no-op
            logger.warning(f"GraphRAG extension error (no-op): {e}")
