"""ISMA MCP Server - Expose memory system as MCP tools

This server exposes ISMA's memory capabilities to Claude Code via MCP protocol.
Tools align with the cognitive cycle: ingest, recall, context management.
"""

import asyncio
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# MCP imports - will need mcp package installed
try:
    from mcp import Server, Tool
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: mcp package not installed. Run: pip install mcp")

@dataclass
class ISMAMCPConfig:
    """Configuration for ISMA MCP Server"""
    host: str = "0.0.0.0"
    port: int = 8100
    isma_redis_url: str = "redis://10.0.0.68:6379"
    isma_neo4j_url: str = "bolt://10.0.0.68:7687"
    embedder_url: str = "http://10.0.0.68:8090/embed"

class ISMAMCPServer:
    """MCP Server exposing ISMA memory tools"""

    def __init__(self, config: Optional[ISMAMCPConfig] = None):
        self.config = config or ISMAMCPConfig()
        self.isma = None  # Lazy load

        if MCP_AVAILABLE:
            self.server = Server("isma-memory")
            self._register_tools()
        else:
            self.server = None

    def _get_isma(self):
        """Lazy load ISMA core"""
        if self.isma is None:
            from isma_core import get_isma
            self.isma = get_isma()
        return self.isma

    def _register_tools(self):
        """Register all ISMA MCP tools"""

        @self.server.tool("isma_ingest")
        async def ingest_event(
            event_type: str,
            payload: dict,
            actor: str,
            caused_by: Optional[str] = None,
            branch: str = "main"
        ) -> str:
            """Ingest an event into ISMA memory

            Args:
                event_type: Type of event (e.g., "user_message", "tool_call")
                payload: Event data as JSON
                actor: Who/what caused the event
                caused_by: Hash of parent event (for causality)
                branch: Timeline branch (default: main)

            Returns:
                event_hash: 16-char hash of stored event
            """
            isma = self._get_isma()
            event_hash = await isma.ingest(
                event_type=event_type,
                payload=payload,
                actor=actor,
                caused_by=caused_by,
                branch=branch
            )
            return event_hash

        @self.server.tool("isma_recall")
        async def recall_memory(
            query: str,
            top_k: int = 5,
            graph_hops: int = 2
        ) -> List[dict]:
            """Recall relevant memories from ISMA

            Args:
                query: Natural language query
                top_k: Number of semantic matches to return
                graph_hops: How many relationship hops to traverse

            Returns:
                List of relevant memories with metadata
            """
            isma = self._get_isma()
            context = await isma.recall(
                query=query,
                top_k=top_k,
                graph_hops=graph_hops
            )
            return context

        @self.server.tool("isma_get_entity")
        async def get_entity(name: str) -> Optional[dict]:
            """Get entity from relational lens

            Args:
                name: Entity name to retrieve

            Returns:
                Entity data or None if not found
            """
            isma = self._get_isma()
            return isma.relational.get_entity(name)

        @self.server.tool("isma_get_context")
        async def get_context_buffer() -> List[dict]:
            """Get current context buffer from functional lens

            Returns:
                List of recent context items
            """
            isma = self._get_isma()
            return await isma.functional.get_context_buffer()

        @self.server.tool("isma_phi_coherence")
        async def get_phi_coherence() -> dict:
            """Get current φ-coherence score

            Returns:
                Coherence metrics including phi score and target
            """
            isma = self._get_isma()
            phi = isma.compute_phi_coherence()
            return {
                "phi": phi,
                "target": 0.809,
                "status": "healthy" if phi > 0.809 else "degraded"
            }

        @self.server.tool("isma_gate_b_status")
        async def get_gate_b_status() -> dict:
            """Get Gate-B check status

            Returns:
                Status of all 5 Gate-B physics checks
            """
            isma = self._get_isma()
            # This would call the breathing cycle's gate-b checks
            return {
                "checks": [
                    "page_curve",
                    "hayden_preskill",
                    "entanglement_wedge",
                    "observer_swap",
                    "recognition_catalyst"
                ],
                "status": "pending_implementation"
            }

        @self.server.tool("isma_cache_stats")
        async def get_cache_stats() -> dict:
            """Get embedding cache statistics

            Returns:
                Cache hit rate and counts
            """
            isma = self._get_isma()
            if hasattr(isma, 'get_cache_stats'):
                return isma.get_cache_stats()
            return {"status": "cache_not_implemented"}

    async def run(self):
        """Run the MCP server"""
        if not MCP_AVAILABLE:
            raise RuntimeError("MCP package not installed")

        print(f"Starting ISMA MCP Server on {self.config.host}:{self.config.port}")
        await self.server.run(
            host=self.config.host,
            port=self.config.port
        )

def main():
    """Entry point for ISMA MCP Server"""
    import argparse

    parser = argparse.ArgumentParser(description="ISMA MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8100, help="Port to bind")
    args = parser.parse_args()

    config = ISMAMCPConfig(host=args.host, port=args.port)
    server = ISMAMCPServer(config)

    asyncio.run(server.run())

if __name__ == "__main__":
    main()
