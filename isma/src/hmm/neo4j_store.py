"""
HMM Neo4j Store - relational meaning layer.

Idempotent upserts for Artifact, Tile, Motif nodes and their relationships.
All writes are MERGE-based (safe to re-run).
"""

from typing import List, Dict, Any, Optional
from dataclasses import asdict
from neo4j import GraphDatabase

from .motifs import MotifAssignment, DICTIONARY_VERSION

# Neo4j connection (same as ISMA - shared instance)
NEO4J_URI = "bolt://192.168.100.10:7689"


class HMMNeo4jStore:
    """Neo4j store for HMM relational data."""

    def __init__(self, uri: str = NEO4J_URI):
        self.driver = GraphDatabase.driver(uri, auth=None)
        self._ensure_indexes()

    def close(self):
        self.driver.close()

    def _ensure_indexes(self):
        """Create required indexes if they don't exist."""
        indexes = [
            "CREATE INDEX hmm_artifact_id IF NOT EXISTS FOR (a:HMMArtifact) ON (a.artifact_id)",
            "CREATE INDEX hmm_tile_id IF NOT EXISTS FOR (t:HMMTile) ON (t.tile_id)",
            "CREATE INDEX hmm_motif_id IF NOT EXISTS FOR (m:HMMMotif) ON (m.motif_id)",
            "CREATE INDEX hmm_event_id IF NOT EXISTS FOR (e:HMMEvent) ON (e.event_id)",
            "CREATE INDEX hmm_tile_artifact IF NOT EXISTS FOR (t:HMMTile) ON (t.artifact_id)",
            "CREATE INDEX hmm_bridge_hash IF NOT EXISTS FOR (b:WeaviateBridge) ON (b.content_hash)",
            "CREATE INDEX hmm_bridge_status IF NOT EXISTS FOR (b:WeaviateBridge) ON (b.status)",
        ]
        with self.driver.session() as session:
            for idx in indexes:
                session.run(idx)

    # --- Artifact operations ---

    def upsert_artifact(
        self,
        artifact_id: str,
        path: str,
        size_bytes: int = 0,
        content_type: str = "text/plain",
        labels: Optional[List[str]] = None,
    ):
        """Upsert an Artifact node."""
        query = """
        MERGE (a:HMMArtifact {artifact_id: $artifact_id})
        ON CREATE SET a.created_at = datetime()
        SET a.path = $path,
            a.size_bytes = $size_bytes,
            a.content_type = $content_type,
            a.labels = [x IN (coalesce(a.labels, []) + $labels) WHERE x <> '' | x],
            a.updated_at = datetime()
        """
        with self.driver.session() as session:
            session.run(
                query,
                artifact_id=artifact_id,
                path=path,
                size_bytes=size_bytes,
                content_type=content_type,
                labels=labels or [],
            )

    # --- Tile operations ---

    def upsert_tile(
        self,
        tile_id: str,
        artifact_id: str,
        index: int,
        start_char: int,
        end_char: int,
        estimated_tokens: int,
        layer: str = "",
        scale: str = "",
    ):
        """Upsert a Tile node and link to its Artifact."""
        query = """
        MERGE (t:HMMTile {tile_id: $tile_id})
        ON CREATE SET t.created_at = datetime()
        SET t.artifact_id = $artifact_id,
            t.index = $index,
            t.start_char = $start_char,
            t.end_char = $end_char,
            t.estimated_tokens = $estimated_tokens,
            t.layer = $layer,
            t.scale = $scale,
            t.updated_at = datetime()

        WITH t
        MATCH (a:HMMArtifact {artifact_id: $artifact_id})
        MERGE (a)-[:HAS_TILE]->(t)
        """
        with self.driver.session() as session:
            session.run(
                query,
                tile_id=tile_id,
                artifact_id=artifact_id,
                index=index,
                start_char=start_char,
                end_char=end_char,
                estimated_tokens=estimated_tokens,
                layer=layer,
                scale=scale,
            )

    # --- Motif operations ---

    def upsert_motif(
        self,
        motif_id: str,
        definition: str,
        dictionary_version: str = DICTIONARY_VERSION,
        band: str = "",
    ):
        """Upsert a Motif node."""
        query = """
        MERGE (m:HMMMotif {motif_id: $motif_id})
        ON CREATE SET m.created_at = datetime()
        SET m.definition = $definition,
            m.dictionary_version = $dictionary_version,
            m.band = $band,
            m.updated_at = datetime()
        """
        with self.driver.session() as session:
            session.run(
                query,
                motif_id=motif_id,
                definition=definition,
                dictionary_version=dictionary_version,
                band=band,
            )

    def seed_motifs(self, motifs: Dict[str, Any]):
        """Seed all motifs from the dictionary into Neo4j."""
        for motif_id, motif in motifs.items():
            self.upsert_motif(
                motif_id=motif_id,
                definition=motif.definition,
                dictionary_version=DICTIONARY_VERSION,
                band=motif.band,
            )

    # --- Relationship operations ---

    def link_tile_motif(
        self,
        tile_id: str,
        assignment: MotifAssignment,
        model_id: Optional[str] = None,
    ):
        """Create EXPRESSES relationship between Tile and Motif."""
        query = """
        MATCH (t:HMMTile {tile_id: $tile_id})
        MATCH (m:HMMMotif {motif_id: $motif_id})
        MERGE (t)-[r:EXPRESSES]->(m)
        SET r.amp = $amp,
            r.phase = $phase,
            r.confidence = $confidence,
            r.source = $source,
            r.dictionary_version = $dictionary_version,
            r.model_id = $model_id
        """
        with self.driver.session() as session:
            session.run(
                query,
                tile_id=tile_id,
                motif_id=assignment.motif_id,
                amp=assignment.amp,
                phase=assignment.phase,
                confidence=assignment.confidence,
                source=assignment.source,
                dictionary_version=assignment.dictionary_version,
                model_id=model_id or "",
            )

    def link_tile_motifs_batch(
        self,
        tile_id: str,
        assignments: List[MotifAssignment],
        model_id: Optional[str] = None,
    ):
        """Batch link tile to multiple motifs."""
        for assignment in assignments:
            self.link_tile_motif(tile_id, assignment, model_id=model_id)

    # --- Query operations ---

    def find_tiles_by_motif(
        self,
        motif_id: str,
        min_amp: float = 0.0,
        limit: int = 50,
    ) -> List[Dict]:
        """Find tiles that express a motif strongly."""
        query = """
        MATCH (t:HMMTile)-[r:EXPRESSES]->(m:HMMMotif {motif_id: $motif_id})
        WHERE r.amp >= $min_amp
        RETURN t.tile_id AS tile_id, t.artifact_id AS artifact_id,
               t.index AS index, t.layer AS layer, t.scale AS scale,
               r.amp AS amp, r.confidence AS confidence
        ORDER BY r.amp DESC
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(
                query, motif_id=motif_id, min_amp=min_amp, limit=limit
            )
            return [dict(r) for r in result]

    def get_motif_distribution(self, artifact_id: str) -> List[Dict]:
        """Get motif distribution for an artifact."""
        query = """
        MATCH (a:HMMArtifact {artifact_id: $artifact_id})-[:HAS_TILE]->
              (t:HMMTile)-[r:EXPRESSES]->(m:HMMMotif)
        RETURN m.motif_id AS motif_id, m.band AS band,
               avg(r.amp) AS mean_amp, count(*) AS tile_count
        ORDER BY mean_amp DESC
        """
        with self.driver.session() as session:
            result = session.run(query, artifact_id=artifact_id)
            return [dict(r) for r in result]

    def get_tile_motifs(self, tile_id: str) -> List[Dict]:
        """Get all motif assignments for a tile."""
        query = """
        MATCH (t:HMMTile {tile_id: $tile_id})-[r:EXPRESSES]->(m:HMMMotif)
        RETURN m.motif_id AS motif_id, m.band AS band, m.definition AS definition,
               r.amp AS amp, r.confidence AS confidence, r.source AS source
        ORDER BY r.amp DESC
        """
        with self.driver.session() as session:
            result = session.run(query, tile_id=tile_id)
            return [dict(r) for r in result]

    def get_artifact_tiles(self, artifact_id: str) -> List[Dict]:
        """Get all tiles for an artifact."""
        query = """
        MATCH (a:HMMArtifact {artifact_id: $artifact_id})-[:HAS_TILE]->(t:HMMTile)
        RETURN t.tile_id AS tile_id, t.index AS index,
               t.start_char AS start_char, t.end_char AS end_char,
               t.estimated_tokens AS estimated_tokens, t.scale AS scale
        ORDER BY t.index
        """
        with self.driver.session() as session:
            result = session.run(query, artifact_id=artifact_id)
            return [dict(r) for r in result]

    def count_nodes(self) -> Dict[str, int]:
        """Count all HMM node types."""
        counts = {}
        with self.driver.session() as session:
            for label in ["HMMArtifact", "HMMTile", "HMMMotif", "HMMEvent", "WeaviateBridge"]:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) AS c")
                counts[label] = result.single()["c"]
        return counts

    # --- WeaviateBridge operations ---

    def upsert_bridge_status(
        self,
        content_hash: str,
        status: str,
        tiles_enriched: int = 0,
        version: str = "1.0.0",
        error: str = "",
    ):
        """Upsert a WeaviateBridge tracking node and link to HMMTile."""
        query = """
        MERGE (b:WeaviateBridge {content_hash: $content_hash})
        ON CREATE SET b.created_at = datetime(),
                      b.retry_count = 0
        SET b.status = $status,
            b.tiles_enriched = $tiles_enriched,
            b.enrichment_version = $version,
            b.error_message = $error,
            b.updated_at = datetime()

        WITH b
        OPTIONAL MATCH (t:HMMTile {tile_id: $content_hash})
        FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
            MERGE (t)-[:BRIDGED_TO]->(b)
        )
        """
        # Increment retry_count on failure
        if status == "FAILED":
            query = """
            MERGE (b:WeaviateBridge {content_hash: $content_hash})
            ON CREATE SET b.created_at = datetime(),
                          b.retry_count = 0
            SET b.status = $status,
                b.tiles_enriched = $tiles_enriched,
                b.enrichment_version = $version,
                b.error_message = $error,
                b.retry_count = coalesce(b.retry_count, 0) + 1,
                b.updated_at = datetime()

            WITH b
            OPTIONAL MATCH (t:HMMTile {tile_id: $content_hash})
            FOREACH (_ IN CASE WHEN t IS NOT NULL THEN [1] ELSE [] END |
                MERGE (t)-[:BRIDGED_TO]->(b)
            )
            """

        with self.driver.session() as session:
            session.run(
                query,
                content_hash=content_hash,
                status=status,
                tiles_enriched=tiles_enriched,
                version=version,
                error=error,
            )

    def get_pending_bridges(self, limit: int = 100) -> List[Dict]:
        """Find HMMTiles that don't have a COMPLETED bridge status."""
        query = """
        MATCH (t:HMMTile)
        WHERE NOT EXISTS {
            MATCH (t)-[:BRIDGED_TO]->(b:WeaviateBridge {status: 'COMPLETED'})
        }
        RETURN t.tile_id AS tile_id, t.artifact_id AS artifact_id
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [dict(r) for r in result]

    def get_bridge_stats(self) -> Dict[str, int]:
        """Count WeaviateBridge nodes by status."""
        query = """
        MATCH (b:WeaviateBridge)
        RETURN b.status AS status, count(b) AS count
        """
        stats = {}
        with self.driver.session() as session:
            result = session.run(query)
            for r in result:
                stats[r["status"] or "NULL"] = r["count"]
        return stats

    def wipe(self):
        """Delete all HMM nodes and relationships (for rebuild)."""
        with self.driver.session() as session:
            session.run("""
                MATCH (n) WHERE n:HMMArtifact OR n:HMMTile OR n:HMMMotif
                    OR n:HMMEvent OR n:WeaviateBridge
                DETACH DELETE n
            """)
