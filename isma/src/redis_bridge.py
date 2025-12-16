#!/usr/bin/env python3
"""
Redis Bridge - Connects MCP Server (v4) to ISMA (v3)

This daemon:
1. Consumes events from Redis stream (written by MCP server)
2. Ingests them into ISMA via the single write path
3. Runs consolidation on a timer

Stream Keys:
- taey:stream:mcp_events - MCP tool calls and results
- isma:stream:events - ISMA events (for downstream consumers)

Usage:
    python3 -m src.memory.redis_bridge

    # Or as background daemon:
    nohup python3 -m src.memory.redis_bridge &
"""

import json
import time
import signal
import sys
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import redis

try:
    from .isma_core import get_isma, ISMACore
except ImportError:
    from isma_core import get_isma, ISMACore


# Configuration
REDIS_HOST = '10.0.0.68'
REDIS_PORT = 6379
MCP_STREAM = 'taey:stream:mcp_events'
CONSUMER_GROUP = 'isma_bridge'
CONSUMER_NAME = 'bridge_worker_1'
CONSOLIDATION_INTERVAL = 3.236  # One breathing cycle


class RedisBridge:
    """
    Bridge between MCP Server and ISMA.

    Consumes MCP events from Redis stream and ingests to ISMA.
    """

    def __init__(self,
                 redis_host: str = REDIS_HOST,
                 redis_port: int = REDIS_PORT,
                 isma: ISMACore = None):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self._redis: Optional[redis.Redis] = None
        self.isma = isma or get_isma()

        self._running = False
        self._consumer_thread: Optional[threading.Thread] = None
        self._consolidation_thread: Optional[threading.Thread] = None

        # Stats
        self._events_processed = 0
        self._errors = 0
        self._last_consolidation = None

    def _get_redis(self) -> redis.Redis:
        """Get Redis connection."""
        if self._redis is None:
            self._redis = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True
            )
        return self._redis

    def initialize(self) -> bool:
        """Initialize the bridge."""
        try:
            r = self._get_redis()
            r.ping()

            # Create consumer group if not exists
            try:
                r.xgroup_create(MCP_STREAM, CONSUMER_GROUP, id='0', mkstream=True)
                print(f"Created consumer group {CONSUMER_GROUP} on {MCP_STREAM}")
            except redis.ResponseError as e:
                if "BUSYGROUP" in str(e):
                    print(f"Consumer group {CONSUMER_GROUP} already exists")
                else:
                    raise

            return True

        except Exception as e:
            print(f"Bridge initialization failed: {e}")
            return False

    def start(self):
        """Start the bridge daemon."""
        if self._running:
            return

        print(f"Starting Redis Bridge...")
        print(f"  Stream: {MCP_STREAM}")
        print(f"  Consumer Group: {CONSUMER_GROUP}")
        print(f"  Consumer: {CONSUMER_NAME}")

        self._running = True

        # Start consumer thread
        self._consumer_thread = threading.Thread(
            target=self._consume_loop,
            daemon=True
        )
        self._consumer_thread.start()

        # Start consolidation thread
        self._consolidation_thread = threading.Thread(
            target=self._consolidation_loop,
            daemon=True
        )
        self._consolidation_thread.start()

        # Log start event
        self.isma.ingest(
            event_type='bridge',
            payload={
                'operation': 'started',
                'stream': MCP_STREAM,
                'consumer': CONSUMER_NAME
            },
            actor='redis_bridge'
        )

        print("Redis Bridge started.")

    def stop(self):
        """Stop the bridge."""
        self._running = False

        if self._consumer_thread:
            self._consumer_thread.join(timeout=5)
        if self._consolidation_thread:
            self._consolidation_thread.join(timeout=5)

        # Log stop event
        self.isma.ingest(
            event_type='bridge',
            payload={
                'operation': 'stopped',
                'events_processed': self._events_processed,
                'errors': self._errors
            },
            actor='redis_bridge'
        )

        print(f"Redis Bridge stopped. Processed {self._events_processed} events, {self._errors} errors.")

    def _consume_loop(self):
        """Main consumer loop."""
        r = self._get_redis()

        while self._running:
            try:
                # Read from stream (blocking with timeout)
                messages = r.xreadgroup(
                    CONSUMER_GROUP,
                    CONSUMER_NAME,
                    {MCP_STREAM: '>'},
                    count=10,
                    block=1000  # 1 second timeout
                )

                if not messages:
                    continue

                for stream_name, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        try:
                            self._process_message(msg_id, msg_data)
                            # Acknowledge the message
                            r.xack(MCP_STREAM, CONSUMER_GROUP, msg_id)
                            self._events_processed += 1
                        except Exception as e:
                            print(f"Message processing error: {e}")
                            self._errors += 1

            except Exception as e:
                print(f"Consumer loop error: {e}")
                time.sleep(1)

    def _process_message(self, msg_id: str, msg_data: Dict[str, str]):
        """Process a single message from the stream."""
        # Extract fields
        event_type = msg_data.get('event_type', 'mcp_event')
        actor = msg_data.get('actor', 'mcp_server')
        platform = msg_data.get('platform')
        tool_name = msg_data.get('tool')
        operation = msg_data.get('operation', 'recorded')

        # Parse payload if JSON
        payload_str = msg_data.get('payload', '{}')
        try:
            payload = json.loads(payload_str)
        except json.JSONDecodeError:
            payload = {'raw': payload_str}

        # Add metadata
        payload['operation'] = operation
        payload['platform'] = platform
        payload['tool'] = tool_name
        payload['mcp_msg_id'] = msg_id

        # Extract caused_by if present
        caused_by = msg_data.get('caused_by')

        # Ingest to ISMA
        event_hash = self.isma.ingest(
            event_type=event_type,
            payload=payload,
            actor=actor,
            caused_by=caused_by
        )

        # Log occasionally
        if self._events_processed % 100 == 0:
            print(f"Processed {self._events_processed} events. Latest: {event_hash}")

    def _consolidation_loop(self):
        """Run consolidation on a timer."""
        while self._running:
            try:
                time.sleep(CONSOLIDATION_INTERVAL)

                if not self._running:
                    break

                # Run consolidation
                metrics = self.isma.consolidate_pending(batch_size=20)
                self._last_consolidation = datetime.now().isoformat()

                # Log if anything was processed
                if metrics.get('processed', 0) > 0:
                    print(f"Consolidation: {metrics}")

            except Exception as e:
                print(f"Consolidation error: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            'running': self._running,
            'events_processed': self._events_processed,
            'errors': self._errors,
            'last_consolidation': self._last_consolidation,
            'phi_coherence': self.isma.compute_phi_coherence(),
            'is_coherent': self.isma.is_coherent()
        }


# Global bridge instance
_bridge: Optional[RedisBridge] = None


def get_bridge() -> RedisBridge:
    """Get the bridge singleton."""
    global _bridge
    if _bridge is None:
        _bridge = RedisBridge()
    return _bridge


def start_bridge() -> RedisBridge:
    """Start the Redis bridge."""
    bridge = get_bridge()
    if bridge.initialize():
        bridge.start()
    return bridge


def stop_bridge():
    """Stop the Redis bridge."""
    global _bridge
    if _bridge:
        _bridge.stop()


def main():
    """Main entry point for running as daemon."""
    print("=" * 60)
    print("ISMA Redis Bridge")
    print("Connecting MCP Server (v4) to ISMA (v3)")
    print("=" * 60)

    bridge = start_bridge()

    # Handle signals
    def signal_handler(sig, frame):
        print("\nShutdown signal received...")
        stop_bridge()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Keep running
    print("\nBridge running. Press Ctrl+C to stop.")
    try:
        while bridge._running:
            time.sleep(10)
            stats = bridge.get_stats()
            print(f"Stats: processed={stats['events_processed']}, "
                  f"phi={stats['phi_coherence']:.3f}, "
                  f"coherent={stats['is_coherent']}")
    except KeyboardInterrupt:
        pass

    stop_bridge()


if __name__ == '__main__':
    main()
