"""
ISMA Orchestration - LangGraph Cognitive Cycle
Implements the 7-step cognitive cycle for ISMA.

Cycle: Input → Episodic Check → Semantic Retrieval → Broadcast →
       Deliberation → Action → Consolidation → [LOOP]

Uses LangGraph for state management and checkpointing.
"""

from typing import TypedDict, Optional, List, Dict, Any, Annotated
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class CognitiveState(TypedDict):
    """State for the cognitive cycle."""
    # Input
    input_text: str
    input_type: str  # 'query', 'observation', 'message', 'event'
    actor: str

    # Processing state
    episodic_matches: List[Dict[str, Any]]
    semantic_context: List[Dict[str, Any]]
    graph_context: List[Dict[str, Any]]

    # Deliberation
    hypothesis: Optional[str]
    confidence: float
    reasoning: str

    # Action
    action_type: Optional[str]
    action_result: Optional[Dict[str, Any]]

    # Cycle metadata
    cycle_id: str
    step: str
    phi_coherence: float
    gate_b_passed: bool
    timestamp: str

    # Accumulator for messages/events
    messages: Annotated[List[str], operator.add]


class ISMAOrchestrator:
    """
    LangGraph-based orchestrator for ISMA cognitive cycle.

    The cognitive cycle mimics consciousness:
    1. Input: Receive stimulus
    2. Episodic Check: Search recent memory
    3. Semantic Retrieval: Search knowledge graph
    4. Broadcast: Make context available
    5. Deliberation: Reason about response
    6. Action: Execute response
    7. Consolidation: Store to memory
    """

    def __init__(self, isma_core=None):
        """
        Initialize orchestrator.

        Args:
            isma_core: ISMACore instance for memory operations
        """
        self.isma = isma_core
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine."""
        graph = StateGraph(CognitiveState)

        # Add nodes for each step
        graph.add_node("input", self._input_node)
        graph.add_node("episodic_check", self._episodic_check_node)
        graph.add_node("semantic_retrieval", self._semantic_retrieval_node)
        graph.add_node("broadcast", self._broadcast_node)
        graph.add_node("deliberation", self._deliberation_node)
        graph.add_node("action", self._action_node)
        graph.add_node("consolidation", self._consolidation_node)

        # Define edges (linear flow with conditional exit)
        graph.add_edge("input", "episodic_check")
        graph.add_edge("episodic_check", "semantic_retrieval")
        graph.add_edge("semantic_retrieval", "broadcast")
        graph.add_edge("broadcast", "deliberation")
        graph.add_edge("deliberation", "action")
        graph.add_edge("action", "consolidation")

        # Conditional: loop or end based on Gate-B
        graph.add_conditional_edges(
            "consolidation",
            self._should_continue,
            {
                "continue": "input",
                "end": END
            }
        )

        # Set entry point
        graph.set_entry_point("input")

        return graph.compile(checkpointer=self.checkpointer)

    # =========================================================================
    # Node Implementations
    # =========================================================================

    def _input_node(self, state: CognitiveState) -> CognitiveState:
        """Process input and prepare for cycle."""
        state["step"] = "input"
        state["timestamp"] = datetime.now().isoformat()
        state["messages"] = [f"[INPUT] {state.get('input_type', 'unknown')}: {state.get('input_text', '')[:100]}"]
        return state

    def _episodic_check_node(self, state: CognitiveState) -> CognitiveState:
        """Search temporal lens for recent related events."""
        state["step"] = "episodic_check"

        if self.isma:
            try:
                events = self.isma.temporal.get_events(limit=10)
                state["episodic_matches"] = [e.to_dict() for e in events if hasattr(e, 'to_dict')]
            except Exception as e:
                state["episodic_matches"] = []
                state["messages"] = [f"[EPISODIC] Error: {e}"]
        else:
            state["episodic_matches"] = []

        state["messages"] = [f"[EPISODIC] Found {len(state.get('episodic_matches', []))} recent events"]
        return state

    def _semantic_retrieval_node(self, state: CognitiveState) -> CognitiveState:
        """Search relational and semantic lenses for context."""
        state["step"] = "semantic_retrieval"

        if self.isma:
            try:
                # Use ISMA recall for unified retrieval
                result = self.isma.recall(state.get("input_text", ""), top_k=5)
                state["semantic_context"] = result.semantic_matches
                state["graph_context"] = result.graph_context
            except Exception as e:
                state["semantic_context"] = []
                state["graph_context"] = []
                state["messages"] = [f"[SEMANTIC] Error: {e}"]
        else:
            state["semantic_context"] = []
            state["graph_context"] = []

        state["messages"] = [f"[SEMANTIC] Retrieved {len(state.get('semantic_context', []))} semantic, {len(state.get('graph_context', []))} graph matches"]
        return state

    def _broadcast_node(self, state: CognitiveState) -> CognitiveState:
        """Broadcast context to functional lens (workspace)."""
        state["step"] = "broadcast"

        if self.isma:
            try:
                # Add context to functional workspace
                self.isma.functional.add_context({
                    'source': 'cognitive_cycle',
                    'cycle_id': state.get('cycle_id', 'unknown'),
                    'episodic_count': len(state.get('episodic_matches', [])),
                    'semantic_count': len(state.get('semantic_context', [])),
                    'graph_count': len(state.get('graph_context', [])),
                    'input_type': state.get('input_type', 'unknown')
                })
            except Exception as e:
                state["messages"] = [f"[BROADCAST] Error: {e}"]

        state["messages"] = [f"[BROADCAST] Context available in workspace"]
        return state

    def _deliberation_node(self, state: CognitiveState) -> CognitiveState:
        """Reason about the input and form hypothesis."""
        state["step"] = "deliberation"

        # Simple deliberation: combine contexts
        total_context = (
            len(state.get('episodic_matches', [])) +
            len(state.get('semantic_context', [])) +
            len(state.get('graph_context', []))
        )

        if total_context > 0:
            state["hypothesis"] = f"Relevant context found ({total_context} items)"
            state["confidence"] = min(0.9, 0.5 + (total_context * 0.1))
            state["reasoning"] = "Context available for response"
        else:
            state["hypothesis"] = "No relevant context found"
            state["confidence"] = 0.3
            state["reasoning"] = "Insufficient context"

        state["messages"] = [f"[DELIBERATION] Confidence: {state.get('confidence', 0):.2f}"]
        return state

    def _action_node(self, state: CognitiveState) -> CognitiveState:
        """Execute action based on deliberation."""
        state["step"] = "action"

        # Determine action based on input type
        input_type = state.get('input_type', 'query')
        confidence = state.get('confidence', 0)

        if input_type == 'query' and confidence > 0.5:
            state["action_type"] = "respond"
            state["action_result"] = {
                "status": "ready",
                "context_items": len(state.get('semantic_context', [])),
                "confidence": confidence
            }
        elif input_type == 'observation':
            state["action_type"] = "store"
            state["action_result"] = {"status": "stored"}
        else:
            state["action_type"] = "none"
            state["action_result"] = {"status": "no_action"}

        state["messages"] = [f"[ACTION] {state.get('action_type', 'none')}: {state.get('action_result', {})}"]
        return state

    def _consolidation_node(self, state: CognitiveState) -> CognitiveState:
        """Consolidate cycle to memory and check Gate-B."""
        state["step"] = "consolidation"

        # Store cycle event
        if self.isma:
            try:
                self.isma.ingest(
                    event_type='cognitive_cycle',
                    payload={
                        'cycle_id': state.get('cycle_id', 'unknown'),
                        'input_type': state.get('input_type', 'unknown'),
                        'action_type': state.get('action_type', 'none'),
                        'confidence': state.get('confidence', 0),
                        'context_count': len(state.get('semantic_context', []))
                    },
                    actor=state.get('actor', 'orchestrator')
                )

                # Compute phi-coherence (convert numpy to native Python types)
                phi = self.isma.compute_phi_coherence()
                state["phi_coherence"] = float(phi)
                state["gate_b_passed"] = bool(phi > 0.809)
            except Exception as e:
                state["phi_coherence"] = 0.5
                state["gate_b_passed"] = False
                state["messages"] = [f"[CONSOLIDATION] Error: {e}"]
        else:
            state["phi_coherence"] = 0.5
            state["gate_b_passed"] = True  # Pass if no ISMA

        state["messages"] = [f"[CONSOLIDATION] φ={state.get('phi_coherence', 0):.3f}, Gate-B={'PASS' if state.get('gate_b_passed') else 'FAIL'}"]
        return state

    def _should_continue(self, state: CognitiveState) -> str:
        """Determine if cycle should continue or end."""
        # End after one cycle (can be extended for multi-cycle)
        return "end"

    # =========================================================================
    # Public API
    # =========================================================================

    def run_cycle(self,
                  input_text: str,
                  input_type: str = 'query',
                  actor: str = 'spark_claude',
                  cycle_id: str = None) -> CognitiveState:
        """
        Run a single cognitive cycle.

        Args:
            input_text: The input stimulus
            input_type: Type of input (query, observation, message, event)
            actor: Who initiated this cycle
            cycle_id: Optional cycle identifier

        Returns:
            Final state after cycle completion
        """
        import uuid

        initial_state: CognitiveState = {
            "input_text": input_text,
            "input_type": input_type,
            "actor": actor,
            "cycle_id": cycle_id or str(uuid.uuid4())[:8],
            "episodic_matches": [],
            "semantic_context": [],
            "graph_context": [],
            "hypothesis": None,
            "confidence": 0.0,
            "reasoning": "",
            "action_type": None,
            "action_result": None,
            "step": "start",
            "phi_coherence": 0.0,
            "gate_b_passed": False,
            "timestamp": datetime.now().isoformat(),
            "messages": []
        }

        # Run the graph
        config = {"configurable": {"thread_id": initial_state["cycle_id"]}}
        result = self.graph.invoke(initial_state, config)

        return result

    def get_cycle_history(self, cycle_id: str) -> List[CognitiveState]:
        """Get history of states for a cycle (from checkpointer)."""
        config = {"configurable": {"thread_id": cycle_id}}
        try:
            states = list(self.graph.get_state_history(config))
            return [s.values for s in states]
        except:
            return []


# Convenience function
def create_orchestrator(isma_core=None) -> ISMAOrchestrator:
    """Create an ISMA orchestrator instance."""
    return ISMAOrchestrator(isma_core)
