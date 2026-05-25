import json
import random
import os
from datetime import datetime
import numpy as np
from typing import Dict, List, Optional
from spectrum import IntentSpectrum, StrategySpectrum


class DenseGraph:
    """
    Dense weighted directed graph where every node connects to every other node.
    Edge weights sum to 1.0 for each source node.
    """
    
    def __init__(self, nodes: List[str], initial_weights: Dict[str, Dict[str, float]] = None):
        """
        Initialize dense graph.
        
        Args:
            nodes: List of node names
            initial_weights: Optional dict of {from_node: {to_node: weight}}
        """
        self.nodes = nodes
        self.update_counts = {f"{from_node}->{to_node}": 0 
                             for from_node in nodes for to_node in nodes}
        
        if initial_weights:
            self.weights = initial_weights
            self._normalize_weights()
        else:
            # Initialize with uniform weights
            self.weights = {}
            for from_node in nodes:
                self.weights[from_node] = {
                    to_node: 1.0 / len(nodes) for to_node in nodes
                }
    
    def _normalize_weights(self):
        """Ensure weights sum to 1.0 for each source node."""
        for from_node in self.nodes:
            total = sum(self.weights[from_node].values())
            if total > 0:
                for to_node in self.nodes:
                    self.weights[from_node][to_node] /= total
    
    def sample_next(
        self, 
        current_node: str, 
        spectrum_current: Optional[object] = None,
        spectrum_candidates: Optional[Dict[str, object]] = None,
        exclude_nodes: Optional[List[str]] = None
    ) -> str:
        """
        Sample next node using edge weights and optional spectrum similarity.
        
        Args:
            current_node: Current node
            spectrum_current: Current spectrum (IntentSpectrum or StrategySpectrum)
            spectrum_candidates: Dict of {node_name: spectrum} for candidates
            exclude_nodes: Nodes to exclude from sampling
            
        Returns:
            Next node name
        """
        if current_node not in self.weights:
            return random.choice(self.nodes)
        
        # Get base weights
        base_weights = self.weights[current_node].copy()
        
        # Exclude nodes if specified
        if exclude_nodes:
            for node in exclude_nodes:
                base_weights[node] = 0.0
        
        # Apply spectrum similarity if provided
        if spectrum_current and spectrum_candidates:
            for node in base_weights:
                if node in spectrum_candidates:
                    similarity = spectrum_current.similarity(spectrum_candidates[node])
                    # Combine edge weight with spectrum similarity
                    base_weights[node] *= (1.0 + similarity)
        
        # Normalize
        total = sum(base_weights.values())
        if total == 0:
            # Fallback to uniform
            return random.choice([n for n in self.nodes if n not in (exclude_nodes or [])])
        
        probs = {node: weight / total for node, weight in base_weights.items()}
        
        # Sample
        nodes_list = list(probs.keys())
        probs_list = [probs[node] for node in nodes_list]
        
        return np.random.choice(nodes_list, p=probs_list)
    
    def update_weight(
        self, 
        from_node: str, 
        to_node: str, 
        adjustment: float,
        learning_rate: float = 0.05
    ):
        """
        Update edge weight based on judge feedback.
        
        Args:
            from_node: Source node
            to_node: Target node
            adjustment: Adjustment value (typically -0.2 to +0.2)
            learning_rate: Learning rate for updates
        """
        if from_node not in self.weights or to_node not in self.weights[from_node]:
            return
        
        # Apply adjustment with learning rate
        delta = learning_rate * adjustment
        self.weights[from_node][to_node] = max(0.01, self.weights[from_node][to_node] + delta)
        
        # Normalize
        self._normalize_weights()
        
        # Track update
        key = f"{from_node}->{to_node}"
        self.update_counts[key] += 1
    
    def get_transition_probability(self, from_node: str, to_node: str) -> float:
        """Get edge weight between two nodes."""
        if from_node not in self.weights or to_node not in self.weights[from_node]:
            return 0.0
        return self.weights[from_node][to_node]
    
    def save_weights(self, filepath: str):
        """Save weights to JSON file."""
        data = {
            "weights": self.weights,
            "update_counts": self.update_counts,
            "nodes": self.nodes
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load_weights(self, filepath: str):
        """Load weights from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.weights = data["weights"]
        self.update_counts = data.get("update_counts", {})
        self.nodes = data.get("nodes", self.nodes)
    
    def save_global_weights(self, filepath: str):
        """
        Save cumulative global weights across all conversations.
        Used for long-horizon learning (2015-2025).
        
        Args:
            filepath: Path to save global weights (e.g., memory/global_intent_graph.json)
        """
        data = {
            "weights": self.weights,
            "update_counts": self.update_counts,
            "nodes": self.nodes,
            "metadata": {
                "total_updates": sum(self.update_counts.values()),
                "last_updated": str(datetime.now())
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def load_global_weights(self, filepath: str) -> bool:
        """
        Load cumulative global weights from previous runs.
        
        Args:
            filepath: Path to global weights file
            
        Returns:
            True if loaded successfully, False if file doesn't exist
        """
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.weights = data["weights"]
        self.update_counts = data.get("update_counts", {})
        self.nodes = data.get("nodes", self.nodes)
        
        return True
    
    def merge_weights(self, other_graph: 'DenseGraph', alpha: float = 0.1):
        """
        Merge weights from another graph for incremental learning.
        
        Args:
            other_graph: Another DenseGraph to merge from
            alpha: Learning rate for merging (0.0 = keep current, 1.0 = use other)
        """
        for from_node in other_graph.weights:
            if from_node not in self.weights:
                continue
            
            for to_node, other_weight in other_graph.weights[from_node].items():
                if to_node not in self.weights[from_node]:
                    continue
                
                # Weighted average
                current_weight = self.weights[from_node][to_node]
                self.weights[from_node][to_node] = (
                    (1 - alpha) * current_weight + alpha * other_weight
                )
        
        # Normalize after merging
        self._normalize_weights()
        
        # Merge update counts
        for key, count in other_graph.update_counts.items():
            self.update_counts[key] = self.update_counts.get(key, 0) + count

    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "weights": self.weights,
            "update_counts": self.update_counts
        }


def create_intent_graph() -> DenseGraph:
    """
    Create intent graph with expert-defined initial weights.
    Based on natural conversation flow from test10111.py.
    """
    nodes = [
        "Initial",
        "Ask Coverage Details",
        "Request quote",
        "Express Concern",
        "Request additional info",
        "Negotiate price",
        "Confirm Plan",
        "Reject Offer"
    ]
    
    # Expert-defined weights based on conversation flow
    initial_weights = {
        "Initial": {
            "Ask Coverage Details": 0.40,
            "Request quote": 0.30,
            "Request additional info": 0.20,
            "Express Concern": 0.05,
            "Negotiate price": 0.03,
            "Confirm Plan": 0.01,
            "Reject Offer": 0.01,
            "Initial": 0.00
        },
        "Ask Coverage Details": {
            "Express Concern": 0.30,
            "Request quote": 0.25,
            "Request additional info": 0.25,
            "Negotiate price": 0.10,
            "Initial": 0.05,
            "Confirm Plan": 0.03,
            "Reject Offer": 0.02,
            "Ask Coverage Details": 0.00
        },
        "Request quote": {
            "Negotiate price": 0.35,
            "Ask Coverage Details": 0.25,
            "Express Concern": 0.20,
            "Request additional info": 0.10,
            "Confirm Plan": 0.05,
            "Reject Offer": 0.03,
            "Initial": 0.02,
            "Request quote": 0.00
        },
        "Express Concern": {
            "Request additional info": 0.35,
            "Ask Coverage Details": 0.25,
            "Negotiate price": 0.20,
            "Request quote": 0.10,
            "Reject Offer": 0.05,
            "Initial": 0.03,
            "Confirm Plan": 0.02,
            "Express Concern": 0.00
        },
        "Request additional info": {
            "Request quote": 0.30,
            "Ask Coverage Details": 0.25,
            "Express Concern": 0.20,
            "Negotiate price": 0.15,
            "Confirm Plan": 0.05,
            "Initial": 0.03,
            "Reject Offer": 0.02,
            "Request additional info": 0.00
        },
        "Negotiate price": {
            "Confirm Plan": 0.40,
            "Express Concern": 0.25,
            "Request additional info": 0.15,
            "Reject Offer": 0.10,
            "Request quote": 0.05,
            "Ask Coverage Details": 0.03,
            "Initial": 0.02,
            "Negotiate price": 0.00
        },
        "Confirm Plan": {
            "Confirm Plan": 0.70,
            "Request additional info": 0.15,
            "Ask Coverage Details": 0.05,
            "Express Concern": 0.03,
            "Negotiate price": 0.03,
            "Request quote": 0.02,
            "Reject Offer": 0.01,
            "Initial": 0.01
        },
        "Reject Offer": {
            "Reject Offer": 0.70,
            "Request additional info": 0.10,
            "Express Concern": 0.08,
            "Ask Coverage Details": 0.05,
            "Request quote": 0.03,
            "Negotiate price": 0.02,
            "Confirm Plan": 0.01,
            "Initial": 0.01
        }
    }
    
    return DenseGraph(nodes, initial_weights)


def create_strategy_graph() -> DenseGraph:
    """
    Create strategy graph with expert-defined initial weights.
    """
    nodes = ["Default", "Credibility", "Emotional", "Logical", "Personal", "Persona"]
    
    initial_weights = {
        "Default": {
            "Credibility": 0.25,
            "Logical": 0.25,
            "Personal": 0.25,
            "Emotional": 0.15,
            "Persona": 0.05,
            "Default": 0.05
        },
        "Credibility": {
            "Logical": 0.30,
            "Personal": 0.25,
            "Emotional": 0.20,
            "Default": 0.15,
            "Persona": 0.05,
            "Credibility": 0.05
        },
        "Emotional": {
            "Personal": 0.30,
            "Logical": 0.25,
            "Credibility": 0.20,
            "Default": 0.15,
            "Persona": 0.05,
            "Emotional": 0.05
        },
        "Logical": {
            "Personal": 0.30,
            "Credibility": 0.25,
            "Emotional": 0.20,
            "Default": 0.15,
            "Persona": 0.05,
            "Logical": 0.05
        },
        "Personal": {
            "Emotional": 0.30,
            "Logical": 0.25,
            "Credibility": 0.20,
            "Default": 0.15,
            "Persona": 0.05,
            "Personal": 0.05
        },
        "Persona": {
            "Credibility": 0.30,
            "Personal": 0.25,
            "Emotional": 0.20,
            "Logical": 0.15,
            "Default": 0.05,
            "Persona": 0.05
        }
    }
    
    return DenseGraph(nodes, initial_weights)


if __name__ == "__main__":
    print("Testing DenseGraph...")
    
    # Test intent graph
    print("\n1. Testing Intent Graph...")
    intent_graph = create_intent_graph()
    
    print(f"Nodes: {intent_graph.nodes}")
    print(f"\nTransition from 'Initial':")
    for node in intent_graph.nodes:
        prob = intent_graph.get_transition_probability("Initial", node)
        if prob > 0.05:
            print(f"  -> {node}: {prob:.3f}")
    
    # Test sampling
    print("\n2. Testing sampling (10 transitions from 'Initial'):")
    samples = []
    for _ in range(10):
        next_node = intent_graph.sample_next("Initial")
        samples.append(next_node)
    
    from collections import Counter
    counts = Counter(samples)
    for node, count in counts.most_common():
        print(f"  {node}: {count}/10")
    
    # Test weight update
    print("\n3. Testing weight update...")
    original_weight = intent_graph.get_transition_probability("Initial", "Request quote")
    print(f"Original weight (Initial -> Request quote): {original_weight:.3f}")
    
    intent_graph.update_weight("Initial", "Request quote", adjustment=0.15)
    new_weight = intent_graph.get_transition_probability("Initial", "Request quote")
    print(f"Updated weight (Initial -> Request quote): {new_weight:.3f}")
    
    # Test save/load
    print("\n4. Testing save/load...")
    intent_graph.save_weights("test_graph_weights.json")
    print("✓ Saved to test_graph_weights.json")
    
    # Test strategy graph
    print("\n5. Testing Strategy Graph...")
    strategy_graph = create_strategy_graph()
    print(f"Nodes: {strategy_graph.nodes}")
    
    print("\n✓ All tests passed!")
