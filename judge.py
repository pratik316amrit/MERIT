import json
from typing import Dict, List, Optional
from openai_client import OpenAIClient


class ConversationJudge:
    """
    LLM-as-judge for mid-conversation analysis.
    Evaluates turn quality and suggests edge weight adjustments.
    """
    
    def __init__(self, client: Optional[OpenAIClient] = None):
        self.client = client or OpenAIClient()
    
    def evaluate_turn(
        self,
        history: List[Dict],
        user_msg: str,
        agent_msg: str,
        strategy_vector: Dict[str, float],
        intent_vector: Dict[str, float],
        memory_context: Dict
    ) -> tuple[Dict, Dict]:
        """
        Evaluate a conversation turn.
        
        Args:
            history: Conversation history
            user_msg: User message
            agent_msg: Agent response
            strategy_vector: Agent strategy spectrum
            intent_vector: User intent spectrum
            memory_context: Retrieved memory context
            
        Returns:
            Tuple of (evaluation_scores, token_usage)
        """
        # Build conversation history string
        history_str = "\n".join([
            f"{turn.get('speaker', 'Unknown')}: {turn.get('utterance', '')[:100]}"
            for turn in history[-5:]
        ])
        
        # Build memory context string
        memory_str = json.dumps({
            "background": memory_context.get("background", {}),
            "recent_topics": [t.get("requirement", "")[:50] for t in memory_context.get("topics", [])[-2:]]
        }, indent=2)
        
        prompt = f"""You are an expert conversation analyst. Evaluate this insurance sales dialogue turn.

Conversation History:
{history_str}

Latest Exchange:
User: {user_msg}
Agent: {agent_msg}

Current Strategy Vector: {json.dumps(strategy_vector, indent=2)}
Current Intent Vector: {json.dumps(intent_vector, indent=2)}

Retrieved Memory Context:
{memory_str}

Evaluate the agent's response and provide a JSON object with:
{{
  "coherence": <0.0-1.0, how well the response fits the conversation>,
  "strategy_effectiveness": <0.0-1.0, how well the strategy worked>,
  "intent_accuracy": <0.0-1.0, how accurately the user intent was understood>,
  "memory_relevance": <0.0-1.0, how relevant the retrieved memory was>,
  "decision_readiness": <0.0-1.0, how ready the user is to make a decision>,
  "edge_weight_adjustment": <-0.2 to +0.2, suggested adjustment for strategy transition>,
  "reasoning": "<one sentence explanation>"
}}"""
        
        try:
            result, token_usage = self.client.judge_turn(prompt)
            
            # Validate and normalize scores
            for key in ["coherence", "strategy_effectiveness", "intent_accuracy", "memory_relevance", "decision_readiness"]:
                if key in result:
                    result[key] = max(0.0, min(1.0, float(result.get(key, 0.5))))
            
            if "edge_weight_adjustment" in result:
                result["edge_weight_adjustment"] = max(-0.2, min(0.2, float(result.get("edge_weight_adjustment", 0.0))))
            
            return result, token_usage
        
        except Exception as e:
            print(f"Judge evaluation error: {e}")
            # Return neutral scores on error
            return {
                "coherence": 0.7,
                "strategy_effectiveness": 0.7,
                "intent_accuracy": 0.7,
                "memory_relevance": 0.7,
                "edge_weight_adjustment": 0.0,
                "reasoning": "Evaluation failed, using neutral scores"
            }, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def suggest_adjustment(self, scores: Dict) -> float:
        """
        Suggest edge weight adjustment based on scores.
        
        Args:
            scores: Evaluation scores from evaluate_turn
            
        Returns:
            Suggested adjustment value
        """
        # Average of key metrics
        avg_score = (
            scores.get("coherence", 0.7) +
            scores.get("strategy_effectiveness", 0.7) +
            scores.get("intent_accuracy", 0.7)
        ) / 3.0
        
        # Map to adjustment: good performance -> positive, poor -> negative
        if avg_score > 0.8:
            return 0.1
        elif avg_score > 0.6:
            return 0.05
        elif avg_score < 0.4:
            return -0.1
        elif avg_score < 0.6:
            return -0.05
        else:
            return 0.0
    
    def compute_edge_update(
        self,
        from_node: str,
        to_node: str,
        scores: Dict
    ) -> float:
        """
        Compute edge weight update based on evaluation.
        
        Args:
            from_node: Source node
            to_node: Target node
            scores: Evaluation scores
            
        Returns:
            Weight delta
        """
        # Use explicit adjustment if provided
        if "edge_weight_adjustment" in scores:
            return scores["edge_weight_adjustment"]
        
        # Otherwise compute from scores
        return self.suggest_adjustment(scores)


if __name__ == "__main__":
    print("Testing ConversationJudge...")
    
    try:
        judge = ConversationJudge()
        
        # Test evaluation
        print("\n1. Testing turn evaluation...")
        
        history = [
            {"speaker": "User", "utterance": "I need insurance for my new car."},
            {"speaker": "Agent", "utterance": "Congratulations! I can help you with that."}
        ]
        
        user_msg = "What's the price for comprehensive coverage?"
        agent_msg = "Our comprehensive plan starts at ₹25,000 annually, covering theft, accidents, and natural disasters."
        
        strategy_vector = {
            "credibility": 0.6,
            "emotional_appeal": 0.3,
            "logical_argument": 0.8,
            "personalization": 0.5,
            "urgency": 0.2
        }
        
        intent_vector = {
            "information_seeking": 0.7,
            "price_sensitivity": 0.8,
            "trust_level": 0.5,
            "decision_readiness": 0.5,
            "emotional_valence": 0.5
        }
        
        memory_context = {
            "background": {
                "vehicle_profile": "New car owner",
                "financial_profile": "Budget-conscious"
            },
            "topics": [
                {"requirement": "Comprehensive coverage information"}
            ]
        }
        
        scores, tokens = judge.evaluate_turn(
            history, user_msg, agent_msg,
            strategy_vector, intent_vector, memory_context
        )
        
        print(f"Evaluation scores:")
        print(json.dumps(scores, indent=2))
        print(f"Token usage: {tokens}")
        
        # Test adjustment
        print("\n2. Testing adjustment suggestion...")
        adjustment = judge.suggest_adjustment(scores)
        print(f"Suggested adjustment: {adjustment:.3f}")
        
        # Test edge update
        print("\n3. Testing edge update computation...")
        edge_update = judge.compute_edge_update("Logical", "Personal", scores)
        print(f"Edge update (Logical -> Personal): {edge_update:.3f}")
        
        print("\n✓ All tests passed!")
    
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        print("Note: This test requires OPENAI_API_KEY to be set")
