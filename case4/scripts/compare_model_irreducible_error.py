#!/usr/bin/env python3
"""
Compare irreducible error between DeepSeek and GPT-4o-mini on the same text.

This helps understand if different models have similar uncertainty patterns
on text that should be in their training data.

Usage:
    python compare_model_irreducible_error.py

Environment variables:
    OPENROUTER_KEY: API key for OpenRouter
"""

import json
import math
import os
import subprocess
import sys

# The northern gannet Wikipedia text to use as context
GANNET_TEXT = """The wings of the northern gannet are long and narrow and are positioned towards the front of the body, allowing efficient use of air currents when flying. Even in calm weather they can attain velocities of between 55 and 65 km/h (30 and 35 kn) although their flying muscles are relatively small: in other birds, flying muscles make up around 20% of total weight, while in northern gannets the flying muscles are less than 13%. Despite their speed, they cannot manoeuvre in flight as well as other seabirds.[74] Northern gannets need to warm up before flying. They also walk with difficulty, and this means that they have problems getting airborne from a"""

MODELS = [
    ("deepseek/deepseek-chat-v3.1", {"require_parameters": True, "allow_fallbacks": False}),
    ("openai/gpt-4o-mini", None),  # No special provider config needed
]


def get_completion_with_logprobs(prompt, model, provider_config=None, max_tokens=1, top_logprobs=5):
    """Call OpenRouter API to get completion with log probabilities."""
    api_key = os.environ.get("OPENROUTER_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_KEY environment variable not set")
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"Complete this sentence with a single word: {prompt}"}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": top_logprobs,
    }
    
    if provider_config:
        payload["provider"] = provider_config
    
    cmd = [
        "curl", "-s", "https://openrouter.ai/api/v1/chat/completions",
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout)


def analyze_logprobs(response):
    """Extract and analyze log probabilities from API response."""
    if "error" in response:
        return {"error": response["error"]["message"]}
    
    choices = response.get("choices", [])
    if not choices:
        return {"error": "No choices in response"}
    
    logprobs_data = choices[0].get("logprobs")
    if not logprobs_data or not logprobs_data.get("content"):
        return {"error": "No logprobs in response (provider may not support them)"}
    
    content = logprobs_data["content"][0]
    top_token = content["token"]
    top_logprob = content["logprob"]
    top_prob = math.exp(top_logprob)
    
    # Calculate entropy from top_logprobs
    top_logprobs = content.get("top_logprobs", [])
    probs = [math.exp(t["logprob"]) for t in top_logprobs]
    
    # Entropy calculation (note: this is a lower bound since we only have top 5)
    entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probs)
    
    return {
        "predicted_token": top_token,
        "logprob": top_logprob,
        "probability": top_prob,
        "top_tokens": [(t["token"], math.exp(t["logprob"])) for t in top_logprobs],
        "entropy_lower_bound": entropy,
        "provider": response.get("provider", "unknown")
    }


def main():
    print("=" * 70)
    print("COMPARING IRREDUCIBLE ERROR ACROSS MODELS")
    print("=" * 70)
    print()
    print("Comparing how different models predict the next token on Wikipedia text")
    print("about Northern Gannets (likely in training data for both models).")
    print()
    print("Context ending: '...problems getting airborne from a'")
    print()
    
    for model, provider_config in MODELS:
        print("-" * 70)
        print(f"Model: {model}")
        print("-" * 70)
        
        response = get_completion_with_logprobs(GANNET_TEXT, model, provider_config)
        analysis = analyze_logprobs(response)
        
        if "error" in analysis:
            print(f"  ERROR: {analysis['error']}")
            continue
        
        print(f"  Provider: {analysis['provider']}")
        print(f"  Predicted next token: '{analysis['predicted_token']}'")
        print(f"  Probability: {analysis['probability']:.4f} ({analysis['probability']*100:.2f}%)")
        print(f"  Log probability: {analysis['logprob']:.4f}")
        print(f"  Entropy lower bound: {analysis['entropy_lower_bound']:.4f} bits")
        print()
        print("  Top 5 predictions:")
        for token, prob in analysis['top_tokens']:
            print(f"    '{token}': {prob:.6f} ({prob*100:.4f}%)")
        print()
    
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print("""
Key observations:
1. If both models predict the same token with similar probability, this
   suggests the uncertainty is intrinsic to the text (irreducible error).

2. If models differ significantly, some of the error might be model-specific
   (reducible with better training).

3. The Wikipedia text about Northern Gannets (from the species article) should
   be in the training data of both models, so we expect memorization effects.

4. Lower entropy = more confident = lower irreducible error for this position.
""")


if __name__ == "__main__":
    main()
