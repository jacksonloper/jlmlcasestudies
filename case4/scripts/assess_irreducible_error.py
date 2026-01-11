#!/usr/bin/env python3
"""
Assess how DeepSeek handles irreducible error by examining log probabilities
on text that should be in its training data.

This script uses the OpenRouter API to get log probabilities for next-token
predictions on Wikipedia text about northern gannets.

Usage:
    python assess_irreducible_error.py

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

# Different truncation points to test
TEST_PREFIXES = [
    # Original ending from the problem statement
    ("...problems getting airborne from a", GANNET_TEXT),
    # Earlier truncation points
    ("...warm up before flying. They also", 
     "The wings of the northern gannet are long and narrow and are positioned towards the front of the body, allowing efficient use of air currents when flying. Even in calm weather they can attain velocities of between 55 and 65 km/h (30 and 35 kn) although their flying muscles are relatively small: in other birds, flying muscles make up around 20% of total weight, while in northern gannets the flying muscles are less than 13%. Despite their speed, they cannot manoeuvre in flight as well as other seabirds.[74] Northern gannets need to warm up before flying. They also"),
    ("...less than 13%. Despite their",
     "The wings of the northern gannet are long and narrow and are positioned towards the front of the body, allowing efficient use of air currents when flying. Even in calm weather they can attain velocities of between 55 and 65 km/h (30 and 35 kn) although their flying muscles are relatively small: in other birds, flying muscles make up around 20% of total weight, while in northern gannets the flying muscles are less than 13%. Despite their"),
]


def get_completion_with_logprobs(prompt, model="deepseek/deepseek-chat-v3.1", max_tokens=1, top_logprobs=5):
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
        "provider": {
            "require_parameters": True,
            "allow_fallbacks": False
        }
    }
    
    cmd = [
        "curl", "-s", "https://openrouter.ai/api/v1/chat/completions",
        "-H", f"Authorization: Bearer {api_key}",
        "-H", "Content-Type: application/json",
        "-d", json.dumps(payload)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"error": {"message": f"curl failed: {result.stderr}"}}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return {"error": {"message": f"JSON parse error: {e}"}}


def analyze_logprobs(response):
    """Extract and analyze log probabilities from API response."""
    if "error" in response:
        return {"error": response["error"]["message"]}
    
    choices = response.get("choices", [])
    if not choices:
        return {"error": "No choices in response"}
    
    logprobs_data = choices[0].get("logprobs")
    if not logprobs_data or not logprobs_data.get("content"):
        return {"error": "No logprobs in response"}
    
    content = logprobs_data["content"][0]
    top_token = content["token"]
    top_logprob = content["logprob"]
    top_prob = math.exp(top_logprob)
    
    # Calculate entropy from top_logprobs
    top_logprobs = content.get("top_logprobs", [])
    probs = [math.exp(t["logprob"]) for t in top_logprobs]
    
    # Entropy calculation (note: this is a lower bound since we only have top tokens)
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
    print("ASSESSING DEEPSEEK IRREDUCIBLE ERROR ON TRAINING DATA")
    print("=" * 70)
    print()
    print("Using DeepSeek V3.1 via OpenRouter to analyze next-token predictions")
    print("on Wikipedia text about Northern Gannets that should be in training data.")
    print()
    
    for desc, prefix in TEST_PREFIXES:
        print("-" * 70)
        print(f"Test: {desc}")
        print("-" * 70)
        
        response = get_completion_with_logprobs(prefix)
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
            print(f"    '{token}': {prob:.4f} ({prob*100:.2f}%)")
        print()
    
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("""
Irreducible error refers to the inherent uncertainty in predicting the next
token that cannot be reduced even with perfect knowledge of the training data.

For text that is verbatim from Wikipedia (likely in the training data):
- High probability (>90%) on the correct token suggests low irreducible error
- The entropy of the distribution measures the uncertainty

When the top token has probability ~99%, this suggests:
1. The model has memorized this text very well
2. The next word is highly predictable from context
3. Irreducible error is low for this continuation

Low entropy (close to 0) = highly predictable = low irreducible error
High entropy = many plausible continuations = higher irreducible error
""")


if __name__ == "__main__":
    main()
