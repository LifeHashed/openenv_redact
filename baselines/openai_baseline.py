import os
import json
from openai import OpenAI
from redaction_env.graders import GraderDataset
from redaction_env.reward import calculate_redaction_reward
from redaction_env.models import RedactionAction

def run_baseline():
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy-key"))
    dataset = GraderDataset()
    
    # We will simulate the openai gpt-4o completion
    for difficulty in ["easy", "medium", "hard"]:
        print(f"--- Running tasks for {difficulty} difficulty ---")
        # In this script, we'll evaluate a sample from the given difficulty tier
        for i in range(2):
            sample = dataset.get_sample(difficulty)
            
            print(f"[START] {json.dumps({'difficulty': difficulty, 'task_id': i})}")
            print(f"[STEP] {json.dumps({'tokens': sample.tokens, 'context': sample.context_info})}")
            
            prompt = f"""
Given the following tokens and context, determine purely which tokens to redact.
Tokens: {json.dumps(sample.tokens)}
Context: {sample.context_info}
Public Record: {sample.is_public_record}

Output ONLY a JSON array of booleans (true for redact, false for keep), equal in length to the tokens array.
"""
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a specialized PII redaction assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                
                # Parse output to list of booleans
                response_text = response.choices[0].message.content.strip()
                # Try to parse string to get JSON
                if "```json" in response_text:
                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                elif "```" in response_text:
                    response_text = response_text.replace("```", "").strip()
                    
                action = RedactionAction(redact_mask=response_text)
                predicted_mask = action.redact_mask
                
            except Exception as e:
                print(f"Error calling OpenAI API or parsing result: {e}")
                # Fallback zero mask
                predicted_mask = [False] * len(sample.tokens)
                
            reward = calculate_redaction_reward(predicted_mask, sample.ground_truth_mask)
            
            print(f"[END] {json.dumps({'predicted_mask': predicted_mask, 'reward': reward})}")

if __name__ == "__main__":
    run_baseline()
