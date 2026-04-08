import os
import json
import textwrap
from typing import List, Optional
from dotenv import load_dotenv

from openai import OpenAI
from redaction_env.client import RedactionEnv
from redaction_env.models import RedactionAction

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-key")
IMAGE_NAME = os.getenv("IMAGE_NAME") or "redaction_env_env:latest"

TASK_NAME = os.getenv("REDACTION_TASK", "redaction")
BENCHMARK = os.getenv("REDACTION_BENCHMARK", "redaction_env")
MAX_STEPS = 8
TEMPERATURE = 0.0
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

MAX_TOTAL_REWARD = 10.0  # Assumed max total reward

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a specialized PII redaction assistant.
    Given the tokens and context, determine purely which tokens to redact.
    Output ONLY a JSON array of booleans (true for redact, false for keep), equal in length to the tokens array.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, tokens: List[str], context_info: str, is_public_record: bool, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last reward: {last_reward:.2f}
        Previous steps history:
        {history_block}
        
        New Observation:
        Tokens: {json.dumps(tokens)}
        Context: {context_info}
        Public Record: {is_public_record}
        
        Send your boolean mask array to redact the required tokens.
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, tokens: List[str], context_info: str, is_public_record: bool, last_reward: float, history: List[str]) -> str:
    user_prompt = build_user_prompt(step, tokens, context_info, is_public_record, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if "```json" in text:
            text = text.replace("```json", "").replace("```", "").strip()
        elif "```" in text:
            text = text.replace("```", "").strip()
        return text if text else "[]"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "[]"


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    # Initialize the client synchronously (based on the provided client.py structure)
    try:
        env = RedactionEnv.from_docker_image(IMAGE_NAME)
    except Exception as e:
        print(f"Failed to load env: {e}. Falling back to default URL.")
        env = RedactionEnv(base_url="http://localhost:8000")

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset()
        obs = result.observation
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            tokens = getattr(obs, "tokens", [])
            context_info = getattr(obs, "context_info", "")
            is_public_record = getattr(obs, "is_public_record", False)

            message = get_model_message(client, step, tokens, context_info, is_public_record, last_reward, history)

            try:
                action_list = json.loads(message)
                if not isinstance(action_list, list):
                    action_list = []
            except BaseException:
                action_list = []
                
            try:
                # The model is expected to output a JSON array of booleans mask representation
                action_str = json.dumps(action_list, separators=(",", ":"))
                action = RedactionAction(redact_mask=action_list)
                result = env.step(action)
                obs = result.observation
                error = None
            except Exception as e:
                error = str(e)
                action_str = message
                # Fallback obs
                obs = result.observation

            reward = result.reward or 0.0
            done = result.done or False

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: Action: {action_str!r} -> Reward: {reward:+.2f}")

            if done:
                break

        # Max total reward is steps_taken * 1.0 (since each step is accuracy [0,1])
        score = sum(rewards) / float(steps_taken) if steps_taken > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
