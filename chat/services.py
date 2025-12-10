import subprocess
import os

def ask_ollama(prompt: str) -> str:
    # Force CPU mode to avoid CUDA errors
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    
    result = subprocess.run(
        ["ollama", "run", "llama3", prompt],
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace',
        timeout=180,
        env=env
    )
    
    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "Unknown error"
        raise Exception(f"Ollama error: {error_msg}")
    
    if result.stdout is None:
        raise Exception("No output from Ollama")
    
    return result.stdout.strip()
