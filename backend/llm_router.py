from .groq_client import groq_generate
from .gemini_client import gemini_generate

async def llm_generate(prompt: str, task_type: str):
    """
    Route tasks to appropriate model
    """

    # 🔹 Cheap / fast tasks → Gemini
    if task_type in ["decompose", "planning", "classification"]:
        return await gemini_generate(prompt)

    # 🔹 Heavy tasks → Groq
    return await groq_generate(prompt)