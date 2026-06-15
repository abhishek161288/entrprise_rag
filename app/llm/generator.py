import os
import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    http_client=httpx.Client(verify=False)
)

def generate_answer(
        question,
        context
):

    prompt = f"""
You are an enterprise knowledge assistant.

Answer ONLY from context.

If information is unavailable,
say so.

Context:

{context}

Question:

{question}
"""

    response = client.responses.create(
        model=os.getenv("OPEN_AI_MODEL"),
        input=prompt
    )

    return response.output_text