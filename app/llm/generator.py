import os
import httpx
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
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
        model="gpt-5.4",
        input=prompt
    )

    return response.output_text