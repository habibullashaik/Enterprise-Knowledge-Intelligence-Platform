from src.prompts import SYSTEM_PROMPT


def generate_answer(llm, query, context):

    prompt = SYSTEM_PROMPT + f"""

CONTEXT:

{context}


QUESTION:

{query}


ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content
