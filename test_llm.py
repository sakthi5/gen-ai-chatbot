from app.llm import llm

response = llm.invoke("Explain Generative AI in one paragraph.")

print(response.content)