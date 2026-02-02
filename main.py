import os
import redis
from dotenv import load_dotenv
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.extensions.cache.embeddings import EmbeddingsCache
from redisvl.extensions.cache.llm import SemanticCache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

try:
    r = redis.Redis.from_url(REDIS_URL)
    r.ping()
    print("✅ Redis is running and accessible")
except redis.ConnectionError:
    print("❌ Cannot connect to Redis")
    raise



langcache_embed = HFTextVectorizer(
    model="redis/langcache-embed-v1",
    cache=EmbeddingsCache(redis_client=r, ttl=3600),
)


cache = SemanticCache(
    name="faq-cache",
    vectorizer=langcache_embed,
    redis_client=r,
    distance_threshold=0.3,
)

cache.set_ttl(86400)



MODEL_NAME = "gemini-2.5-flash"

llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=0.1,
    max_output_tokens=512,
)


def get_llm_response(question: str) -> str:
    prompt = f"""
You are a helpful customer support assistant. Answer this customer question concisely and professionally.

Question: {question}

Provide a helpful response in 1-2 sentences.
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()




def answer_question(question: str) -> str:
    cached = cache.check(question)

    if cached:
        hit = cached[0]
        print(
            f"✅ CACHE HIT "
            f"(distance={hit['vector_distance']:.3f})"
        )
        return hit["response"]

    print("❌ CACHE MISS — calling Gemini")
    response = get_llm_response(question)
    cache.store(prompt=question, response=response)
    return response


if __name__ == "__main__":
    questions = [
        "Who is Angel di Maria?",
        "Can you tell me about the person who scored the first goal in fifa world cup 2022 final?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {answer_question(q)}")
