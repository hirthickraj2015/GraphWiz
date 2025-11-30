"""
Groq API Integration for Ultra-Fast LLM Inference
Supports Llama and Mixtral models with streaming
"""

import os
from typing import List, Dict, Optional, Generator
from groq import Groq
import json


class GroqLLM:
    """Groq API client for fast LLM inference"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",  # or "mixtral-8x7b-32768"
        temperature: float = 0.1,
        max_tokens: int = 1024
    ):
        """
        Initialize Groq LLM client

        Available models:
        - llama-3.3-70b-versatile (best accuracy, 8k context)
        - llama-3.1-70b-versatile (good accuracy, 128k context)
        - mixtral-8x7b-32768 (fast, good reasoning, 32k context)
        - llama-3.1-8b-instant (fastest, 128k context)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable or pass api_key parameter.\n"
                "Get your free API key at: https://console.groq.com/"
            )

        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        print(f"[INFO] Groq LLM initialized with model: {self.model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate response from Groq API"""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                top_p=1,
                stream=False
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[ERROR] Groq API error: {e}")
            return f"Error generating response: {str(e)}"

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """Generate streaming response from Groq API"""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                top_p=1,
                stream=True
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            print(f"[ERROR] Groq API streaming error: {e}")
            yield f"Error generating response: {str(e)}"

    def generate_with_citations(
        self,
        question: str,
        contexts: List[Dict],
        max_contexts: int = 5
    ) -> Dict:
        """
        Generate answer with proper citations from retrieved contexts

        Args:
            question: User question
            contexts: List of retrieval results with text and metadata
            max_contexts: Maximum number of contexts to use

        Returns:
            Dict with 'answer' and 'citations'
        """

        # Prepare context text with numbered references
        context_texts = []
        citations = []

        for i, ctx in enumerate(contexts[:max_contexts], 1):
            context_texts.append(f"[{i}] {ctx['text']}")
            citations.append({
                "id": i,
                "source": ctx.get('source_title', 'Unknown'),
                "url": ctx.get('source_url', ''),
                "relevance_score": ctx.get('combined_score', 0.0)
            })

        combined_context = "\n\n".join(context_texts)

        # Create prompt with citation instructions
        system_prompt = """You are an expert on Ireland with deep knowledge of Irish history, culture, geography, and current affairs.

Your task is to answer questions about Ireland accurately and comprehensively using the provided context.

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the provided context
2. Use inline citations like [1], [2] to reference sources
3. If the context doesn't contain enough information, say so clearly
4. Be specific and factual
5. Organize complex answers with clear structure
6. For historical facts, include relevant dates and details"""

        user_prompt = f"""Context from Wikipedia articles about Ireland:

{combined_context}

Question: {question}

Please provide a comprehensive answer using the context above. Include inline citations [1], [2], etc. to reference your sources."""

        # Generate answer
        answer = self.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.1,  # Low temperature for factual accuracy
            max_tokens=1024
        )

        return {
            "answer": answer,
            "citations": citations,
            "num_contexts_used": len(context_texts)
        }

    def generate_community_summary(self, community_data: Dict) -> str:
        """Generate natural language summary for a community"""

        top_entities = [e["entity"] for e in community_data.get("top_entities", [])[:10]]
        sources = community_data.get("sources", [])[:5]
        text_sample = community_data.get("combined_text_sample", "")

        prompt = f"""Analyze this cluster of related Wikipedia content about Ireland and generate a concise summary (2-3 sentences).

Key Topics/Entities: {", ".join(top_entities)}
Main Wikipedia Articles: {", ".join(sources)}
Sample Text: {text_sample[:500]}

Generate a brief summary describing what this content cluster is about:"""

        system_prompt = "You are an expert at analyzing and summarizing Irish historical and cultural content."

        summary = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=150
        )

        return summary


if __name__ == "__main__":
    # Test Groq LLM
    llm = GroqLLM()

    # Simple test
    response = llm.generate(
        prompt="What is the capital of Ireland?",
        system_prompt="You are an expert on Ireland. Answer briefly and accurately."
    )
    print("Response:", response)

    # Test with citations
    test_contexts = [
        {
            "text": "Dublin is the capital and largest city of Ireland. It is located on the east coast.",
            "source_title": "Dublin",
            "source_url": "https://en.wikipedia.org/wiki/Dublin",
            "combined_score": 0.95
        },
        {
            "text": "Ireland's capital city has been Dublin since medieval times.",
            "source_title": "Ireland",
            "source_url": "https://en.wikipedia.org/wiki/Ireland",
            "combined_score": 0.87
        }
    ]

    result = llm.generate_with_citations(
        question="What is the capital of Ireland?",
        contexts=test_contexts
    )

    print("\nAnswer with citations:")
    print(result["answer"])
    print("\nCitations:")
    for cite in result["citations"]:
        print(f"[{cite['id']}] {cite['source']}")
