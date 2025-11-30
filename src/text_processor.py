"""
Advanced Text Chunking and Preprocessing Pipeline
Intelligently chunks Wikipedia articles while preserving context and semantic coherence.
"""

import re
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import spacy
from tqdm import tqdm


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    chunk_id: str
    text: str
    source_title: str
    source_url: str
    section: str
    chunk_index: int
    total_chunks: int
    char_start: int
    char_end: int
    word_count: int
    has_entities: bool = False
    entities: List[Dict] = None


class AdvancedTextProcessor:
    """Advanced text processing with intelligent chunking"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128, spacy_model: str = "en_core_web_sm"):
        self.chunk_size = chunk_size  # tokens
        self.chunk_overlap = chunk_overlap  # tokens

        # Load spaCy model for sentence segmentation and entity recognition
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"[INFO] Downloading spaCy model: {spacy_model}")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)

        # Disable unnecessary components for speed
        self.nlp.select_pipes(enable=["tok2vec", "tagger", "parser", "ner"])

    def clean_text(self, text: str) -> str:
        """Clean Wikipedia text"""
        if not text:
            return ""

        # Remove Wikipedia markup
        text = re.sub(r'\{\{[^}]+\}\}', '', text)  # Remove templates
        text = re.sub(r'\[\[File:[^\]]+\]\]', '', text)  # Remove file links
        text = re.sub(r'\[\[Image:[^\]]+\]\]', '', text)  # Remove image links

        # Clean internal links but keep text
        text = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', text)  # [[Link|Text]] -> Text
        text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # [[Link]] -> Link

        # Remove external links
        text = re.sub(r'\[http[s]?://[^\]]+\]', '', text)

        # Remove citations
        text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
        text = re.sub(r'<ref[^>]*/?>', '', text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def chunk_by_sentences(self, text: str, source_title: str, source_url: str, section: str = "main") -> List[TextChunk]:
        """Chunk text by sentences with overlap"""
        if not text:
            return []

        # Clean text first
        text = self.clean_text(text)

        # Process with spaCy
        doc = self.nlp(text)
        sentences = list(doc.sents)

        if not sentences:
            return []

        chunks = []
        current_chunk_tokens = []
        current_chunk_start = 0
        chunk_index = 0

        for sent_idx, sent in enumerate(sentences):
            sent_tokens = [token.text for token in sent]

            # If adding this sentence exceeds chunk size, save current chunk
            if len(current_chunk_tokens) + len(sent_tokens) > self.chunk_size and current_chunk_tokens:
                # Create chunk
                chunk_text = " ".join(current_chunk_tokens)
                chunk = TextChunk(
                    chunk_id=f"{source_title.replace(' ', '_')}_{chunk_index}",
                    text=chunk_text,
                    source_title=source_title,
                    source_url=source_url,
                    section=section,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will update later
                    char_start=current_chunk_start,
                    char_end=current_chunk_start + len(chunk_text),
                    word_count=len(current_chunk_tokens)
                )
                chunks.append(chunk)
                chunk_index += 1

                # Create overlap by keeping last N tokens
                overlap_tokens = current_chunk_tokens[-self.chunk_overlap:] if len(current_chunk_tokens) > self.chunk_overlap else []
                current_chunk_tokens = overlap_tokens + sent_tokens
                current_chunk_start = current_chunk_start + len(chunk_text) - len(" ".join(overlap_tokens))
            else:
                current_chunk_tokens.extend(sent_tokens)

        # Add final chunk
        if current_chunk_tokens:
            chunk_text = " ".join(current_chunk_tokens)
            chunk = TextChunk(
                chunk_id=f"{source_title.replace(' ', '_')}_{chunk_index}",
                text=chunk_text,
                source_title=source_title,
                source_url=source_url,
                section=section,
                chunk_index=chunk_index,
                total_chunks=0,
                char_start=current_chunk_start,
                char_end=current_chunk_start + len(chunk_text),
                word_count=len(current_chunk_tokens)
            )
            chunks.append(chunk)

        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def extract_entities(self, chunk: TextChunk) -> TextChunk:
        """Extract named entities from chunk"""
        doc = self.nlp(chunk.text)
        entities = []

        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        chunk.has_entities = len(entities) > 0
        chunk.entities = entities
        return chunk

    def process_article(self, article: Dict) -> List[TextChunk]:
        """Process a single article into chunks"""
        chunks = []

        # Process main summary
        if article.get("summary"):
            summary_chunks = self.chunk_by_sentences(
                article["summary"],
                article["title"],
                article["url"],
                section="summary"
            )
            chunks.extend(summary_chunks)

        # Process full text (skip summary part to avoid duplication)
        if article.get("full_text"):
            full_text = article["full_text"]
            # Remove summary from full text if it's at the beginning
            if article.get("summary") and full_text.startswith(article["summary"][:100]):
                full_text = full_text[len(article["summary"]):]

            main_chunks = self.chunk_by_sentences(
                full_text,
                article["title"],
                article["url"],
                section="full_article"
            )
            chunks.extend(main_chunks)

        # Extract entities for all chunks
        chunks = [self.extract_entities(chunk) for chunk in chunks]

        return chunks

    def process_all_articles(self, articles: List[Dict]) -> List[Dict]:
        """Process all articles into chunks"""
        print(f"[INFO] Processing {len(articles)} articles into chunks...")
        all_chunks = []

        for article in tqdm(articles, desc="Processing articles"):
            chunks = self.process_article(article)
            all_chunks.extend(chunks)

        print(f"[SUCCESS] Created {len(all_chunks)} chunks from {len(articles)} articles")

        # Convert to dict for JSON serialization
        chunks_dict = []
        for chunk in all_chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source_title": chunk.source_title,
                "source_url": chunk.source_url,
                "section": chunk.section,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "word_count": chunk.word_count,
                "has_entities": chunk.has_entities,
                "entities": chunk.entities if chunk.entities else []
            }
            chunks_dict.append(chunk_dict)

        return chunks_dict

    def save_chunks(self, chunks: List[Dict], output_path: str = "dataset/wikipedia_ireland/chunks.json"):
        """Save chunks to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        # Save statistics
        stats = {
            "total_chunks": len(chunks),
            "avg_chunk_length": sum(c["word_count"] for c in chunks) / len(chunks),
            "chunks_with_entities": sum(1 for c in chunks if c["has_entities"]),
            "total_entities": sum(len(c["entities"]) for c in chunks)
        }

        stats_path = output_path.replace("chunks.json", "chunk_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"[SUCCESS] Saved {len(chunks)} chunks to {output_path}")
        print(f"[INFO] Statistics saved to {stats_path}")

        return output_path


if __name__ == "__main__":
    # Test with sample articles
    with open("dataset/wikipedia_ireland/ireland_articles.json", 'r') as f:
        articles = json.load(f)

    processor = AdvancedTextProcessor(chunk_size=512, chunk_overlap=128)
    chunks = processor.process_all_articles(articles)
    processor.save_chunks(chunks)
