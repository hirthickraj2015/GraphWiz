"""
Comprehensive Wikipedia Ireland Data Extractor
Extracts ALL Ireland-related Wikipedia articles with full content, metadata, and links.
"""

import wikipediaapi
import time
import json
import re
from typing import List, Dict, Set
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


class IrelandWikipediaExtractor:
    """Extract comprehensive Ireland-related Wikipedia content"""

    def __init__(self, output_dir="dataset/wikipedia_ireland"):
        self.wiki = wikipediaapi.Wikipedia(
            user_agent='IrelandKG/1.0 (educational research project)',
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            timeout=60  # Increase timeout to 60 seconds
        )
        self.output_dir = output_dir
        self.ireland_categories = [
            "Category:Ireland",
            "Category:History of Ireland",
            "Category:Geography of Ireland",
            "Category:Culture of Ireland",
            "Category:Politics of Ireland",
            "Category:Economy of Ireland",
            "Category:Education in Ireland",
            "Category:Irish people",
            "Category:Irish language",
            "Category:Counties of Ireland",
            "Category:Cities and towns in Ireland",
            "Category:Buildings and structures in Ireland",
            "Category:Sport in Ireland",
            "Category:Irish literature",
            "Category:Irish music",
            "Category:Irish mythology",
            "Category:Religion in Ireland",
            "Category:Transport in Ireland",
            "Category:Science and technology in Ireland",
            "Category:Environment of Ireland",
            "Category:Northern Ireland",
            "Category:Republic of Ireland"
        ]

    def get_category_members(self, category_name: str, depth: int = 2, retries: int = 3) -> Set[str]:
        """Recursively get all pages in a category and its subcategories"""
        print(f"[INFO] Fetching category: {category_name} (depth={depth})")
        pages = set()

        for attempt in range(retries):
            try:
                cat = self.wiki.page(category_name)
                if not cat.exists():
                    print(f"[WARNING] Category not found: {category_name}")
                    return pages
                break
            except Exception as e:
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                    print(f"[RETRY] Attempt {attempt + 1} failed: {str(e)[:100]}")
                    print(f"[RETRY] Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"[ERROR] Failed after {retries} attempts: {e}")
                    print(f"[ERROR] Skipping category: {category_name}")
                    return pages

        # Add all pages in this category
        for page_title in cat.categorymembers.keys():
            member = cat.categorymembers[page_title]
            if member.ns == wikipediaapi.Namespace.MAIN:  # Article namespace
                pages.add(page_title)
            elif member.ns == wikipediaapi.Namespace.CATEGORY and depth > 0:
                # Recursively get subcategory members with rate limiting
                time.sleep(1)  # Wait 1 second between subcategory requests
                subcategory_pages = self.get_category_members(page_title, depth - 1)
                pages.update(subcategory_pages)

        return pages

    def get_all_ireland_pages(self) -> List[str]:
        """Get ALL Ireland-related Wikipedia page titles"""
        print("[INFO] Collecting all Ireland-related Wikipedia pages...")
        all_pages = set()

        # Get pages from all Ireland categories
        for idx, category in enumerate(self.ireland_categories, 1):
            print(f"[INFO] Processing category {idx}/{len(self.ireland_categories)}: {category}")
            pages = self.get_category_members(category, depth=2)
            all_pages.update(pages)
            print(f"[INFO] Found {len(pages)} pages. Total unique: {len(all_pages)}")
            time.sleep(2)  # Increased rate limiting to 2 seconds

        # Add core Ireland articles that might be missed
        core_pages = [
            "Ireland",
            "Republic of Ireland",
            "Northern Ireland",
            "Dublin",
            "Belfast",
            "Irish language",
            "History of Ireland",
            "Politics of Ireland",
            "Economy of Ireland"
        ]
        all_pages.update(core_pages)

        print(f"[SUCCESS] Total unique pages found: {len(all_pages)}")
        return sorted(list(all_pages))

    def extract_article_content(self, page_title: str, retries: int = 3) -> Dict:
        """Extract full article content with metadata"""
        for attempt in range(retries):
            try:
                page = self.wiki.page(page_title)

                if not page.exists():
                    return None
                break
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                else:
                    print(f"[ERROR] Failed to fetch {page_title}: {e}")
                    return None

        try:

            # Extract links to other Wikipedia articles
            links = [link for link in page.links.keys() if not link.startswith("Category:")]

            # Extract categories
            categories = [cat for cat in page.categories.keys()]

            # Extract sections
            sections = self._extract_sections(page)

            return {
                "title": page.title,
                "url": page.fullurl,
                "summary": page.summary[:1000] if page.summary else "",
                "full_text": page.text,
                "text_length": len(page.text),
                "links": links[:100],  # Limit to avoid huge files
                "categories": categories,
                "sections": sections,
                "backlinks_count": 0,  # Will populate later if needed
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception as e:
            print(f"[ERROR] Failed to extract {page_title}: {e}")
            return None

    def _extract_sections(self, page) -> List[Dict]:
        """Extract section structure from Wikipedia page"""
        sections = []

        def traverse_sections(section_list, level=1):
            for section in section_list:
                sections.append({
                    "title": section.title,
                    "level": level,
                    "text_length": len(section.text)
                })
                if hasattr(section, 'sections'):
                    traverse_sections(section.sections, level + 1)

        if hasattr(page, 'sections'):
            traverse_sections(page.sections)

        return sections

    def extract_all_articles(self, page_titles: List[str], max_workers: int = 5, checkpoint_every: int = 100):
        """Extract all articles in parallel with checkpointing"""
        import os

        checkpoint_file = f"{self.output_dir}/checkpoint_articles.json"
        progress_file = f"{self.output_dir}/extraction_progress.json"

        # Load existing articles if checkpoint exists
        articles = []
        extracted_titles = set()
        start_index = 0

        if os.path.exists(checkpoint_file):
            print(f"[RESUME] Found checkpoint file, loading...")
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            extracted_titles = {a['title'] for a in articles}
            start_index = len(articles)
            print(f"[RESUME] Resuming from {start_index}/{len(page_titles)} articles")

        # Filter out already extracted articles
        remaining_titles = [t for t in page_titles if t not in extracted_titles]

        if not remaining_titles:
            print(f"[INFO] All {len(page_titles)} articles already extracted!")
            return articles

        print(f"[INFO] Extracting {len(remaining_titles)} remaining articles...")
        print(f"[INFO] Using {max_workers} parallel workers")
        print(f"[INFO] Checkpointing every {checkpoint_every} articles")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.extract_article_content, title): title
                      for title in remaining_titles}

            with tqdm(total=len(remaining_titles), desc="Extracting articles", initial=0) as pbar:
                batch_count = 0
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        articles.append(result)
                        batch_count += 1

                        # Checkpoint every N articles
                        if batch_count % checkpoint_every == 0:
                            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                                json.dump(articles, f, ensure_ascii=False, indent=2)
                            with open(progress_file, 'w') as f:
                                json.dump({
                                    'total': len(page_titles),
                                    'completed': len(articles),
                                    'remaining': len(page_titles) - len(articles)
                                }, f)
                            print(f"\n[CHECKPOINT] Saved progress: {len(articles)}/{len(page_titles)} articles")

                    pbar.update(1)

        # Final save
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] Extracted {len(articles)} total articles")
        return articles

    def save_articles(self, articles: List[Dict], filename: str = "ireland_articles.json"):
        """Save articles to JSON file"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)

        output_path = f"{self.output_dir}/{filename}"

        # Remove checkpoint file after final save
        checkpoint_file = f"{self.output_dir}/checkpoint_articles.json"
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"[CLEANUP] Removed checkpoint file")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        print(f"[SUCCESS] Saved {len(articles)} articles to {output_path}")

        # Save statistics
        stats = {
            "total_articles": len(articles),
            "total_text_length": sum(a["text_length"] for a in articles),
            "avg_text_length": sum(a["text_length"] for a in articles) / len(articles),
            "total_links": sum(len(a.get("links", [])) for a in articles),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        stats_path = f"{self.output_dir}/extraction_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"[INFO] Statistics saved to {stats_path}")
        return output_path

    def run_full_extraction(self):
        """Run complete extraction pipeline"""
        print("=" * 80)
        print("IRELAND WIKIPEDIA COMPREHENSIVE EXTRACTION")
        print("=" * 80)

        # Step 1: Get all page titles
        page_titles = self.get_all_ireland_pages()

        # Save page titles
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        with open(f"{self.output_dir}/page_titles.json", 'w') as f:
            json.dump(page_titles, f, indent=2)

        # Step 2: Extract all articles
        articles = self.extract_all_articles(page_titles)

        # Step 3: Save articles
        output_path = self.save_articles(articles)

        print("=" * 80)
        print("EXTRACTION COMPLETE!")
        print(f"Output: {output_path}")
        print("=" * 80)

        return articles


if __name__ == "__main__":
    extractor = IrelandWikipediaExtractor()
    extractor.run_full_extraction()
