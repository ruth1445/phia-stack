import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urljoin


# -------- CONFIG: EDIT THESE --------

# Public search URLs for categories you care about
SEARCH_URLS = {
    "women_boots": "https://poshmark.com/search?department=Women&category_id=boots",
    "women_coats": "https://poshmark.com/search?department=Women&category_id=coats",
    "women_dresses": "https://poshmark.com/search?department=Women&category_id=dresses",
}

MAX_PAGES_PER_CATEGORY = 8   # ~40-48 items per page -> 300-400 per category
REQUEST_DELAY = 3.0          # seconds between requests (be polite)


# -------- SCRAPER LOGIC --------

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PhiaResearchBot/0.1; +https://github.com/ruth1445)",
}


def scrape_search_page(url: str) -> list[dict]:
    """
    Scrape a single search results page and return a list of listing dicts.
    This uses the HTML structure of Poshmark's cards -- you may need to tweak
    selectors if they change their markup.
    """
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    items = []
    cards = soup.select("div.tile")  # main listing cards

    for card in cards:
        try:
            link_tag = card.select_one("a")
            if not link_tag:
                continue
            rel_url = link_tag.get("href", "")
            full_url = urljoin("https://poshmark.com", rel_url)

            title_tag = card.select_one(".title, .tile__title")
            title = title_tag.get_text(strip=True) if title_tag else None

            brand_tag = card.select_one(".brand, .tile__brand")
            brand = brand_tag.get_text(strip=True) if brand_tag else None

            price_tag = card.select_one(".price, .tile__price")
            price = price_tag.get_text(strip=True) if price_tag else None

            # optional extra bits
            size_tag = card.select_one(".size, .tile__size")
            size = size_tag.get_text(strip=True) if size_tag else None

            img_tag = card.select_one("img")
            image_url = img_tag.get("src") if img_tag else None

            items.append(
                {
                    "title": title,
                    "brand": brand,
                    "price_text": price,
                    "size": size,
                    "url": full_url,
                    "image_url": image_url,
                }
            )
        except Exception:
            continue

    return items


def scrape_category(label: str, base_url: str, max_pages: int) -> pd.DataFrame:
    """
    Scrape multiple pages for a single category.
    Assumes pagination via ?page=2, ?page=3, ... which works on many Poshmark searches.
    """
    all_items: list[dict] = []
    print(f"\n=== Scraping category: {label} ===")

    for page in range(1, max_pages + 1):
        paged_url = f"{base_url}&page={page}"
        print(f"[{label}] Page {page} -> {paged_url}")
        try:
            page_items = scrape_search_page(paged_url)
            if not page_items:
                print(f"[{label}] No items found on page {page}, stopping.")
                break
            all_items.extend(page_items)
            print(f"[{label}] Total items so far: {len(all_items)}")
        except Exception as e:
            print(f"[{label}] Error on page {page}: {e}")
            break

        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_items)
    df["category_label"] = label
    return df


def scrape_all_categories() -> pd.DataFrame:
    all_dfs = []
    for label, url in SEARCH_URLS.items():
        df_cat = scrape_category(label, url, MAX_PAGES_PER_CATEGORY)
        all_dfs.append(df_cat)
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


if __name__ == "__main__":
    df_all = scrape_all_categories()
    print(f"\nScraped total rows: {len(df_all)}")

    # Save to your data folder
    output_path = "data/poshmark_multicat.csv"
    df_all.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
