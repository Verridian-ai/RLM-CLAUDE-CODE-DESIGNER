#!/usr/bin/env python3
"""
Document Extraction Tool for RLM System
Supports multiple PDF extraction methods for robust text extraction
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Set UTF-8 encoding for Windows console output
if sys.platform == 'win32':
    import codecs
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def extract_with_pymupdf(pdf_path):
    """Extract text using PyMuPDF (fitz)"""
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = ""
        metadata = {
            "page_count": len(doc),
            "metadata": doc.metadata,
            "method": "PyMuPDF"
        }

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += f"\n--- PAGE {page_num + 1} ---\n"
            text += page.get_text()

        doc.close()
        return text, metadata
    except Exception as e:
        return None, {"error": str(e), "method": "PyMuPDF"}

def extract_with_pdfplumber(pdf_path):
    """Extract text using pdfplumber"""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            metadata = {
                "page_count": len(pdf.pages),
                "metadata": pdf.metadata,
                "method": "pdfplumber"
            }

            for page_num, page in enumerate(pdf.pages):
                text += f"\n--- PAGE {page_num + 1} ---\n"
                page_text = page.extract_text()
                if page_text:
                    text += page_text

                # Extract tables if any
                tables = page.extract_tables()
                if tables:
                    text += f"\n[TABLES ON PAGE {page_num + 1}]\n"
                    for table_num, table in enumerate(tables):
                        text += f"\nTable {table_num + 1}:\n"
                        for row in table:
                            text += " | ".join([cell or "" for cell in row]) + "\n"

        return text, metadata
    except Exception as e:
        return None, {"error": str(e), "method": "pdfplumber"}

def extract_with_pypdf(pdf_path):
    """Extract text using pypdf"""
    try:
        import pypdf
        text = ""
        reader = pypdf.PdfReader(pdf_path)

        metadata = {
            "page_count": len(reader.pages),
            "metadata": reader.metadata,
            "method": "pypdf"
        }

        for page_num, page in enumerate(reader.pages):
            text += f"\n--- PAGE {page_num + 1} ---\n"
            text += page.extract_text()

        return text, metadata
    except Exception as e:
        return None, {"error": str(e), "method": "pypdf"}

def save_extraction_result(text, metadata, output_path):
    """Save extracted text and metadata"""
    result = {
        "timestamp": datetime.now().isoformat(),
        "extraction_metadata": metadata,
        "content": text
    }

    # Save as JSON
    with open(f"{output_path}.json", 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Save text only
    with open(f"{output_path}.txt", 'w', encoding='utf-8') as f:
        f.write(text)

    return result

def main():
    if len(sys.argv) != 2:
        print("Usage: python pdf_extractor.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found")
        sys.exit(1)

    base_name = Path(pdf_path).stem
    output_dir = Path("extracted_content")
    output_dir.mkdir(exist_ok=True)

    print(f"Extracting content from: {pdf_path}")
    print("=" * 50)

    # Try multiple extraction methods
    methods = [
        ("PyMuPDF", extract_with_pymupdf),
        ("pdfplumber", extract_with_pdfplumber),
        ("pypdf", extract_with_pypdf)
    ]

    best_result = None
    best_score = 0

    for method_name, extract_func in methods:
        print(f"\nTrying {method_name}...")
        text, metadata = extract_func(pdf_path)

        if text:
            # Score based on text length and readability
            score = len(text.strip())
            print(f"  [OK] Success - {score} characters extracted")

            if score > best_score:
                best_score = score
                best_result = (text, metadata, method_name)

            # Save individual method result
            output_path = output_dir / f"{base_name}_{method_name.lower()}"
            save_extraction_result(text, metadata, str(output_path))
        else:
            print(f"  [FAIL] Failed - {metadata.get('error', 'Unknown error')}")

    if best_result:
        text, metadata, method = best_result
        print(f"\nBest extraction: {method}")
        print(f"Content length: {len(text)} characters")

        # Save best result as primary output
        primary_output = output_dir / f"{base_name}_extracted"
        result = save_extraction_result(text, metadata, str(primary_output))

        print(f"\nSaved to: {primary_output}.txt and {primary_output}.json")

        # Print first 1000 characters as preview
        print("\nContent Preview:")
        print("-" * 50)
        print(text[:1000] + "..." if len(text) > 1000 else text)

        return True
    else:
        print("\n[ERROR] All extraction methods failed")
        return False

if __name__ == "__main__":
    main()