"""
Day 3 (journal) - Document Loaders: TextLoader, PyPDFLoader, DirectoryLoader, WebBaseLoader
------------------------------------------------------------------------------------
Companion implementation for the "Day 3: Document Loaders" entry in
journal.md (2026-11-15). Named day3_document_loaders.py -- distinct from
day3.py, an earlier, unrelated "live HTTP GET request" exercise from the
day1.py-day8-2.py series; see the numbering note in README.md.

With the Runnable interface from the "Day 2" entry in hand, the next step in
the roadmap is getting real-world data into a chain at all -- LangChain's
document loaders. Every loader returns a list of Document objects, each
holding:

  - page_content : the raw text
  - metadata      : a dict with things like `source`, `page`, or whatever
                     field the loader attaches

so downstream steps (splitting, embedding, retrieval) can rely on one
consistent shape no matter where the data came from.

Loaders covered today (all from langchain_community.document_loaders --
still the working/installed import for these generic loaders today, though
the LangChain project is gradually migrating integrations to standalone
per-provider packages; see https://github.com/langchain-ai/langchain-community):

  - TextLoader      : reads a single plain-text file into one Document.
  - PyPDFLoader      : parses a PDF, one Document per page (metadata["page"]
                       is set automatically -- a natural chunk boundary).
  - DirectoryLoader  : walks a folder and applies a loader to every file
                       matching a glob pattern.
  - WebBaseLoader    : fetches a URL with requests + BeautifulSoup to build
                       a Document (metadata["source"] is the page URL).

No API key is needed for any of this. To keep the script fully runnable
offline/without network access to third-party sites, the sample .txt/.pdf
files are generated into a temp directory, and WebBaseLoader is pointed at
a tiny local HTTP server instead of a real public website.
"""

import http.server
import os
import tempfile
import threading
from pathlib import Path

# WebBaseLoader warns if this isn't set; it's just the User-Agent header
# sent with its HTTP requests, no secret involved.
os.environ.setdefault("USER_AGENT", "langchain-journey-day3/1.0")

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)


def make_sample_text_files(folder: Path) -> None:
    """A couple of plain-text 'notes' for TextLoader and DirectoryLoader."""
    (folder / "note_langchain.txt").write_text(
        "LangChain is a framework for building applications powered by "
        "large language models, with composable building blocks for "
        "prompts, chains, retrieval, and agents.",
        encoding="utf-8",
    )
    (folder / "note_rag.txt").write_text(
        "Retrieval-Augmented Generation (RAG) combines a retriever, which "
        "fetches relevant documents, with an LLM, which uses those "
        "documents as extra context when generating an answer.",
        encoding="utf-8",
    )


def make_sample_pdf(path: Path) -> None:
    """One tiny one-page PDF so PyPDFLoader has something real to parse."""
    try:
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(path))
        c.drawString(72, 720, "Day 3 sample PDF")
        c.drawString(72, 700, "PyPDFLoader turns each page into one Document.")
        c.save()
    except Exception as exc:
        # Basic error handling: PDF generation is a side exercise here, not
        # the point of the lesson, so fail soft and let the rest of the
        # script continue with the loaders that don't need it.
        print(f"Could not generate sample PDF ({exc}); skipping PyPDFLoader demo.")


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002 - matches base signature
        pass  # keep the demo output focused on the loader results


def start_local_http_server(folder: Path):
    """
    Serve `folder` on 127.0.0.1 on an OS-assigned free port, in a background
    thread, so WebBaseLoader has a real HTTP endpoint to fetch without
    depending on any external site being reachable from this environment.
    """
    handler = lambda *args, **kwargs: _QuietHandler(*args, directory=str(folder), **kwargs)
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return server, f"http://127.0.0.1:{port}/page.html"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="langchain_day3_") as tmp:
        folder = Path(tmp)
        make_sample_text_files(folder)
        make_sample_pdf(folder / "sample.pdf")
        (folder / "page.html").write_text(
            "<html><body><h1>Day 3</h1>"
            "<p>WebBaseLoader strips this HTML down to plain text.</p>"
            "</body></html>",
            encoding="utf-8",
        )

        # 1. TextLoader - one file -> one Document.
        text_docs = TextLoader(str(folder / "note_langchain.txt")).load()
        print("TextLoader:", len(text_docs), "document(s)")
        print("  metadata:", text_docs[0].metadata)

        # 2. DirectoryLoader - every *.txt in the folder, via TextLoader.
        dir_docs = DirectoryLoader(
            str(folder), glob="*.txt", loader_cls=TextLoader
        ).load()
        print("\nDirectoryLoader:", len(dir_docs), "document(s)")
        for doc in dir_docs:
            print("  metadata:", doc.metadata)

        # 3. PyPDFLoader - one Document per PDF page.
        pdf_path = folder / "sample.pdf"
        if pdf_path.exists():
            pdf_docs = PyPDFLoader(str(pdf_path)).load()
            print("\nPyPDFLoader:", len(pdf_docs), "document(s)")
            print("  metadata:", pdf_docs[0].metadata)
        else:
            print("\nPyPDFLoader: skipped (sample PDF was not generated)")

        # 4. WebBaseLoader - fetch over real HTTP, against our local server.
        server, url = start_local_http_server(folder)
        try:
            web_docs = WebBaseLoader(url).load()
            print("\nWebBaseLoader:", len(web_docs), "document(s)")
            print("  metadata:", web_docs[0].metadata)
            print("  content snippet:", web_docs[0].page_content.strip()[:80])
        except Exception as exc:
            # Basic error handling: a real WebBaseLoader call depends on
            # network access, so report the failure clearly instead of
            # letting the whole script crash.
            print(f"\nWebBaseLoader failed ({exc}); the other three loaders "
                  "above do not depend on network access.")
        finally:
            server.shutdown()

    # Exercise (left for next time, once text splitting is covered): take
    # `dir_docs` from DirectoryLoader above and split each Document with
    # RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20),
    # then compare len(dir_docs) before vs. the number of chunks after.
    # That's the natural next topic: these raw Documents are usually too
    # large to embed as-is.


if __name__ == "__main__":
    main()
