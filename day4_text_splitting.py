"""
Day 4 (journal) - Text Splitting: CharacterTextSplitter and
RecursiveCharacterTextSplitter
------------------------------------------------------------------------------
Companion implementation for the "Day 4: Text Splitting" entry in journal.md
(2026-11-15). Named day4_text_splitting.py to keep going with the
journal.md-specific numbering track started by day2_lcel_runnables.py and
day3_document_loaders.py; see the numbering note in README.md.
------------------------------------------------------------------------------
Day 3 loaded raw sources into `Document` objects (page_content + metadata).
Those Documents are usually far too large to hand straight to an embedding
model or stuff into an LLM's context window, so the next step in the
roadmap (Days 11-15: document loaders -> text splitting -> embeddings ->
vector databases) is splitting them into smaller, overlapping "chunks":

  - chunk_size    : the target maximum size of each chunk (in characters,
                     for the splitters used here).
  - chunk_overlap : how many characters the end of one chunk repeats at the
                     start of the next, so a sentence that straddles a chunk
                     boundary still appears whole in at least one chunk --
                     this preserves local context for retrieval later.

Splitters covered today (both from langchain_text_splitters, the standalone
package LangChain migrated these utilities into -- langchain.text_splitter
still re-exports them for backwards compatibility, but the dedicated package
is the current recommended import):

  - CharacterTextSplitter          : splits on a single separator (default
                                      "\\n\\n"), falling back to a hard cut
                                      only if a "paragraph" is still bigger
                                      than chunk_size.
  - RecursiveCharacterTextSplitter : tries a *list* of separators in order
                                      (paragraph -> line -> sentence-ish ->
                                      word -> character), recursing into
                                      whichever piece is still too big. This
                                      is the splitter LangChain recommends by
                                      default for generic text, since it
                                      keeps semantically-related text
                                      together far more often than a single
                                      fixed separator.

Both splitters expose the same two entry points used below:
  - split_text(str) -> list[str]            : for raw strings.
  - split_documents(list[Document]) -> list[Document] : splits each
    Document's page_content while copying its metadata onto every chunk, so
    a chunk still knows which file/page it came from.

No API key or network access is needed for any of this.
"""

from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# A deliberately "messy" sample: short lines, one long run-on paragraph, and
# a blank-line-separated section, so the two splitters visibly disagree on
# where the best cut points are.
SAMPLE_TEXT = """\
LangChain Text Splitting

Text splitting breaks large documents into smaller chunks before they are \
embedded and stored in a vector database. Picking a good chunk size and \
overlap matters a lot: chunks that are too large waste context and dilute \
the embedding's focus, while chunks that are too small lose the surrounding \
context a retriever needs to make sense of a passage. Overlap between \
consecutive chunks helps a sentence that falls on a chunk boundary still \
appear whole in at least one chunk.

RecursiveCharacterTextSplitter is the default choice for generic text \
because it tries paragraph breaks first, then line breaks, then sentence \
breaks, then words, only falling back to a hard character cut as a last \
resort. This keeps related sentences together far more often than a single \
fixed separator would.

Day 5 will take these chunks and turn them into embeddings, the next stop \
on the roadmap toward a working vector store and retriever.
"""


def build_sample_documents() -> list[Document]:
    """
    Two short Documents, mimicking what DirectoryLoader/TextLoader produced
    in Day 3 -- metadata included, so we can check it survives splitting.
    """
    return [
        Document(
            page_content=SAMPLE_TEXT,
            metadata={"source": "notes/text_splitting.txt"},
        ),
        Document(
            page_content=(
                "Short note.\n\nThis one is intentionally tiny, so a sane "
                "splitter should return it as a single chunk rather than "
                "cutting it up for no reason."
            ),
            metadata={"source": "notes/short_note.txt"},
        ),
    ]


def compare_splitters(chunk_size: int = 220, chunk_overlap: int = 40) -> None:
    """Run both splitters on the same text with the same settings and
    print how differently they cut it up."""
    char_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    char_chunks = char_splitter.split_text(SAMPLE_TEXT)
    recursive_chunks = recursive_splitter.split_text(SAMPLE_TEXT)

    print(f"Settings: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print(f"CharacterTextSplitter          -> {len(char_chunks)} chunk(s)")
    for i, chunk in enumerate(char_chunks):
        print(f"  [{i}] ({len(chunk)} chars) {chunk[:60]!r}...")

    print(f"\nRecursiveCharacterTextSplitter -> {len(recursive_chunks)} chunk(s)")
    for i, chunk in enumerate(recursive_chunks):
        print(f"  [{i}] ({len(chunk)} chars) {chunk[:60]!r}...")


def split_documents_preserving_metadata() -> list[Document]:
    """
    split_documents() is the version to reach for once you have real
    Document objects (e.g. from Day 3's loaders): each output chunk is
    still a Document, and it inherits the parent's metadata automatically.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=220, chunk_overlap=40)
    docs = build_sample_documents()
    chunks = splitter.split_documents(docs)

    print(f"\nsplit_documents(): {len(docs)} input Document(s) -> "
          f"{len(chunks)} chunk(s)")
    for chunk in chunks:
        # Error handling: metadata should never silently disappear during a
        # split -- assert it instead of letting a later retrieval step fail
        # mysteriously with no "source" to point to.
        assert "source" in chunk.metadata, "metadata was lost during split!"
        print(f"  source={chunk.metadata['source']!r} "
              f"len={len(chunk.page_content)} "
              f"preview={chunk.page_content[:50]!r}")
    return chunks


def main() -> None:
    try:
        compare_splitters()
        split_documents_preserving_metadata()
    except Exception as exc:
        # Basic error handling: splitting is pure-Python/offline, so a
        # failure here almost certainly means a bad chunk_size/chunk_overlap
        # combination (e.g. overlap >= chunk_size) rather than a network or
        # credentials problem -- surface it clearly either way.
        print(f"Text splitting failed ({exc}).")
        raise

    # Exercise: re-run compare_splitters() with a much smaller chunk_size
    # (e.g. 60) and chunk_overlap=10, and notice two things:
    #   1. Both splitters now produce far more chunks.
    #   2. RecursiveCharacterTextSplitter still tends to break on word
    #      boundaries where it can, while CharacterTextSplitter, once a
    #      "paragraph" no longer fits, falls back to a harder, less
    #      readable cut. Try it and compare the printed chunk previews.
    print("\nExercise: try compare_splitters(chunk_size=60, chunk_overlap=10) "
          "and compare how each splitter's chunk boundaries look.")
    compare_splitters(chunk_size=60, chunk_overlap=10)


if __name__ == "__main__":
    main()
