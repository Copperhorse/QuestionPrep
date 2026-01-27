from markdown_chunker import MarkdownChunkingStrategy

# Create a chunking strategy with default configuration
strategy = MarkdownChunkingStrategy(add_metadata=True)

# Or customize the parameters
strategy = MarkdownChunkingStrategy(
    min_chunk_len=512,  # Minimum chunk size (default: 512)
    soft_max_len=1024,  # Preferred maximum chunk size (default: 1024)
    hard_max_len=2048,  # Absolute maximum chunk size (default: 2048)
    detect_headers_footers=True,  # Detect and remove repeating headers/footers
    remove_duplicates=True,  # Remove duplicate chunks
    add_metadata=True,  # Add metadata in each chunk as YAML front matter
)

# Chunk a Markdown document
with open("/home/copper/Desktop/Project/test.md", "r") as f:
    content = f.read()

chunks = strategy.chunk_markdown(content)

# Process the chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:")
    print(chunk)
    print("-" * 80)
