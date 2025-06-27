import os
import shutil
import time
from collections import defaultdict
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

DATA_DIR = "./text_files"
PERSIST_DIR = "./storage"

def main():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Directory {DATA_DIR} does not exist. Please create it and add some documents.")
        return

    try:
        files_in_dir = os.listdir(DATA_DIR)
        if not files_in_dir:
            print(f"❌ Directory {DATA_DIR} is empty. Please add some documents to index.")
            return
    except Exception as e:
        print(f"❌ Error accessing directory {DATA_DIR}: {e}")
        return

    print(f"📁 Loading documents from {DATA_DIR}")
    print(f"📄 Found files: {files_in_dir}")

    # Enhanced document loading with per-file tracking
    documents = []
    file_info = []
    for filename in files_in_dir:
        filepath = os.path.join(DATA_DIR, filename)
        if os.path.isfile(filepath):
            try:
                # Log file info before processing
                file_type = filename.split('.')[-1] if '.' in filename else 'unknown'
                file_size = os.path.getsize(filepath)
                print(f"\n📄 Processing {filename}")
                print(f"  - Type: {file_type}")
                print(f"  - Size: {file_size} bytes")

                loader = SimpleDirectoryReader(
                input_files=[filepath],
                file_metadata=lambda _: {"source": filename}  # Force metadata
                )
                
                # Load the file
                file_docs = loader.load_data()
                documents.extend(file_docs)
                file_info.append({
                    'filename': filename,
                    'type': file_type,
                    'size': file_size,
                    'doc_count': len(file_docs),
                    'processed': True
                })
                print(f"  - Documents extracted: {len(file_docs)}")
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                file_info.append({
                    'filename': filename,
                    'processed': False,
                    'error': str(e)
                })

    if not documents:
        print("❌ No valid documents were loaded")
        return

    print("\n🔄 Processing documents into chunks...")
    start_chunk = time.time()
    splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=20,
    include_metadata=True,  # Ensure metadata flows through
    paragraph_separator="\n\n"  # Prefer splitting at paragraphs
    )  # Using 20 overlap as requested
    nodes = splitter.get_nodes_from_documents(documents)
    print("nodes", nodes)


    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    print("\n🔍 Token Counts per Chunk:")
    for i, node in enumerate(nodes):
        token_count = len(tokenizer.encode(node.text))
        print(f"Chunk {i+1}: {token_count} tokens | Source: {os.path.basename(str(node.metadata.get('source', 'unknown')))}")
        if i < 3:  # Print preview of first 3 chunks
            print(f"Preview: {node.text[:100]}...\n")
    
    # Track chunks per file
    chunks_per_file = defaultdict(int)
    filename_mapping = {os.path.basename(f['filename']): f['filename'] for f in file_info}
    
    for node in nodes:
        if hasattr(node, 'metadata') and node.metadata:
            source = node.metadata.get('source', '') or node.metadata.get('file_name', '')
            if not source and 'file_path' in node.metadata:
                source = node.metadata['file_path']
            
            if source:
                # Get the base filename
                source_base = os.path.basename(str(source))
                # Clean up any added suffixes
                source_base = source_base.split('_page_')[0].split('_section_')[0]
                
                # Try to match with our original filenames
                matched_filename = None
                for orig_filename in filename_mapping.values():
                    if orig_filename in source_base or source_base in orig_filename:
                        matched_filename = orig_filename
                        break
                
                if matched_filename:
                    chunks_per_file[matched_filename] += 1
                else:
                    # Fallback to whatever we could extract
                    chunks_per_file[source_base] += 1

    # Print comprehensive chunking statistics
    print("\n📊 Chunking Report:")
    total_chunks = 0
    for info in file_info:
        filename = info['filename']
        base_name = os.path.basename(filename)
        if info.get('processed', False):
            # Try multiple ways to match the filename
            chunks = (chunks_per_file.get(filename, 0) or 
                     chunks_per_file.get(base_name, 0) or
                     sum(v for k,v in chunks_per_file.items() if filename in k or k in filename))
            total_chunks += chunks
            print(f"  - {filename}:")
            print(f"    • Type: {info['type']}")
            print(f"    • Size: {info['size']} bytes")
            print(f"    • Documents: {info['doc_count']}")
            print(f"    • Chunks created: {chunks}")
    
    chunk_time = time.time() - start_chunk
    print(f"\n✅ Total chunks created: {len(nodes)} | Time taken: {chunk_time:.2f} seconds")

    print("\n🔄 Loading embedding model (this may take a moment)...")
    try:
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Embedding model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        return

    if os.path.exists(PERSIST_DIR):
        print(f"\n🔄 Removing existing storage at {PERSIST_DIR}")
        try:
            shutil.rmtree(PERSIST_DIR)
        except Exception as e:
            print(f"⚠️  Could not remove existing storage: {e}")

    try:
        os.makedirs(PERSIST_DIR, exist_ok=True)
        print(f"\n✅ Created storage directory at {PERSIST_DIR}")
    except Exception as e:
        print(f"❌ Error creating storage directory: {e}")
        return

    print("\n🔄 Building vector index (this may take several minutes)...")
    start_index = time.time()
    try:
        storage_context = StorageContext.from_defaults()
        index = VectorStoreIndex(nodes, embed_model=embed_model, storage_context=storage_context)
        index_time = time.time() - start_index
        print(f"\n✅ Vector index built successfully | Time taken: {index_time:.2f} seconds")
    except Exception as e:
        print(f"❌ Error building index: {e}")
        return

    print("\n🔄 Saving index to disk...")
    try:
        storage_context.persist(persist_dir=PERSIST_DIR)
        print(f"✅ Index saved to {PERSIST_DIR}")
        
        # Verify embeddings
        stored_nodes = len(index.docstore.docs)
        print("\n🔍 Embedding Verification:")
        print(f"  - Total chunks created: {len(nodes)}")
        print(f"  - Successfully embedded: {stored_nodes}")
        print(f"  - Embedding success rate: {stored_nodes/len(nodes)*100:.2f}%")
        if stored_nodes != len(nodes):
            print("⚠️  Warning: Some chunks were not properly embedded/stored")
        
    except Exception as e:
        print(f"❌ Error saving index: {e}")
        return

    abs_path = os.path.abspath(PERSIST_DIR)
    print("\n🎉 Index creation completed successfully!")
    print(f"📍 Index location: {abs_path}")
    print(f"⏱️  Timing Statistics:")
    print(f"  - Chunking Time: {chunk_time:.2f} seconds")
    print(f"  - Index Creation Time: {index_time:.2f} seconds")
    print("\n💡 You can now use this index to query your documents.")

if __name__ == "__main__":
    main()