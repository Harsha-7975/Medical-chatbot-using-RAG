def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 700,chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
print(len(text_chunks))