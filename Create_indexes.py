def process_pdf(pdf_file, index_name):


    import os
    import re
    from PyPDF2 import PdfReader
    import openai
    import pinecone
    from tqdm.auto import tqdm
    from dotenv import load_dotenv
    import pyfiglet


    # Load environment variables
    load_dotenv()

    # Set up OpenAI API key and model
    openai.api_key = os.getenv("OPENAI_API_KEY")
    MODEL = "text-embedding-ada-002"

    # Set up Pinecone API key
    pinecone.api_key = os.getenv("PINECONE_API_KEY")


    def load_pdf_extract_text(pdf_file):
        with open(pdf_file, 'rb') as f:
            pdf = PdfReader(f)
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + " "
            paragraphs = [text[i:i+1000] for i in range(0, len(text), 600)]
        return paragraphs

    def generate_embeddings(paragraphs, model):
        batch_size = 500
        embeds = []
        for i in tqdm(range(0, len(paragraphs), batch_size)):
            i_end = min(i + batch_size, len(paragraphs))
            lines_batch = paragraphs[i:i_end]
            res = openai.Embedding.create(input=lines_batch, engine=model)
            embeds_batch = [record['embedding'] for record in res['data']]
            embeds += embeds_batch
        return embeds

    def create_pinecone_index(index_name, dimension):
        pinecone.init(api_key=pinecone.api_key, environment="us-east1-gcp")
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(name=index_name, dimension=dimension)
        return pinecone.Index(index_name)



    def insert_embeddings(index, paragraphs, embeds):
        batch_size = 100
        for i in tqdm(range(0, len(paragraphs), batch_size)):
            i_end = min(i + batch_size, len(paragraphs))
            lines_batch = paragraphs[i:i_end]
            ids_batch = [str(n) for n in range(i, i_end)]
            embeds_batch = embeds[i:i_end]
            meta = [{'text': line} for line in lines_batch]
            to_upsert = zip(ids_batch, embeds_batch, meta)
            index.upsert(vectors=list(to_upsert))


    
    paragraphs = load_pdf_extract_text(pdf_file)
    embeds = generate_embeddings(paragraphs, MODEL)

    # Get user input for index_name
    index_name = input("Ingresa el nombre del índice (solo minúsculas y guiones): ").strip().lower()
    index_name = re.sub(r"[^a-z-]+", "", index_name)

    index = create_pinecone_index(index_name, len(embeds[0]))
    insert_embeddings(index, paragraphs, embeds)

    print(f"Índice '{index_name}' creado con éxito.")


