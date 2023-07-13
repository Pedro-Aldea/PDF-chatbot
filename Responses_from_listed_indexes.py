import os
import time
import openai
import pinecone
from dotenv import load_dotenv
import pyfiglet

MODEL = "text-embedding-ada-002"



def load_environment_variables():
    load_dotenv(".env")
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
    os.environ["PINECONE_ENVIRONMENT"] = os.getenv("PINECONE_ENVIRONMENT")




def list_pinecone_indexes():
    indexes = pinecone.list_indexes()
    return indexes


def connect_to_index(index_name):
    api_key = os.getenv("PINECONE_API_KEY")
    pinecone.init(api_key=api_key, environment="us-east1-gcp")
    return pinecone.Index(index_name)



def ask_questions_using_gpt(index, model):
    continue_loop = True
    while continue_loop:
        start = time.time()
        question = input("Cual es tu pregunta: ")
        print("User:\n" + question)

        if question == 'salir':
            continue_loop = False
            break
        else:
            embed_question = openai.Embedding.create(input=question, engine=model)['data'][0]['embedding']
            res = index.query([embed_question], top_k=10, include_metadata=True)

            matching_texts = []
            for match in res['matches']:
                matching_texts.append(match['metadata']['text'])
                matching_texts_str = ' '.join(matching_texts)

            entrada = f"Basándote en el siguiente texto: {matching_texts_str}\nResponde de la manera más detallada siendo fiel a la información a la siguiente pregunta: {question}\n"

            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": entrada}])

            if completion is not None:
                print("\n\nRespuesta de ChatGPT:\n" + completion.choices[0].message.content + "\n\n\n")
                end = time.time()
                print("Tiempo de respuesta: ", end - start, "segundos\n")
            else:
                print("Error: No se pudo obtener una respuesta del modelo.\n")

def process_question(question, index, model):

    embed_question = openai.Embedding.create(input=question, engine=model)['data'][0]['embedding']
    res = index.query([embed_question], top_k=10, include_metadata=True)

    matching_texts = []
    for match in res['matches']:
        matching_texts.append(match['metadata']['text'])
        matching_texts_str = ' '.join(matching_texts)

    entrada = f"Basándote en el siguiente texto: {matching_texts_str}\nResponde de la manera más detallada siendo fiel a la información a la siguiente pregunta: {question}\n"

    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": entrada}])

    if completion is not None:
        return completion.choices[0].message.content
    else:
        return "Error: No se pudo obtener una respuesta del modelo."

def main():
    T = "Welcome to chat_PDF"
    ASCII_art_1 = pyfiglet.figlet_format(T)
    print(ASCII_art_1)

    load_environment_variables()
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east1-gcp")
    indexes = list_pinecone_indexes()

    print("Índices disponibles en Pinecone:")
    for idx, index_name in enumerate(indexes, start=1):
        print(f"{idx}. {index_name}")

    chosen_index = int(input("Elige el índice al que deseas conectarte (ingresa el número): "))
    index_name = indexes[chosen_index - 1]
    index = connect_to_index(index_name)

    print(f"Conectado al índice '{index_name}'.")
    
    continue_loop = True
    while continue_loop:
        start = time.time()
        question = input("Cual es tu pregunta: ")
        print("User:\n" + question)

        if question == 'salir':
            continue_loop = False
            break
        else:
            response = process_question(question, index, MODEL)
            print("\n\nRespuesta de ChatGPT:\n" + response + "\n\n\n")
            end = time.time()
            print("Tiempo de respuesta: ", end - start, "segundos\n")

    if __name__ == "__main__":
        main()

