from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

def crear_agente(nombre):
    llm = OllamaLLM(
        model="llama3.2",
        temperature=0.7,
        system="Eres una persona normal llamada " + nombre + ". Responde de forma casual y breve EN ESPAÑOL como lo haría un humano."
    )
    
    template = "Humano: {input}\n" + nombre + ":"
    prompt = PromptTemplate(input_variables=["input"], template=template)
    
    chain = prompt | llm | StrOutputParser()
    return chain

def main():
    # Conversación simple
    juan = crear_agente("Juan")
    maria = crear_agente("María")
    
    print("\nJuan: ¡Hola! ¿Cómo estás?")
    respuesta = maria.invoke({"input": "¡Hola! ¿Cómo estás?"})
    print("María:", respuesta)
    
    respuesta = juan.invoke({"input": respuesta})
    print("Juan:", respuesta)
    
    print("\nFin de la conversación")

if __name__ == "__main__":
    main()