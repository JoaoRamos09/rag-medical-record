from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import logging
import os
from dotenv import load_dotenv
from app.exceptions.openai_errors import OpenAIInvokingError, OpenAIModelError

# Carrega as variáveis de ambiente
load_dotenv()

# Configuração da chave da API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente. Verifique se o arquivo .env está configurado corretamente.")

async def invoking_model(user_input, model_name="gpt-4o-mini", temperature=0, prompt_system=None, structured_output=None):
    try:
        model = get_model(model_name, temperature, structured_output)
        return await model.ainvoke(get_messages(user_input, get_system_prompt(prompt_system)))
    except Exception as e:
        logging.error(f"[OPENAI] Error invoking model: {e}")
        raise OpenAIInvokingError(message=f"[OPENAI] Error invoking model: {e}")


def invoking_model_with_messages_placeholder(messages, model_name="gpt-4o-mini", prompt_system=None, temperature=0, structured_output=None):
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_system_prompt(prompt_system)),
        ("placeholder", "{messagess}")
    ])
    
    try:
        chain = prompt | get_model(model_name, temperature, structured_output)
        return chain.invoke({"messages": messages})
    except Exception as e:
        logging.error(f"[OPENAI] Error invoking model: {e}")
        raise OpenAIInvokingError(message=f"[OPENAI] Error invoking model: {e}")
    
    
async def invoking_model_with_few_shot_prompt(user_input, few_shot_prompt, model_name="gpt-4o-mini", prompt_system=None, temperature=0, structured_output=None):
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_system_prompt(prompt_system)),
        (few_shot_prompt),
        ("user", "{input}")
    ])
    
    try:
        chain = prompt | get_model(model_name, temperature, structured_output)
        return await chain.ainvoke({"input": user_input})
    except Exception as e:
        logging.error(f"[OPENAI] Error invoking model: {e}")
        raise OpenAIInvokingError(message=f"[OPENAI] Error invoking model: {e}")
            

def get_model(model_name, temperature, structured_output):
    try:
        if structured_output is None:
            return ChatOpenAI(model_name=model_name, temperature=temperature, api_key=api_key)
        else:
            return ChatOpenAI(model_name=model_name, temperature=temperature, api_key=api_key).with_structured_output(structured_output)
    except Exception as e:
        raise OpenAIModelError(message=f"[OPENAI] Error getting model: {e}")

def get_embeddings(model_name="text-embedding-3-small"):
    try:
        return OpenAIEmbeddings(model=model_name, api_key=api_key)
    except Exception as e:
        raise OpenAIModelError(message=f"[OPENAI] Error getting embeddings: {e}")
    

def get_messages(input_user, prompt_system):
    
    return [
        SystemMessage(content=prompt_system),
        HumanMessage(content=input_user)
    ]

def get_system_prompt(prompt_system):
    return prompt_system if prompt_system is not None else "You are a helpful assistant."
