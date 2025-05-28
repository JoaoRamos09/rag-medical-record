from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import logging


def invoking_model(input_user, model_name="gpt-4o-mini", temperature=0, prompt_system=None, structured_output=None):
    try:
        model = get_model(model_name, temperature, structured_output)
        return model.invoke(get_messages(input_user, get_system_prompt(prompt_system)))
    except Exception as e:
        logging.error(f"[OPENAI] Error invoking model: {e}")
        raise Exception() #todo: adicionar erro personalizado

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
        raise Exception() #todo: adicionar erro personalizado
    
def invoking_model_with_few_shot_prompt(input_user, few_shot_prompt, model_name="gpt-4o-mini", prompt_system=None, temperature=0, structured_output=None):
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_system_prompt(prompt_system)),
        (few_shot_prompt),
        ("user", "{input}")
    ])
    
    try:
        chain = prompt | get_model(model_name, temperature, structured_output)
        return chain.invoke({"input": input_user})
    except Exception as e:
        logging.error(f"[OPENAI] Error invoking model: {e}")
        raise Exception() #todo: adicionar erro personalizado
            

def get_model(model_name, temperature, structured_output):
    try:
        if structured_output is None:
            return ChatOpenAI(model_name=model_name, temperature=temperature)
        else:
            return ChatOpenAI(model_name=model_name, temperature=temperature).with_structured_output(structured_output)
    except Exception as e:
        logging.error(f"[OPENAI] Error in getting model: {e}")
        raise Exception() #todo: adicionar erro personalizado
    
def get_messages(input_user, prompt_system):
    
    return [
        SystemMessage(content=prompt_system),
        HumanMessage(content=input_user)
    ]

def get_system_prompt(prompt_system):
    if prompt_system is None:
        prompt_system = "You are a helpful assistant."
    return prompt_system