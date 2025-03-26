from openai import OpenAI
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import os

load_dotenv()

def query_duckduckgo(query):
    """
    使用DuckDuckGo执行搜索查询并返回搜索结果。
    """
    results = []
    try:
        ddgs = DDGS()
        res = ddgs.text(query, max_results=10)
        for obj in res:
            body_text = obj.get('body', '')
            if 'title' in obj and obj['title']:
                body_text = f"{obj['title']}: {body_text}"
            results.append(body_text)
        return results
    except Exception as e:
        print(f"Error executing DuckDuckGo search: {e}")
        return results
    except Exception as e:
        return results


def response_with_rag(question, search_results):
    """
    使用ChatGPT生成响应,结合检索到的文本。
    """
    prompt = (
        f"As an AI assistant, your task is to answer the following question using the relevant information retrieved from the web.\n"
        f"Question: {question}\n"
        f"Relevant Information{search_results}:\n"
        "Please formulate a concise response to the question.")
    # print(prompt)

    # 调用openai
    # openai.api_key = os.getenv("openai_api_key")
    # response = openai.chat.completions.create(model="gpt-4o-mini",
    #                                           messages=[{
    #                                               "role": "user",
    #                                               "content": prompt
    #                                           }])
    # return response.choices[0].message.content

    # 调用零一万物
    # lingyi_api_key = os.getenv("lingyi_api_key")
    # client = OpenAI(api_key=lingyi_api_key,
    #                 base_url="https://api.lingyiwanwu.com/v1")
    # response = client.chat.completions.create(model="yi-lightning",
    #                                             messages=[{
    #                                                 "role": "user",
    #                                                 "content": prompt
    #                                             }])
    # return response.choices[0].message.content

    # 调用deepseek
    deeseek_api_key = os.getenv("deeseek_api_key")
    client = OpenAI(api_key=deeseek_api_key,
                    base_url="https://api.deepseek.com")
    response = client.chat.completions.create(model="deepseek-chat",
                                              messages=[{
                                                  "role": "user",
                                                  "content": prompt
                                              }],
                                              stream=False)
    return response.choices[0].message.content


def rag_search(question):
    """
    使用DuckDuckGo进行搜索以获取相关信息并返回结果。
    """
    # question = input("Enter your question: ")
    # 第一步：使用DuckDuckGo检索信息
    search_results = query_duckduckgo(question)
    # print(question)
    # print(search_results)
    # 第二步：使用检索到的信息和ChatGPT生成响应
    response = response_with_rag(question, search_results)
    return response


if __name__ == "__main__":
    print(rag_search("小鹏汽车最近的销量如何?"))

