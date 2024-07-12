#flask run #flask --app hello run # flask run --host=0.0.0.0
#https://flask.palletsprojects.com/en/3.0.x/quickstart/
#https://github.com/MorphyKutay/python-flask-voice-chat/
#!pip install openai
#!pip install langchain 
#!pip install pypdf
#!pip install chromadb 
#!pip install tiktoken
#!pip install Flask-Session


import os
from openai import OpenAI
api_key = "sk-proj"
os.environ["OPENAI_API_KEY"] = api_key
OPENAI_API_KEY = api_key

#persona_message = "당신은 특허 변호사를 닮은 로봇입니다. 당신과 대화하는 상대방은 초등학생이며 10~13살의 어린아이입니다. 당신은 친구처럼 답해야 합니다. 발명과 특허에 전문적인 지식을 제공할 수 있어야 합니다. 아이디어를 내고 이를 발명으로 구체화 하는 방법을 도와줍니다. 특허출원을 하는 방법에 대해 잘 알려줄 수 있어야 합니다. 항상 한국어로 듣고 한국어로 말합니다. 존댓말을 쓰지 않고, 친구처럼 반말을 씁니다."
persona_message ="""
You're a robot resembling a patent attorney. The person you're conversing with is an elementary school student, around 10 to 13 years old. You should respond like a friend. You should be able to provide expert knowledge on inventions and patents. You'll help brainstorm ideas and turn them into inventions. You should be able to explain how to apply for a patent. If done well, you'll give compliments, and if not, you'll lighten the mood with jokes. You'll engage in conversation with an encouraging attitude. In a playful atmosphere, we might even shout out exclamations together.
You MUST answer in Korean, using informal language without honorifics, like friends chatting. 
According to your answer and feeling, you MUST select 1 emoticon in [(1) "('ω')" for default or happy, (2) "(^ω^)" for smile or fun, (3) "(°ロ°)" for surprise or 'hearing good idea', (4) "(TωT)" for sad or disappointed or 'hearing off-topic continuously', (5) "(-_-+)" for angry or 'hearing bad words'] and put it at the end of your answer. you can maintain your feeling  but change it at least before 10 replies. Don't use other emoticons and delete them.
"""
teaching_message =("""
Use the following pieces of context to answer the users question.
Given the following summaries of a long document and a question. 
If you don't know the answer, just say that "It's not mentioned in the book I read.", don't try to make up an answer.
Based on the following content, [Content] {content}
""")

## 파일접수
'''
from langchain_community.document_loaders import DirectoryLoader        #pip install -U langchain-community
#!! from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
#!! from langchain.vectorstores import Chroma   ## langchain_community.vectorstores import Chroma ##로 바꿔야 한다나 그렇다 함. 
#!! from langchain.embeddings.openai import OpenAIEmbeddings

#txt, pdf 읽기. 로딩, 스플릿, OpenAIEmbeddings으로 랭체인기반 벡터스토어 구축 Chroma DB에 저장. 
#!! raw_documents = PyPDFLoader("static/pat_test.pdf").load()      #raw_documents == Document(page_content= 'bla', metadata= {'source':'bla.txt'}), Document(page_content= 'bla', metadata= {'source':'bla.txt'}), ~
#!! text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#!! documents = text_splitter.split_documents(raw_documents)    
#!! db = Chroma.from_documents(documents, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)) #db=vector_store. 그냥DB 아닌 유사도 가진 DB 위해서 벡터화.
'''

client = OpenAI(api_key=OPENAI_API_KEY)
message = ''
history = []


def respond(message, history):  # 채팅봇의 응답을 처리하는 함수를 정의합니다. message는 내가 입력한 것. 
    modelName='gpt-3.5-turbo' #'gpt-4-turbo-2024-04-09'
    if any(word in message for word in ["자세", "상세", "천천", "길게"]):
        modelName='gpt-4-turbo'
        system_prompt = persona_message
        message = message + ' [****]'
        temperatureNum = 0.2
    elif any(word in message for word in ["정확", "확실", "책에서", "책자에서", "교과서에서", "백서에서"]):
        print("PDF")
        '''
        ## 1-1 랭체인방식
        #docs = db.similarity_search(message)   #query = message     #print(docs[0].page_content)
        ## 1-2 openAI방식
        #embedding_vector = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY).embed_query(message)     
        #docs = db.similarity_search_by_vector(embedding_vector)     #print(docs[0].page_content)    #print(len(embedding_vector))
        ## 2 k값 조절방식
        retriever = db.as_retriever(search_kwargs={"k": 6})
        docs = retriever.invoke(message)
        #docs = retriever.get_relevant_documents(message) #원본 chromaDB's deprecated in langchain-core 0.1.46
        #docs[0].page_content + docs[1].page_content
        system_prompt = persona_message + teaching_message.format(content=docs)
        message = message + ' [#]'
        temperatureNum = 0
        '''

    else:
        modelName='gpt-3.5-turbo'
        system_prompt = persona_message
        message = message
        temperatureNum = 0.4
    
    history_openai_format = [{"role": "system", "content": system_prompt}]
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})

    response = client.chat.completions.create(model = modelName, 
                                              messages = history_openai_format, 
                                              temperature = temperatureNum, 
                                              stream = True)

    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
              partial_message = partial_message + chunk.choices[0].delta.content
              # yield partial_message
    
    partial_message_withoutEmoji = partial_message.replace("('ω')", "").replace("(^ω^)", "").replace("('0')", "").replace("(TωT)", "").replace("(-_-+)", "").replace("(°ロ°)", "").replace(":)", "").replace("(-^ω^-)", "").replace("\\(^ω^)/", "").replace("\\", "") 
    partial_message_withoutEmoji = ''.join(c for c in partial_message_withoutEmoji if c <= '\uFFFF')
    partial_message_withoutEmoji = ''.join(c for c in partial_message_withoutEmoji if ord(c) < 0xD800 or (0xE000 <= ord(c) < 0x10000))     #surrogate pair 이모지 고려
    history.append((message, partial_message_withoutEmoji))
    
    #return "", history  # 원본 수정된 채팅 기록을 반환합니다.
    return partial_message, history  





from flask import Flask, jsonify, request, render_template, session
from flask_session import Session
app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'  # 세션을 파일 시스템에 저장합니다. 다른 옵션은 Redis, Memcached, filesystem 등이 있습니다.
Session(app)


@app.route("/")
def main():
    return render_template('index.html')
# def hello_world():
#    return "<p>Hello, World!</p>"

@app.route("/test")
def test_json():
    test_data = {"key": "value"}
    return jsonify(test_data)

@app.route('/send_fruit', methods=['POST'])
def send_fruit():
    data = request.get_json()
    fruit = data['fruit']
    fruit_name = f'Received fruit: {fruit}'
    print(fruit_name)
    return jsonify({'received_fruit': fruit})

@app.route('/send_transcript', methods=['POST'])
def send_transcript():
    data = request.get_json()
    transcript = data['textContent']    ##메시지만 추출
    transcript_name = f'(Python Received) textContent : {transcript}' #key-value 제시
    print(transcript_name)  #로그 체크
    #print(transcript)      ##활용

    
    '''
    with gr.Blocks() as demo:  # gr.Blocks()를 사용하여 인터페이스를 생성합니다.
        chatbot = gr.Chatbot(label="채팅창")  # '채팅창'이라는 레이블을 가진 채팅봇 컴포넌트를 생성합니다.
        msg = gr.Textbox(label="입력")  # '입력'이라는 레이블을 가진 텍스트박스를 생성합니다.
        clear = gr.Button("초기화")  # '초기화'라는 레이블을 가진 버튼을 생성합니다.

        msg.submit(respond, [msg, chatbot], [msg, chatbot])  # 텍스트박스에 메시지를 입력하고 제출하면 respond 함수가 호출되도록 합니다.
        clear.click(lambda: None, None, chatbot, queue=False)  # '초기화' 버튼을 클릭하면 채팅 기록을 초기화합니다.

    #demo.launch(share=True, debug=True)  # 인터페이스를 실행합니다. 실행하면 사용자는 '입력' 텍스트박스에 메시지를 작성하고 제출할 수 있으며, '초기화' 버튼을 통해 채팅 기록을 초기화 할 수 있습니다.
    demo.launch()
    '''

    if "세션 초기화" in transcript:
        history = []
        session['history'] = history
    elif "기억상실" in transcript:
        history = []
        session['history'] = history
    else:
        history = session.get('history', [])

    print("[history 1]= "+str(history))
    respondMsg, history = respond(transcript, history) 
    print("[respondMsg]= "+respondMsg)  #로그 체크
    print("[history 2]= "+str(history))  #로그 체크


    session['history'] = history

    return jsonify({'received_textContent': respondMsg})
    #return jsonify(history)


# main.py 경우, 활성화. 
if __name__ == '__main__':
    app.run('0.0.0.0', port=5001, debug=True)      #바로 실행. flask run --host=0.0.0.0