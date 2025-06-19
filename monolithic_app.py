import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time

# Конфигурация
st.set_page_config(
    page_title="QA ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Инициализация session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'messages' not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def setup_pinecone(api_key):
    """Настройка Pinecone с кешированием"""
    try:
        pc = Pinecone(api_key=api_key)
        index_name = "qa-chatbot-streamlit"
        
        existing_indexes = [idx['name'] for idx in pc.list_indexes().indexes]
        
        if index_name not in existing_indexes:
            with st.spinner("Создаю индекс в Pinecone..."):
                pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                
                while not pc.describe_index(index_name).status['ready']:
                    time.sleep(1)
        
        return pc.Index(index_name)
    except Exception as e:
        st.error(f"Ошибка настройки Pinecone: {e}")
        return None

def process_document(uploaded_file, openai_key, pinecone_key):
    """Обработка документа"""
    try:
        # Проверка API ключей
        if not openai_key or not pinecone_key:
            return False, "Необходимо указать API ключи!"
        
        # Создание временного файла
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Прогресс бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Загрузка PDF
        status_text.text("Загружаю PDF...")
        progress_bar.progress(20)
        
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
        
        if not pages:
            return False, "Не удалось извлечь текст из PDF"
        
        # Разбиение на чанки
        status_text.text("Разбиваю на части...")
        progress_bar.progress(40)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(pages)
        
        # Настройка Pinecone
        status_text.text("Настраиваю векторную базу...")
        progress_bar.progress(60)
        
        index = setup_pinecone(pinecone_key)
        if not index:
            return False, "Ошибка настройки Pinecone"
        
        # Создание embeddings
        status_text.text("Создаю векторные представления...")
        progress_bar.progress(80)
        
        embeddings = OpenAIEmbeddings(api_key=openai_key)
        vectorstore = PineconeVectorStore(index, embeddings, "text")
        
        # Добавление документов
        status_text.text("Сохраняю в базу данных...")
        vectorstore.add_documents(splits)
        
        progress_bar.progress(100)
        status_text.text("Готово!")
        
        # Сохранение в session state
        st.session_state.vectorstore = vectorstore
        st.session_state.documents_loaded = True
        
        # Очистка
        os.unlink(tmp_file_path)
        
        return True, f"✅ Успешно обработано {len(splits)} частей из {len(pages)} страниц"
        
    except Exception as e:
        return False, f"❌ Ошибка: {str(e)}"

def answer_question(question, openai_key):
    """Ответ на вопрос"""
    try:
        if not st.session_state.vectorstore:
            return "Сначала загрузите документ"
        
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        prompt = PromptTemplate.from_template(
            """Ты - эксперт-аналитик. Ответь на вопрос на основе предоставленного контекста.
            
Контекст:
{context}

Вопрос: {question}

Инструкции:
1. Отвечай только на основе предоставленного контекста
2. Если в контексте нет информации для ответа, так и скажи
3. Структурируй ответ ясно и логично
4. Укажи источник информации, если возможно

Ответ:"""
        )
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=openai_key
        )
        
        def format_docs(docs):
            return "\n\n".join([f"Источник {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
        )
        
        response = rag_chain.invoke(question)
        return response.content
        
    except Exception as e:
        return f"❌ Ошибка при ответе: {str(e)}"

# Заголовок
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🤖 QA ChatBot</h1>
    <p style="color: white; margin: 0; opacity: 0.8;">Загрузите PDF и задавайте вопросы по содержанию</p>
</div>
""", unsafe_allow_html=True)

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # API ключи
    st.subheader("🔑 API Ключи")
    
    # Получаем ключи из secrets или пользовательского ввода
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, 'secrets') else "",
        help="Ваш API ключ от OpenAI"
    )
    
    pinecone_key = st.text_input(
        "Pinecone API Key",
        type="password",
        value=st.secrets.get("PINECONE_API_KEY", "") if hasattr(st, 'secrets') else "",
        help="Ваш API ключ от Pinecone"
    )
    
    # Проверка ключей
    keys_status = st.empty()
    if openai_key and pinecone_key:
        keys_status.success("✅ API ключи установлены")
    else:
        keys_status.warning("⚠️ Необходимо указать API ключи")
    
    st.divider()
    
    # Загрузка файла
    st.subheader("📄 Загрузка документа")
    
    uploaded_file = st.file_uploader(
        "Выберите PDF файл",
        type=['pdf'],
        help="Максимальный размер: 10MB"
    )
    
    if uploaded_file:
        st.info(f"📄 Файл: {uploaded_file.name}")
        st.info(f"📊 Размер: {uploaded_file.size / 1024:.1f} KB")
        
        if st.button("🚀 Обработать документ", key="process_btn"):
            if not openai_key or not pinecone_key:
                st.error("❌ Необходимо указать API ключи!")
            else:
                with st.spinner("Обрабатываю документ..."):
                    success, message = process_document(uploaded_file, openai_key, pinecone_key)
                    if success:
                        st.success(message)
                        st.balloons()
                    else:
                        st.error(message)
    
    st.divider()
    
    # Статистика
    if st.session_state.documents_loaded:
        st.subheader("📊 Статистика")
        st.metric("Вопросов задано", len([m for m in st.session_state.messages if m["role"] == "user"]))
        
        if st.button("🗑️ Очистить историю"):
            st.session_state.messages = []
            st.rerun()

# Основная область
if st.session_state.documents_loaded:
    st.markdown("""
    <div class="success-box">
        <h3 style="color: #155724; margin: 0;">✅ Документ готов к работе!</h3>
        <p style="color: #155724; margin: 0;">Задавайте вопросы в чате ниже</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Чат интерфейс
    st.subheader("💬 Чат с документом")
    
    # Контейнер для сообщений
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Поле ввода
    if prompt := st.chat_input("Задайте вопрос по документу..."):
        # Добавляем вопрос пользователя
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Получаем ответ
        with st.chat_message("assistant"):
            with st.spinner("🤔 Думаю..."):
                response = answer_question(prompt, openai_key)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

else:
    # Приветственная страница
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>👋 Добро пожаловать!</h2>
            <p>Для начала работы:</p>
            <ol style="text-align: left;">
                <li>Введите API ключи в боковой панели</li>
                <li>Загрузите PDF документ</li>
                <li>Нажмите "Обработать документ"</li>
                <li>Задавайте вопросы по содержанию!</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        # Пример использования
        with st.expander("💡 Примеры вопросов"):
            st.markdown("""
            - "О чем этот документ?"
            - "Какие основные выводы?"
            - "Что говорится о [конкретной теме]?"
            - "Перечисли основные пункты"
            - "Какие рекомендации приводятся?"
            """)
        
        # Информация о системе
        with st.expander("ℹ️ Как это работает"):
            st.markdown("""
            **QA ChatBot** использует современные AI технологии:
            
            1. **Обработка документов**: Ваш PDF разбивается на логические части
            2. **Векторизация**: Каждая часть преобразуется в числовое представление
            3. **Поиск**: При вопросе система находит наиболее релевантные части
            4. **Генерация ответа**: AI формирует ответ на основе найденной информации
            
            **Технологии:**
            - 🤖 OpenAI GPT для генерации ответов
            - 🔍 Pinecone для векторного поиска
            - 📄 LangChain для обработки документов
            """)

# Футер
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <small>
            🤖 QA ChatBot | Powered by OpenAI & Pinecone<br>
            Made with ❤️ using Streamlit
        </small>
    </div>
    """, unsafe_allow_html=True)

# Скрытая информация для отладки (только в development)
if st.checkbox("🔧 Debug Info", value=False):
    st.json({
        "documents_loaded": st.session_state.documents_loaded,
        "vectorstore_exists": st.session_state.vectorstore is not None,
        "messages_count": len(st.session_state.messages),
        "openai_key_set": bool(openai_key),
        "pinecone_key_set": bool(pinecone_key)
    })
