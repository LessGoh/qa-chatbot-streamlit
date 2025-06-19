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
import hashlib

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

# Константы
SHARED_INDEX_NAME = "qa-chatbot-shared"  # Один индекс для всех документов

# Инициализация session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_document_id' not in st.session_state:
    st.session_state.current_document_id = None

def get_document_id(filename, content):
    """Создает уникальный ID для документа"""
    content_hash = hashlib.md5(content).hexdigest()[:8]
    return f"{filename}_{content_hash}"

@st.cache_resource
def setup_pinecone_shared(api_key):
    """Настройка общего индекса Pinecone (один для всех документов)"""
    try:
        pc = Pinecone(api_key=api_key)
        
        # Проверяем существующие индексы
        existing_indexes = [idx['name'] for idx in pc.list_indexes().indexes]
        
        if SHARED_INDEX_NAME not in existing_indexes:
            with st.spinner("Создаю общий индекс в Pinecone... (это может занять минуту)"):
                pc.create_index(
                    name=SHARED_INDEX_NAME,
                    dimension=1536,
                    metric='cosine',
                    spec=ServerlessSpec(cloud='aws', region='us-east-1')
                )
                
                # Ждем готовности индекса
                max_wait = 60  # максимум 60 секунд
                waited = 0
                while waited < max_wait:
                    try:
                        status = pc.describe_index(SHARED_INDEX_NAME).status
                        if status['ready']:
                            break
                    except:
                        pass
                    time.sleep(2)
                    waited += 2
        
        return pc.Index(SHARED_INDEX_NAME)
    except Exception as e:
        st.error(f"Ошибка настройки Pinecone: {e}")
        return None

def clear_document_from_index(vectorstore, document_id):
    """Удаляет предыдущий документ из индекса"""
    try:
        # Получаем все векторы с нашим document_id
        index = vectorstore._index
        
        # Используем простой способ - получаем статистику
        stats = index.describe_index_stats()
        
        # Если индекс не пустой, очищаем его
        if stats.total_vector_count > 0:
            # Для простоты, будем очищать весь индекс при загрузке нового документа
            # В продакшене лучше использовать namespaces или metadata фильтры
            index.delete(delete_all=True)
            st.info("🧹 Очистил предыдущий документ из базы данных")
            
    except Exception as e:
        st.warning(f"Не удалось очистить предыдущий документ: {e}")

def process_document(uploaded_file, openai_key, pinecone_key):
    """Обработка документа с использованием общего индекса"""
    try:
        # Проверка API ключей
        if not openai_key or not pinecone_key:
            return False, "Необходимо указать API ключи!"
        
        # Создание уникального ID для документа
        file_content = uploaded_file.getvalue()
        document_id = get_document_id(uploaded_file.name, file_content)
        
        # Создание временного файла
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        # Прогресс бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Загрузка PDF
        status_text.text("📄 Загружаю PDF...")
        progress_bar.progress(10)
        
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
        
        if not pages:
            return False, "Не удалось извлечь текст из PDF"
        
        # Разбиение на чанки
        status_text.text("✂️ Разбиваю на части...")
        progress_bar.progress(30)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(pages)
        
        # Добавляем метаданные с document_id
        for split in splits:
            split.metadata['document_id'] = document_id
            split.metadata['source_file'] = uploaded_file.name
        
        # Настройка Pinecone
        status_text.text("🔍 Настраиваю векторную базу...")
        progress_bar.progress(50)
        
        index = setup_pinecone_shared(pinecone_key)
        if not index:
            return False, "Ошибка настройки Pinecone"
        
        # Создание embeddings
        status_text.text("🧠 Создаю векторные представления...")
        progress_bar.progress(70)
        
        embeddings = OpenAIEmbeddings(api_key=openai_key)
        vectorstore = PineconeVectorStore(index, embeddings, "text")
        
        # Очищаем предыдущий документ если есть
        if st.session_state.current_document_id:
            status_text.text("🧹 Очищаю предыдущий документ...")
            clear_document_from_index(vectorstore, st.session_state.current_document_id)
        
        # Добавление документов
        status_text.text("💾 Сохраняю в базу данных...")
        progress_bar.progress(90)
        
        vectorstore.add_documents(splits)
        
        progress_bar.progress(100)
        status_text.text("✅ Готово!")
        
        # Сохранение в session state
        st.session_state.vectorstore = vectorstore
        st.session_state.documents_loaded = True
        st.session_state.current_document_id = document_id
        st.session_state.messages = []  # Очищаем историю чата для нового документа
        
        # Очистка
        os.unlink(tmp_file_path)
        
        return True, f"✅ Документ '{uploaded_file.name}' успешно загружен!\n📊 Обработано {len(splits)} частей из {len(pages)} страниц"
        
    except Exception as e:
        error_msg = str(e)
        if "FORBIDDEN" in error_msg and "max serverless indexes" in error_msg:
            return False, "❌ Превышен лимит бесплатных индексов Pinecone!\n\n💡 Решения:\n1. Удалите старые индексы в Pinecone Console\n2. Или обновите план Pinecone\n3. Или используйте другой API ключ"
        return False, f"❌ Ошибка: {error_msg}"

def answer_question(question, openai_key):
    """Ответ на вопрос с учетом текущего документа"""
    try:
        if not st.session_state.vectorstore:
            return "Сначала загрузите документ"
        
        # Создаем retriever с фильтром по текущему документу
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={
                "k": 3,
                "filter": {"document_id": st.session_state.current_document_id}
            }
        )
        
        prompt = PromptTemplate.from_template(
            """Ты - эксперт-аналитик. Ответь на вопрос на основе предоставленного контекста из загруженного документа.
            
Контекст из документа:
{context}

Вопрос: {question}

Инструкции:
1. Отвечай только на основе предоставленного контекста
2. Если в контексте нет информации для ответа, так и скажи
3. Структурируй ответ ясно и логично
4. Будь конкретным и полезным

Ответ:"""
        )
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=openai_key
        )
        
        def format_docs(docs):
            if not docs:
                return "Релевантная информация не найдена в документе."
            return "\n\n".join([f"Фрагмент {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
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
    
    # Информация об индексе
    st.info(f"🏠 Используется общий индекс: `{SHARED_INDEX_NAME}`\n\nЭто экономит вашу квоту Pinecone!")
    
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
            
            **Оптимизация для бесплатного плана:**
            - 🏠 Использует один общий индекс (экономит квоту)
            - 🧹 Автоматически очищает старые документы
            - 💾 Умное управление памятью
            """)

# Футер
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <small>
            🤖 QA ChatBot | Powered by OpenAI & Pinecone<br>
            Made with ❤️ using Streamlit<br>
            <em>Оптимизировано для бесплатного плана Pinecone</em>
        </small>
    </div>
    """, unsafe_allow_html=True)

# Скрытая информация для отладки
if st.checkbox("🔧 Debug Info", value=False):
    st.json({
        "documents_loaded": st.session_state.documents_loaded,
        "vectorstore_exists": st.session_state.vectorstore is not None,
        "messages_count": len(st.session_state.messages),
        "openai_key_set": bool(openai_key),
        "pinecone_key_set": bool(pinecone_key),
        "current_document_id": st.session_state.current_document_id,
        "shared_index_name": SHARED_INDEX_NAME
    })
