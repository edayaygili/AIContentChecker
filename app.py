
import streamlit as st
from sentence_transformers import SentenceTransformer, util

st.title("🧠 AI-Based Content Relevance Checker")

st.write("Bu sistem, bir metnin verilen konuyla ne kadar alakalı olduğunu AI yardımıyla belirler.")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

topic = st.text_input("🔍 Topic (örnek: technology, fashion, politics...)")
text = st.text_area("📝 Content (değerlendirmek istediğin metni gir)")

if st.button("Check Relevance"):
    if topic.strip() == "" or text.strip() == "":
        st.warning("Lütfen hem topic hem de content girin.")
    else:
        topic_emb = model.encode(topic, convert_to_tensor=True)
        text_emb = model.encode(text, convert_to_tensor=True)

        score = util.cos_sim(topic_emb, text_emb).item()

        st.write(f"📊 Relevance Score: **{score:.2f}**")

        if score > 0.5:
            st.success("✅ Bu metin **relevant** (ilişkili).")
        else:
            st.error("❌ Bu metin **non-relevant** (ilişkisiz).")
