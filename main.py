import streamlit as st
import os
import tempfile
from civic_rag.backend import database
from civic_rag.backend import rag

st.set_page_config(page_title="Civic RAG for Nepal", layout="wide")

database.init_db()

st.title("🇳🇵 Civic Protest Guidance & Insights")

tab1, tab2 = st.tabs(["Ask a Question", "Upload PDF & Metadata"])

with tab2:
	st.header("Upload Historical Protest PDF")
	uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
	country = st.text_input("Country", max_chars=64)
	year = st.text_input("Year", max_chars=8)
	topic = st.text_input("Topic", max_chars=128)
	if uploaded_file and country and year and topic:
		if st.button("Ingest PDF"):
			with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
				tmp_file.write(uploaded_file.read())
				tmp_path = tmp_file.name
			metadata = {"country": country, "year": year, "topic": topic}
			docs = rag.ingest_pdf(tmp_path, metadata)
			rag.build_vector_store(docs)
			st.success(f"PDF for {country} ({year}) on '{topic}' ingested and indexed.")

with tab1:
	st.header("Ask a Question about Protests")
	question = st.text_area("Your Question", max_chars=500)
	country_filter = st.text_input("Filter by Country (optional)")
	topic_filter = st.text_input("Filter by Topic (optional)")
	if st.button("Get Answer") and question:
		answer = rag.answer_query(question)
		database.save_query(question, answer, country=country_filter or None, topic=topic_filter or None)
		st.markdown(f"**Answer:** {answer}")

	st.subheader("Past Citizen Queries")
	queries = database.get_queries(country=country_filter or None, topic=topic_filter or None, limit=20)
	if queries:
		for q in queries:
			st.markdown(f"- **Q:** {q['question']}\n    - **A:** {q['answer']}\n    - *{q['timestamp']} | {q['country']} | {q['topic']}*")
	else:
		st.info("No queries found for the selected filters.")

