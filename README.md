# SupportIQ-Ticket-Classification

SupportIQ: Hierarchical AI Ticket Triage System SupportIQ is an intelligent helpdesk assistant that utilizes a three-tier architecture to resolve technical issues efficiently. By combining traditional information retrieval with modern Generative AI, the system ensures fast response times for common queries while providing deep technical analysis for complex incidents.

🚀 Key Features Tier 1 (L1) - FAQ Retrieval: Uses TF-IDF Vectorization and Cosine Similarity to provide instant answers to common administrative questions.

Tier 2 (L2) - Knowledge Base RAG: Leverages FAISS (Facebook AI Similarity Search) and Sentence-Transformers (all-MiniLM-L6-v2) to perform semantic searches across technical documentation.

Tier 3 (L3) - Generative Escalation: Integrates Google Gemini 1.5 Flash to analyze unique or complex technical problems that are not present in the local database.

🛠️ Tech Stack Backend: Python, Flask

Machine Learning: Scikit-learn, Sentence-Transformers

Vector Database: FAISS (CPU)

LLM API: Google Generative AI (Gemini)

Frontend: HTML5, CSS3 (Modern UI), JavaScript (Async/Await)

🧠 Logic Flow: The system processes user queries through a fallback pipeline to maximize efficiency and minimize API costs:

Check FAQ (L1): If a high-similarity match (
>
0.7
) is found in the FAQ data, return it.
Check RAG (L2): If no FAQ match, search the vector database. If a relevant context is found, return it.
Escalate (L3): If no local match is found, the query is sent to Gemini 1.5 Flash for a generative solution.
