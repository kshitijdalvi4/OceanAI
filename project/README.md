# 🌊 OceanAI Document Generation Platform

> An AI-powered document authoring system using RAG (Retrieval Augmented Generation) and LLMs to intelligently generate, refine, and manage professional business documents.

---

## 🎯 Problem Statement

Creating professional business documents (Word docs, PowerPoint presentations) is time-consuming and requires:
- Hours of research and content creation
- Consistent formatting and structure
- Multiple revision cycles
- Context awareness across sections

### Our Solution
An intelligent document generation platform that:
1. **Automatically generates** structured business documents from topics
2. **Maintains context** across sections using RAG
3. **Enables iterative refinement** with natural language prompts
4. **Provides AI-powered outlines** to jumpstart document creation
5. **Exports ready-to-use** .docx and .pptx files

---

## 🏗️ Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                   (React + TypeScript + Tailwind)               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │Create Project│  │Generate Docs │  │ Refine & Chat│           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                   API ENDPOINTS                          │   │
│  │  POST /api/projects/         │  POST /generate/          │   │
│  │  POST /refine/               │  GET  /export/            │   │
│  │  POST /suggest-outline/      │  POST /chat/              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PROCESSING LAYER                            │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐    │
│  │Content Memory  │  │  LLM Engine    │  │  RAG Pipeline   │    │
│  │    Buffer      │  │  (Gemini 2.0)  │  │  (LangChain)    │    │
│  └────────────────┘  └────────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      STORAGE LAYER                              │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐    │
│  │  Vector DB     │  │   SQLite DB    │  │  Embeddings     │    │
│  │  (ChromaDB)    │  │  (SQLAlchemy)  │  │  (Gemini API)   │    │
│  └────────────────┘  └────────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### Component Breakdown

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | React 18, TypeScript, Tailwind CSS | Modern, responsive UI |
| **Backend** | FastAPI, Python 3.10+ | RESTful API server |
| **Authentication** | JWT + bcrypt | Secure user management |
| **LLM** | Google Gemini 2.0 Flash Lite | Text generation & analysis |
| **Embeddings** | Gemini text-embedding-004 | Semantic vector generation |
| **Vector DB** | ChromaDB | Context-aware memory storage |
| **Database** | SQLite + SQLAlchemy | Project & content management |
| **Document Export** | python-docx, python-pptx | .docx/.pptx generation |
| **RAG Framework** | LangChain | RAG orchestration |

---

## 🔄 RAG Pipeline Architecture

### What is RAG?
**Retrieval Augmented Generation** combines information retrieval with LLM generation to provide context-aware, consistent document generation.

### Our RAG Implementation

```
┌──────────────────────────────────────────────────────────────┐
│                    Content Generation Flow                   │
└──────────────────────────────────────────────────────────────┘
                              ▼
        ┌─────────────────────────────────────────┐
        │  1. User Creates Project                │
        │     - Title, Topic, Document Type       │
        └─────────────────────────────────────────┘
                              ▼
        ┌─────────────────────────────────────────┐
        │  2. AI Suggests Outline (Optional)      │
        │     - 6-8 sections for Word             │
        │     - 8-10 slides for PowerPoint        │
        └─────────────────────────────────────────┘
                              ▼
        ┌─────────────────────────────────────────┐
        │  3. Content Generation (Section-wise)   │
        │     ┌─────────────────────────────────┐ │
        │     │ a. Query ContentMemoryBuffer    │ │
        │     │    (Get previous sections)      │ │
        │     └─────────────────────────────────┘ │
        │     ┌─────────────────────────────────┐ │
        │     │ b. Generate content with context│ │
        │     │    using Gemini LLM             │ │
        │     └─────────────────────────────────┘ │
        │     ┌─────────────────────────────────┐ │
        │     │ c. Store in ContentMemoryBuffer │ │
        │     │    (ChromaDB vector storage)    │ │
        │     └─────────────────────────────────┘ │
        └─────────────────────────────────────────┘
                              ▼
        ┌─────────────────────────────────────────┐
        │  4. Iterative Refinement                │
        │     - User provides natural language    │
        │       refinement prompt                 │
        │     - RAG retrieves relevant context    │
        │     - Generates refined version         │
        │     - Maintains version history         │
        └─────────────────────────────────────────┘
                              ▼
        ┌─────────────────────────────────────────┐
        │  5. Export Document                     │
        │     - Combine all sections              │
        │     - Apply formatting                  │
        │     - Generate .docx or .pptx           │
        └─────────────────────────────────────────┘
```

### ContentMemoryBuffer: The Core Innovation

The **ContentMemoryBuffer** is a custom RAG implementation that maintains project context:

```python
class ContentMemoryBuffer:
    """
    Manages context and memory for document generation using RAG.
    Stores all generated content in ChromaDB for context-aware refinements.
    """
    
    def __init__(self, project_id: int):
        self.project_id = project_id
        self.collection_name = f"project_{project_id}"
        self.collection = chroma_client.get_or_create_collection(
            name=self.collection_name
        )
    
    def add_content(self, section_id: str, title: str, 
                    content: str, metadata: Dict):
        """Store generated content with embeddings"""
        embedding = embeddings.embed_query(content)
        self.collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[f"{project_id}_{section_id}_{uuid}"]
        )
    
    def query_context(self, query: str, n_results: int = 3):
        """Retrieve relevant context for refinement"""
        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
```

**Benefits:**
- ✅ Maintains consistency across document sections
- ✅ Context-aware refinements (knows what was written before)
- ✅ Enables intelligent Q&A about document content
- ✅ Version control with semantic search

---

## ✨ Features

### 🎯 Core Functionality

- **📝 Project Management**
  - Create unlimited projects
  - Choose between Word (.docx) or PowerPoint (.pptx)
  - Set topic and custom outline
  - Track project status and history

- **🤖 AI-Powered Outline Generation**
  - One-click AI outline suggestions
  - 6-8 sections for Word documents
  - 8-10 slides for PowerPoint presentations
  - Customizable after generation

- **📄 Intelligent Content Generation**
  - Context-aware section generation
  - Maintains consistency across document
  - Professional writing style
  - 200-400 words per section

- **🔄 Iterative Refinement**
  - Natural language refinement prompts
  - "Make more formal", "Add bullet points", etc.
  - RAG-powered context retrieval
  - Version history tracking

- **💬 Document Q&A (Chat)**
  - Ask questions about your document
  - RAG retrieves relevant sections
  - Natural language responses
  - Context-aware answers

- **📊 Analytics Dashboard**
  - Project completion percentage
  - Total word count
  - Refinement statistics
  - Feedback tracking (likes/dislikes)

- **📥 Export & Download**
  - Generate .docx for Word documents
  - Generate .pptx for PowerPoint
  - Formatted and ready to use
  - Includes title page and sections

---

## 🛠️ Tech Stack

### Backend
```python
FastAPI           # Web framework
Google Gemini     # LLM (Gemini 2.0 Flash Lite)
LangChain         # RAG framework
ChromaDB          # Vector database
SQLAlchemy        # ORM for SQL database
python-docx       # Word document generation
python-pptx       # PowerPoint generation
JWT + bcrypt      # Authentication
Pydantic          # Data validation
```

### Frontend
```typescript
React 18          # UI framework
TypeScript        # Type safety
Tailwind CSS      # Styling
Lucide Icons      # Icon library
Vite              # Build tool
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Google Gemini API Key

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/oceanai-document-generator.git
cd oceanai-document-generator
```

#### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=your-secret-key-change-in-production
DATABASE_URL=sqlite:///./oceanai.db
CHROMA_PATH=./chroma_db
EOF

# Run backend
python main.py
```

Backend will run at: `http://localhost:8000`

#### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run frontend
npm run dev
```

Frontend will run at: `http://localhost:5173`

---

## 🤖 LLM Prompts & Prompt Engineering

### 1. AI Outline Generation Prompt

```python
OUTLINE_PROMPT = """
You are a professional document outline generator.

Create a detailed outline for a {document_type} document about: {topic}

You MUST create exactly {count} {section_type}. Each section must have a title and description.

Return ONLY a valid JSON array with exactly {count} items, no markdown, no explanation:
[
  {"title": "Section Title 1", "description": "What this section covers"},
  {"title": "Section Title 2", "description": "What this section covers"},
  ...
]

Generate exactly {count} {section_type} now:
"""
```

**Prompt Engineering Techniques:**
- ✅ Clear instruction on output format (JSON only)
- ✅ Exact count specification
- ✅ No markdown constraint to avoid parsing issues
- ✅ Structured output for reliable parsing

### 2. Content Generation Prompt

```python
GENERATION_PROMPT = """
You are a professional business document writer specializing in creating 
high-quality, well-structured content.

Generate professional content for the following section.

DOCUMENT TOPIC: {topic}
SECTION TITLE: {title}
SECTION TYPE: {section_type}

Previous Context:
{context}

Requirements:
- Write clear, professional content (200-400 words)
- Use proper paragraphs and structure
- Be specific and informative
- Maintain consistency with previous sections
- Write in plain text without markdown formatting

Content:
"""
```

**Prompt Engineering Techniques:**
- ✅ Context injection (previous sections via RAG)
- ✅ Word count guidance (200-400 words)
- ✅ Consistency enforcement
- ✅ Plain text requirement for reliable formatting

### 3. Content Refinement Prompt

```python
REFINEMENT_PROMPT = """
You are refining a section of a business document.

Previous related content:
{context}

Current content to refine:
{original}

User's refinement request:
{refinement_prompt}

Provide the refined content below. Write in plain text without markdown formatting.

Refined content:
"""
```

**Prompt Engineering Techniques:**
- ✅ RAG context inclusion (related sections)
- ✅ User intent preservation
- ✅ Plain text enforcement
- ✅ Context-aware refinement

### 4. Document Q&A Prompt

```python
QA_PROMPT = """
You are analyzing a business document project.

Project Topic: {topic}
Document Type: {document_type}

Relevant sections:
{context}

User Question: {question}

Provide a helpful answer based on the document content. 
Write in plain text without markdown formatting.

Answer:
"""
```

**Prompt Engineering Techniques:**
- ✅ Project context setting
- ✅ RAG-retrieved relevant sections
- ✅ Instruction to stay grounded in context
- ✅ Natural language response

---

## 📡 API Reference

### Authentication

#### Register User
```http
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword",
  "full_name": "John Doe"
}
```

#### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword"
}
```

### Projects

#### Create Project
```http
POST /api/projects
Authorization: Bearer {token}
Content-Type: application/json

{
  "title": "Q4 Business Plan",
  "document_type": "docx",
  "main_topic": "Strategic planning for Q4 2025"
}
```

#### Get Project Details
```http
GET /api/projects/{project_id}
Authorization: Bearer {token}
```

#### Update Project Configuration
```http
PUT /api/projects/{project_id}/config
Authorization: Bearer {token}
Content-Type: application/json

{
  "configuration": {
    "sections": [
      {"title": "Introduction", "description": "Overview"},
      {"title": "Analysis", "description": "Market analysis"}
    ]
  }
}
```

### AI Features

#### Generate AI Outline
```http
POST /api/suggest-outline-direct
Authorization: Bearer {token}
Content-Type: application/json

{
  "topic": "Digital transformation strategy",
  "document_type": "docx"
}
```

#### Generate Content
```http
POST /api/projects/{project_id}/generate
Authorization: Bearer {token}
Content-Type: application/json

{
  "section_id": "section-1"  // Optional: generate specific section
}
```

#### Refine Content
```http
POST /api/projects/{project_id}/refine
Authorization: Bearer {token}
Content-Type: application/json

{
  "section_id": "section-1",
  "prompt": "Make this more formal and add bullet points"
}
```

#### Chat with Document
```http
POST /api/projects/{project_id}/chat
Authorization: Bearer {token}
Content-Type: application/json

{
  "project_id": 1,
  "question": "What are the key points in the introduction?"
}
```

### Export

#### Export Document
```http
GET /api/projects/{project_id}/export
Authorization: Bearer {token}
```

Returns: Binary file (.docx or .pptx)

---

## 📁 Project Structure

```
oceanai-document-generator/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Environment variables
│   ├── .env.example           # Environment template
│   ├── oceanai.db             # SQLite database (gitignored)
│   └── chroma_db/             # Vector database (gitignored)
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Main React component
│   │   ├── main.tsx           # Entry point
│   │   └── index.css          # Tailwind CSS
│   ├── public/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── postcss.config.js
│
├── docs/
│   ├── images/                # Screenshots
│   └── README.md              # Additional documentation
│
├── .gitignore
├── LICENSE
└── README.md                  # This file
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "AI Suggest Outline returns only 2 suggestions"

**Problem:** Gemini API response isn't being parsed correctly.

**Solution:**
```python
# Backend automatically fills to 6/8 sections if AI returns fewer
# Check backend console for debug logs:
# 🤖 Raw Gemini response: ...
# 🧹 Cleaned response: ...
# ✅ Successfully parsed X suggestions
```

#### 2. "Token expired" error

**Problem:** JWT token has expired after 7 days.

**Solution:** Log out and log back in to get a new token.

#### 3. Content generation fails

**Problem:** Gemini API rate limit or connection issue.

**Solution:**
- Check GEMINI_API_KEY in .env
- Wait 1 minute and retry
- Check backend console for specific error

#### 4. Export fails with "No content to export"

**Problem:** Content hasn't been generated yet.

**Solution:** Click "Generate Content" before exporting.

---

## 🔒 Security Features

- **JWT Authentication:** Secure token-based auth with 7-day expiry
- **Password Hashing:** bcrypt with salt for secure password storage
- **CORS Protection:** Configured allowed origins
- **Input Validation:** Pydantic models for all API inputs
- **SQL Injection Protection:** SQLAlchemy ORM parameterized queries

---

## 🎨 UI/UX Features

- **Responsive Design:** Works on desktop, tablet, and mobile
- **Real-time Feedback:** Loading states for all async operations
- **Error Handling:** User-friendly error messages
- **Progress Indicators:** Spinners and completion percentages
- **Gradient UI:** Modern blue-purple gradient theme
- **Version Display:** Shows version history for each section

---

## 📊 Performance Optimizations

- **Chunked Generation:** Generates sections individually to avoid timeouts
- **Vector Caching:** ChromaDB caches embeddings for fast retrieval
- **Rate Limiting:** Built-in retry logic for API failures
- **Lazy Loading:** Projects loaded on-demand
- **Optimistic Updates:** UI updates before backend confirmation

---

## 🙏 Acknowledgments

- **Google Gemini** for powerful LLM capabilities and embeddings
- **LangChain** for RAG framework
- **ChromaDB** for efficient vector storage
- **FastAPI** for elegant API design
- **React** community for UI components
- **Tailwind CSS** for beautiful styling

---

## 📞 Contact & Support

- **Developer:** Your Name
- **Email:** your.email@example.com
- **GitHub:** [@yourusername](https://github.com/kshitijdalvi4)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ using Google Gemini and LangChain**

*Last updated: November 2025*
