import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# LangChain + Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="Agentic Career Insights API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Parent's question about career outcomes and support")
    student_branch: Optional[str] = Field(None, description="Optional branch/major for context")
    region: Optional[str] = Field(None, description="Optional region/country context for salaries and employment")


class AskResponse(BaseModel):
    answer: str
    model: str
    meta: Dict[str, Any] = {}


@app.get("/")
def read_root():
    return {"message": "Agentic Career Insights API is running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Used",
        "database_url": None,
        "database_name": None,
        "connection_status": "N/A",
        "collections": [],
    }
    # Check environment variables for DB presence (informational)
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    # Check Gemini
    response["gemini_api_key"] = "✅ Set" if os.getenv("GOOGLE_API_KEY") else "❌ Not Set"
    return response


@app.post("/ask", response_model=AskResponse)
async def ask_insights(payload: AskRequest):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured on server")

    system_preamble = (
        "You are an agentic AI advisor for parents evaluating university programs. "
        "Your job is to provide clear, balanced insights about career outcomes, employment rates, "
        "median salaries, growth paths, and post‑graduation support services. "
        "Include practical takeaways, common job roles, relevant skills, and how the school's career services help. "
        "Use structured sections with concise bullets. Include estimates with ranges when exact numbers aren't available. "
        "If a branch/major is provided, tailor insights to that field. Be neutral and evidence‑minded."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_preamble),
        (
            "human",
            "Question: {question}\n"
            "Branch/Major (optional): {student_branch}\n"
            "Region (optional): {region}\n"
            "Return a helpful, parent‑friendly answer."
        ),
    ])

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, google_api_key=api_key)
    chain = prompt | llm | StrOutputParser()

    try:
        answer = await chain.ainvoke(
            {
                "question": payload.question.strip(),
                "student_branch": payload.student_branch or "",
                "region": payload.region or "",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)[:200]}")

    return AskResponse(answer=answer, model="gemini-1.5-pro", meta={"branch": payload.student_branch, "region": payload.region})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
