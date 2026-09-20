from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from google.cloud.firestore_v1.async_client import AsyncClient

from app.api.dependencies import get_current_user
from app.billing.entitlements import lookup_active_tier
from app.database import get_db
from app.models.prompts import PromptCreate, PromptDB, PromptResponse
from app.models.users import UserDB
from app.repositories.prompts import PromptRepository


async def verify_prompt_library_access(
    user: UserDB = Depends(get_current_user),
    db: AsyncClient = Depends(get_db),
) -> None:
    tier = await lookup_active_tier(db, str(user.id))
    if tier != "premium":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Prompt Library is exclusively available on the Premium plan.",
        )


router = APIRouter(
    prefix="/prompts",
    tags=["Prompts"],
    dependencies=[Depends(verify_prompt_library_access)],
)

BUILTIN_PROMPTS = [
    {
        "id": "code-refactor",
        "title": "Refactor & Clean Code",
        "description": "Improve code quality, modularity, and adherence to clean code principles.",
        "category": "engineering",
        "prompt": "Review and refactor the following code to improve readability, maintainability, and performance. Follow clean code principles, remove redundancies, and ensure proper typing and error handling. Do not change the core business logic.\n\nCode:",
        "tags": ["Refactoring", "Clean Code"],
    },
    {
        "id": "code-unit-tests",
        "title": "Generate Unit Tests",
        "description": "Write comprehensive unit tests covering edge cases and happy paths.",
        "category": "engineering",
        "prompt": "Write comprehensive unit tests for the following code. Use the standard testing framework for the language. Cover the happy path, common edge cases, and potential failure modes. Include mock setups if necessary.\n\nCode:",
        "tags": ["Testing", "TDD"],
    },
    {
        "id": "code-explain",
        "title": "Explain Complex Code",
        "description": "Break down complex logic into easy-to-understand explanations.",
        "category": "engineering",
        "prompt": "Explain what this code does in plain, easy-to-understand language. Break down complex algorithms, identify the core logic, and explain the inputs and outputs. Assume I am a junior developer.\n\nCode:",
        "tags": ["Learning", "Explanation"],
    },
    {
        "id": "write-cold-email",
        "title": "Cold Outreach Email",
        "description": "Write a high-converting, concise B2B cold email.",
        "category": "writing",
        "prompt": "Write a concise, high-converting B2B cold email based on the following context. Focus on a strong hook, clear value proposition, and a low-friction call to action. Keep it under 150 words.\n\nContext:",
        "tags": ["Sales", "Email"],
    },
    {
        "id": "write-marketing-copy",
        "title": "Landing Page Copy",
        "description": "Persuasive copy for hero sections, features, and CTAs.",
        "category": "writing",
        "prompt": "Generate persuasive landing page copy for this product. Include a strong hero headline, subheadline, 3 key feature/benefit bullet points, and a compelling Call to Action (CTA).\n\nProduct details:",
        "tags": ["Marketing", "Copywriting"],
    },
    {
        "id": "write-tech-docs",
        "title": "Technical Documentation & README",
        "description": "Draft clear developer documentation, guides, and setup specs.",
        "category": "writing",
        "prompt": "Draft clear, professional developer documentation for this component or system. Include an overview, installation steps, usage examples, configuration parameters, and caveats.\n\nProject details:",
        "tags": ["Docs", "Markdown"],
    },
    {
        "id": "write-tone-refiner",
        "title": "Grammar & Tone Refiner",
        "description": "Polish draft for professional clarity and authoritative cadence.",
        "category": "writing",
        "prompt": "Refine the grammar, cadence, and tone of this draft to sound polished, authoritative, and engaging while preserving the original intent and voice.\n\nDraft:",
        "tags": ["Editing", "Grammar"],
    },
    {
        "id": "prod-prd-generator",
        "title": "Product Requirements Document (PRD)",
        "description": "Structured PRD with user stories, acceptance criteria, and metrics.",
        "category": "product",
        "prompt": "Generate a comprehensive Product Requirements Document (PRD) for this feature. Include problem statement, user personas, functional requirements, user stories with acceptance criteria, metrics, and non-goals.\n\nFeature idea:",
        "tags": ["PRD", "Product", "Specs"],
    },
    {
        "id": "prod-competitor-teardown",
        "title": "Competitor & Feature Teardown",
        "description": "Analyze market landscape, differentiation, and competitive moat.",
        "category": "product",
        "prompt": "Conduct a strategic competitive teardown for this product concept. Outline key competitors, their strengths/weaknesses, unique differentiation angles, and recommended product strategy.\n\nConcept:",
        "tags": ["Strategy", "Competition"],
    },
    {
        "id": "prod-feedback-synthesizer",
        "title": "User Feedback Synthesizer",
        "description": "Group feedback into thematic clusters, quotes, and priorities.",
        "category": "product",
        "prompt": "Analyze the following customer feedback or user interview notes. Group them into thematic clusters, extract representative quotes, identify major friction points, and rank actionable recommendations.\n\nUser feedback:",
        "tags": ["User Research", "Feedback"],
    },
    {
        "id": "think-first-principles",
        "title": "First-Principles Deconstructor",
        "description": "Break complex problems down to atomic truths and rebuild solutions.",
        "category": "thinking",
        "prompt": "Deconstruct this problem down to its most fundamental, indisputable truths. Rebuild an optimal solution from first principles without relying on convention or legacy assumptions.\n\nProblem:",
        "tags": ["Reasoning", "First Principles"],
    },
    {
        "id": "think-socratic-tutor",
        "title": "Socratic Tutor & Concept Explainer",
        "description": "Learn complex topics via intuitive analogies and guided questions.",
        "category": "thinking",
        "prompt": "Teach me this concept using the Socratic method. Explain the core intuition first with a vivid analogy, then guide me through deeper understanding step-by-step with thought-provoking questions.\n\nTopic:",
        "tags": ["Learning", "Education"],
    },
    {
        "id": "think-devils-advocate",
        "title": "Devil's Advocate & Stress Tester",
        "description": "Challenge assumptions, find blind spots, and test edge cases.",
        "category": "thinking",
        "prompt": "Critique this proposal rigorously as a devil's advocate. Challenge underlying assumptions, point out blind spots, failure modes, scalability traps, and unintended consequences.\n\nProposal:",
        "tags": ["Critique", "Stress Test"],
    },
]


def _build_builtin_prompts() -> list[PromptResponse]:
    now = datetime.now(timezone.utc)
    res = []
    for p in BUILTIN_PROMPTS:
        res.append(
            PromptResponse(
                id=p["id"],
                title=p["title"],
                description=p["description"],
                category=p["category"],
                prompt=p["prompt"],
                tags=p.get("tags", []),
                isCustom=False,
                created_at=now,
                updated_at=now,
            )
        )
    return res


@router.get("", response_model=list[PromptResponse])
async def list_prompts(
    q: str | None = None,
    category: str | None = None,
    db: AsyncClient = Depends(get_db),
    user: UserDB = Depends(get_current_user),
):
    repo = PromptRepository(db)
    user_prompts = await repo.get_user_prompts(user.id)

    # Convert custom prompts to response model
    custom_res = [
        PromptResponse(
            id=p.id,
            title=p.title,
            description=p.description,
            category=p.category,
            prompt=p.prompt,
            tags=p.tags,
            isCustom=True,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
        for p in user_prompts
    ]

    all_prompts = _build_builtin_prompts() + custom_res

    # Filtering logic
    filtered = []
    q_lower = q.strip().lower() if q and q.strip() else None

    for p in all_prompts:
        matches_cat = True
        if category and category != "all":
            if category == "custom":
                matches_cat = p.isCustom
            else:
                matches_cat = p.category == category

        if not matches_cat:
            continue

        if q_lower:
            search_text = f"{p.title} {p.description} {p.prompt} {' '.join(p.tags)}".lower()
            if q_lower not in search_text:
                continue

        filtered.append(p)

    return filtered


@router.post("", response_model=PromptResponse)
async def create_prompt(
    data: PromptCreate,
    db: AsyncClient = Depends(get_db),
    user: UserDB = Depends(get_current_user),
):
    repo = PromptRepository(db)
    new_prompt = PromptDB(
        user_id=str(user.id),
        title=data.title.strip(),
        description=data.description.strip(),
        category="custom",
        prompt=data.prompt.strip(),
        tags=data.tags,
        is_custom=True,
    )
    await repo.create_prompt(new_prompt)
    return PromptResponse(
        id=new_prompt.id,
        title=new_prompt.title,
        description=new_prompt.description,
        category=new_prompt.category,
        prompt=new_prompt.prompt,
        tags=new_prompt.tags,
        isCustom=True,
        created_at=new_prompt.created_at,
        updated_at=new_prompt.updated_at,
    )


@router.delete("/{prompt_id}")
async def delete_prompt(
    prompt_id: str,
    db: AsyncClient = Depends(get_db),
    user: UserDB = Depends(get_current_user),
):
    repo = PromptRepository(db)
    prompt = await repo.get_prompt(prompt_id)
    if not prompt or prompt.user_id != str(user.id):
        raise HTTPException(status_code=404, detail="Prompt not found")

    await repo.delete_prompt(prompt_id)
    return {"success": True}
