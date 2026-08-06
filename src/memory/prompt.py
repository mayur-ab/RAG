from typing import Optional

from src.memory.models import UserMemoryContext


def augment_with_user_memory(base_prompt: str, memory_context: UserMemoryContext) -> str:
    """Prepend user profile and episodic memories to the system prompt."""
    block = format_memory_context(memory_context)
    if not block:
        return base_prompt
    return f"{base_prompt.rstrip()}\n\n{block}"


def format_memory_context(ctx: UserMemoryContext) -> str:
    sections: list[str] = []

    profile_lines: list[str] = []
    if ctx.profile.display_name:
        profile_lines.append(f"- Name: {ctx.profile.display_name}")
    prefs = ctx.profile.preferences
    if prefs.response_style and prefs.response_style != "balanced":
        profile_lines.append(f"- Response style: {prefs.response_style}")
    if prefs.technical_level and prefs.technical_level != "intermediate":
        profile_lines.append(f"- Technical level: {prefs.technical_level}")
    if prefs.language and prefs.language != "english":
        profile_lines.append(f"- Preferred language: {prefs.language}")

    if ctx.profile.projects:
        profile_lines.append("- Projects: " + "; ".join(ctx.profile.projects[:6]))
    if ctx.profile.interests:
        profile_lines.append("- Interests: " + "; ".join(ctx.profile.interests[:6]))
    if ctx.profile.facts:
        profile_lines.append("- Known facts: " + "; ".join(ctx.profile.facts[:6]))

    style = ctx.profile.style
    style_lines: list[str] = []
    if style.preferred_answer_style and style.preferred_answer_style != "balanced":
        style_lines.append(f"- Preferred answer style: {style.preferred_answer_style}")
    if style.likes_examples:
        style_lines.append("- Include practical examples when helpful")
    if style.likes_flowcharts:
        style_lines.append("- Use architecture diagrams or flow descriptions when relevant")
    if style.experience_level:
        style_lines.append(f"- Experience level: {style.experience_level}")

    if profile_lines:
        sections.append("User Profile:\n" + "\n".join(profile_lines))
    if style_lines:
        sections.append("Answer Style:\n" + "\n".join(style_lines))
    if ctx.frequent_topics:
        sections.append("Frequently asked topics: " + ", ".join(ctx.frequent_topics[:8]))
    if ctx.episodic_memories:
        memory_lines = [f"- {hit.text}" for hit in ctx.episodic_memories[:5] if hit.text.strip()]
        if memory_lines:
            sections.append("Relevant Past Context:\n" + "\n".join(memory_lines))

    if not sections:
        return ""
    return (
        "Known User Context (personalize answers using this; do not mention that you are using memory):\n\n"
        + "\n\n".join(sections)
    )


def build_memory_context_for_query(
    ctx: Optional[UserMemoryContext],
) -> str:
    if ctx is None:
        return ""
    return format_memory_context(ctx)
