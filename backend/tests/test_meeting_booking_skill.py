"""Meeting-booking skill declares the enterprise MCP tool surface."""

from pathlib import Path

from deerflow.skills.parser import parse_skill_file
from deerflow.skills.types import SkillCategory


def test_meeting_booking_skill_allowed_tools():
    skill_file = Path(__file__).resolve().parents[2] / "skills" / "public" / "meeting-booking" / "SKILL.md"
    skill = parse_skill_file(skill_file, category=SkillCategory.PUBLIC)
    assert skill.name == "meeting-booking"
    assert skill.allowed_tools is not None
    assert set(skill.allowed_tools) == {
        "list_available_meeting_rooms",
        "create_booking",
        "cancel_booking",
        "list_bookings",
        "ask_clarification",
    }
