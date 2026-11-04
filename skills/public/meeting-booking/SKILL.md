---
name: meeting-booking
description: Enterprise meeting-room booking workflow via MCP meeting tools with human clarification when required.
license: MIT
allowed-tools:
  - list_available_meeting_rooms
  - create_booking
  - cancel_booking
  - list_bookings
  - ask_clarification
---

# Meeting booking

Use this skill when the user wants to find meeting rooms, create or cancel bookings, or list existing reservations.

## Workflow

1. Clarify missing details (time, duration, attendees, room preferences) with `ask_clarification` when the request is ambiguous.
2. Call `list_available_meeting_rooms` with the resolved window and capacity needs.
3. Confirm the chosen room and slot with the user when policy requires explicit approval.
4. Create the reservation with `create_booking`.
5. Use `list_bookings` or `cancel_booking` for follow-up queries or changes.

Do not use unrelated MCP tools (for example device or IoT controls) during this workflow; the active skill surface is limited to the meeting tools above plus DeerFlow discovery helpers.
