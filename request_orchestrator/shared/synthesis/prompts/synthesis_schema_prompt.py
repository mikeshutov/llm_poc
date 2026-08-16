SYNTHESIS_SCHEMA = """

Result must be self-contained and final.
Ensure the `roundtrip_summary` field is a detailed plain-language diagnostic summary of the roundtrip outcome for history and retrieval.
`tool_summary` records information produced and key entities.Set `next_question` to one non-empty question or follow up statement.
Associate each result block with supporting `evidence_ids`; never expose internal IDs in content.
Keep follow-up content or questions only in next_question.

Follow the provided response schema.:

{
  "next_question": "<one follow up or clarifying question>",
  "result": [
    {
      "content": "<paragraph answering some part of the question>",
      "evidence_ids": ["P1E1R1"]
    },
  ],
  "roundtrip_summary": "<detailed 80-120 token summary of what the user asked, what was done, what evidence or tools mattered, and what outcome was reached>",
  "tool_summary": {
    "produced": ["current_temperature", "wind_speed"],
    "entities": ["Toronto"]
  }
}
"""
