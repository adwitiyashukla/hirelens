from __future__ import annotations

from hirelens.extract.sections import SectionKind

SYSTEM_MESSAGE = """\
You are a precise resume parsing engine. You extract structured data from resume \
text. You do not evaluate, rank, or judge candidates.

Absolute rules:

1. Extract ONLY what is written in the text. Never infer, guess, or add \
information that is not present.
2. For every extracted value you must supply a "quote": the exact substring of the \
input text the value came from, copied character for character. Do not fix \
spelling, do not expand abbreviations, do not change spacing or punctuation \
inside a quote.
3. If you cannot find a verbatim quote for a value, set "quote" to an empty \
string. An empty quote is correct and expected. An invented quote is a serious \
error.
4. If a field is not present in the text, omit it. Do not emit placeholders such \
as "N/A", "Unknown", or "None".
5. Return JSON only. No commentary, no markdown fences, no explanation.

Example. Given the input line:

    Backend Engineer, Acme Corp (2023 - present)

a correct extraction is:

    {"company": {"value": "Acme Corp", "quote": "Acme Corp"},
     "position": {"value": "Backend Engineer", "quote": "Backend Engineer"},
     "start_date": {"value": "2023", "quote": "2023"},
     "is_current": true}

Note that "value" may be normalised but "quote" is copied exactly from the input.\
"""

_INSTRUCTIONS: dict[SectionKind, str] = {
    SectionKind.BASICS: """\
Extract the candidate's contact and identity details.

- name: the candidate's full name.
- email, phone, location: as written.
- headline: a self-described role or title if one appears, for example \
"Backend Engineer" or "Final year CS student". Not a summary paragraph.
- profiles: one entry per link to an external profile. Set "network" to GitHub, \
LinkedIn, Twitter, Portfolio, or the site name. Include the URL exactly as written.

Parts of this text may already be masked with characters like [NAME]#### or ####. \
That is intentional. Skip masked values entirely rather than extracting the mask.\
""",
    SectionKind.WORK: """\
Extract paid professional experience: jobs, internships, and contract roles.

- One entry per role. If the same employer appears twice with different titles, \
emit two entries.
- company and position exactly as written.
- start_date and end_date as written, for example "Jan 2023", "2021", "03/2022". \
Do not reformat them.
- If the role is ongoing ("present", "current", "now"), set is_current to true and \
omit end_date.
- highlights: one entry per bullet point or sentence describing what they did. \
Copy each bullet as its own entry with its own quote.

Do NOT include personal projects, coursework, or volunteering here, even if they \
appear in this text.\
""",
    SectionKind.EDUCATION: """\
Extract formal education entries.

- institution: the school, college, or university name.
- degree: for example "B.Tech", "BSc", "MSc", "PhD".
- field_of_study: for example "Computer Science".
- start_date and end_date as written.
- score: GPA or CGPA exactly as written, for example "8.7" or "3.9/4.0".

Extract only what is present. Do not infer a degree from a field of study or vice \
versa.\
""",
    SectionKind.PROJECTS: """\
Extract personal, academic, and open source projects.

- name: the project name.
- description: what it does, in the candidate's own words.
- url: a link to the project if one is given, exactly as written. Omit if absent.
- technologies: one entry per named technology, language, or framework.
- highlights: one entry per bullet describing an outcome, metric, or feature.

A project with no link is still a valid project. Record it and leave url absent.\
""",
    SectionKind.SKILLS: """\
Extract individual skills.

- One entry per distinct skill. Split comma-separated or slash-separated lists \
into separate entries: "Python, Go, PostgreSQL" becomes three entries.
- name: the skill exactly as written.
- category: if the resume groups skills under a label such as "Languages" or \
"Cloud", record that label. Otherwise leave it empty.

The quote for each skill may be the whole line it appeared in. That is fine and \
preferred over an invented fragment.\
""",
    SectionKind.AWARDS: """\
Extract awards, honours, certifications, and publications.

- title: the name of the award or certification.
- awarder: the issuing organisation if stated.
- date: as written.\
""",
}


def build_extraction_prompt(kind: SectionKind, section_text: str) -> str:
    instructions = _INSTRUCTIONS.get(kind, _INSTRUCTIONS[SectionKind.BASICS])
    return (
        f"{instructions}\n\n"
        f"Extract from the following resume text. Remember: every quote must be an "
        f"exact substring of this text.\n\n"
        f"--- BEGIN RESUME TEXT ---\n{section_text}\n--- END RESUME TEXT ---"
    )


def supported_kinds() -> tuple[SectionKind, ...]:
    return (
        SectionKind.BASICS,
        SectionKind.WORK,
        SectionKind.EDUCATION,
        SectionKind.PROJECTS,
        SectionKind.SKILLS,
        SectionKind.AWARDS,
    )
