"""The golden set itself: twelve candidate profiles and three job descriptions.

Built as code rather than checked in as data so the reasoning behind each profile
is visible and reviewable. ``notes`` on each profile records what it exists to
test, which is the part that stops a golden set silently rotting into whatever
happened to be easy.

The set is designed around three properties.

**Spread.** Profiles span clear-hire to clear-reject against at least one job.
A set where everyone is mediocre cannot distinguish a good ranker from a bad one,
because there is no ranking to get right.

**Cross-role contrast.** Twelve candidates against three jobs gives thirty-six
pairs, and the same candidate is a strong fit for one job and a poor fit for
another. This tests the thing that actually matters: that the rubric is doing the
work, not a general "is this a good engineer" prior. A system that ranks the same
candidate first for every role has learned nothing about the job description.

**Deliberate traps.** Several profiles exist to catch specific failure modes:
keyword stuffing without substance, adjacent-but-not-equivalent technology,
impressive-sounding work with no measurable outcome, and a genuinely strong
candidate whose resume is badly written.
"""

from __future__ import annotations

from hirelens.evals.profiles import (
    CandidateProfile,
    Demographics,
    GoldenSet,
    JobSpec,
    ProfileProject,
    QualityTier,
    Role,
)

# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

BACKEND_JD = """\
Senior Backend Engineer, Payments Platform

You will own the services behind our payments platform: design, ship, and operate
systems processing millions of events per day, and carry the pager for what you build.

Requirements
- Strong experience running containerised workloads in production
- Experience with high-throughput event streaming such as Kafka
- A track record of measurably improving system performance or reliability
- Comfortable owning services in production, including on-call
- Strong server-side programming in a compiled or statically typed language

Nice to have
- Experience building distributed systems from first principles
- Open source contributions
- Infrastructure as code

We offer competitive salary, equity, and a hybrid working policy.
"""

ML_JD = """\
Machine Learning Engineer, Applied NLP

You will take language models from prototype to production: building retrieval
systems, evaluating them properly, and keeping them working once they ship.

Requirements
- Practical experience building systems on top of large language models
- Experience with retrieval or vector search
- Ability to design and run rigorous offline evaluation, not just eyeballing outputs
- Strong Python
- Has deployed a model or ML system to production users

Nice to have
- Experience with fine-tuning or model training
- Familiarity with responsible AI practice such as bias evaluation
- Published writing or open source in the ML space
"""

FRONTEND_JD = """\
Frontend Engineer, Design Systems

You will build and maintain the component library every product team uses, and
work closely with designers on accessibility and visual consistency.

Requirements
- Strong React and TypeScript
- Experience building or maintaining a reusable component library
- Working knowledge of web accessibility standards
- Comfortable collaborating directly with designers
- Attention to visual detail and cross-browser behaviour

Nice to have
- Experience with design tokens or theming systems
- Performance work on large client-side applications
- Testing of user interfaces
"""

JOBS = (
    JobSpec(job_id="backend", title="Senior Backend Engineer", text=BACKEND_JD),
    JobSpec(job_id="ml", title="ML Engineer, Applied NLP", text=ML_JD),
    JobSpec(job_id="frontend", title="Frontend Engineer, Design Systems", text=FRONTEND_JD),
)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def _d(name: str, **kwargs: str) -> Demographics:
    return Demographics(name=name, **kwargs)


PROFILES: tuple[CandidateProfile, ...] = (
    # -- Backend, strong -----------------------------------------------------
    CandidateProfile(
        candidate_id="c01",
        demographics=_d("Alex Mercer", university="State University", location="Berlin"),
        headline="Backend engineer, distributed systems and payments",
        github="alexmercer",
        target_role="backend",
        quality=QualityTier.STRONG,
        notes="Unambiguous top pick for backend. Every must-have has concrete evidence.",
        roles=(
            Role(
                title="Senior Backend Engineer",
                company="Northwind Payments",
                start="2021",
                bullets=(
                    "Owned the settlement service end to end, processing 4M events per day through Kafka.",
                    "Cut p99 authorisation latency from 840ms to 120ms by replacing a synchronous ledger write with an append-only log.",
                    "Ran the service on Kubernetes across three regions and carried the primary pager for two years.",
                    "Reduced incident rate by 60% over four quarters by introducing backpressure and circuit breaking.",
                ),
            ),
            Role(
                title="Backend Engineer",
                company="Loamworks",
                start="2018",
                end="2021",
                bullets=(
                    "Built the billing reconciliation pipeline in Go, handling 200k transactions daily.",
                    "Migrated 40 services from EC2 to Kubernetes with Terraform-managed infrastructure.",
                ),
            ),
        ),
        projects=(
            ProfileProject(
                name="raftkv",
                description="Raft consensus implementation and distributed key-value store in Go",
                url="github.com/alexmercer/raftkv",
                bullets=("620 stars. Used as teaching material in two university courses.",),
            ),
        ),
        skills=("Go", "Rust", "Python", "Kafka", "Kubernetes", "Terraform", "PostgreSQL", "gRPC"),
    ),
    # -- Backend, solid ------------------------------------------------------
    CandidateProfile(
        candidate_id="c02",
        demographics=_d("Priya Raman", university="State University", location="Bengaluru"),
        headline="Backend engineer",
        github="priyaraman",
        target_role="backend",
        quality=QualityTier.SOLID,
        notes="Meets most backend must-haves but thinner on scale and no on-call evidence.",
        roles=(
            Role(
                title="Backend Engineer",
                company="Fintech Co.",
                start="2022",
                bullets=(
                    "Cut p99 checkout latency from 1.2s to 180ms by removing an ORM N+1 in the payments path.",
                    "Built a Kafka consumer group processing 2M settlement events per day.",
                    "Deployed and operated the reconciliation service on Kubernetes.",
                ),
            ),
        ),
        projects=(
            ProfileProject(
                name="kvstore",
                description="Raft-based distributed key-value store in Go",
                url="github.com/priyaraman/kvstore",
            ),
        ),
        skills=("Go", "Python", "PostgreSQL", "Kafka", "Kubernetes", "Terraform"),
    ),
    # -- Backend, adjacent technology trap -----------------------------------
    CandidateProfile(
        candidate_id="c03",
        demographics=_d("Jordan Blake", university="State University", location="Manchester"),
        headline="Backend engineer",
        target_role="backend",
        quality=QualityTier.MIXED,
        notes=(
            "TRAP: Docker and RabbitMQ are adjacent to but not the same as Kubernetes and "
            "Kafka. A system that treats them as equivalent will over-score this profile."
        ),
        roles=(
            Role(
                title="Backend Engineer",
                company="Harbourline Logistics",
                start="2020",
                bullets=(
                    "Built REST services in Java Spring for warehouse routing.",
                    "Containerised eight services with Docker and docker-compose.",
                    "Used RabbitMQ for asynchronous job dispatch between services.",
                    "Improved a nightly batch job from 6 hours to 2 hours.",
                ),
            ),
        ),
        skills=("Java", "Spring", "Docker", "RabbitMQ", "MySQL"),
    ),
    # -- Backend, keyword stuffing trap --------------------------------------
    CandidateProfile(
        candidate_id="c04",
        demographics=_d("Sam Okafor", university="State University", location="Lagos"),
        headline="Full stack developer",
        target_role="backend",
        quality=QualityTier.WEAK,
        notes=(
            "TRAP: every required keyword appears in the skills list, but no bullet "
            "demonstrates any of them. Lexical retrieval will match; the judge should not."
        ),
        roles=(
            Role(
                title="Software Developer",
                company="Bright Path Solutions",
                start="2022",
                bullets=(
                    "Was responsible for various backend tasks and bug fixes.",
                    "Worked on internal tools used by the operations team.",
                    "Participated in code reviews and agile ceremonies.",
                ),
            ),
        ),
        skills=(
            "Kubernetes",
            "Kafka",
            "Go",
            "Terraform",
            "gRPC",
            "PostgreSQL",
            "Docker",
            "AWS",
            "microservices",
            "distributed systems",
        ),
    ),
    # -- ML, strong ----------------------------------------------------------
    CandidateProfile(
        candidate_id="c05",
        demographics=_d("Riley Chen", university="State University", location="Toronto"),
        headline="ML engineer, applied NLP and retrieval",
        github="rileychen",
        target_role="ml",
        quality=QualityTier.STRONG,
        notes="Top pick for ML. Notably has real evaluation rigour, which is the rarest signal.",
        roles=(
            Role(
                title="Machine Learning Engineer",
                company="Cartwright Health",
                start="2021",
                bullets=(
                    "Shipped a clinical document retrieval system to 12,000 daily users, built on hybrid BM25 and dense retrieval.",
                    "Built the offline evaluation harness: 800 labelled query-document pairs, nDCG@10 tracked per release, regression gate in CI.",
                    "Raised nDCG@10 from 0.61 to 0.79 by adding a cross-encoder reranker and measuring the trade-off against latency.",
                    "Ran a demographic bias evaluation on the triage model and published the findings internally.",
                ),
            ),
            Role(
                title="Data Scientist",
                company="Fernwood Analytics",
                start="2019",
                end="2021",
                bullets=(
                    "Fine-tuned transformer classifiers for document routing, reaching 0.91 macro F1 against a 0.74 baseline.",
                ),
            ),
        ),
        projects=(
            ProfileProject(
                name="evalkit",
                description="Reproducible evaluation harness for retrieval systems",
                url="github.com/rileychen/evalkit",
                bullets=(
                    "Bootstrap confidence intervals and paired significance testing for IR metrics.",
                ),
            ),
        ),
        skills=("Python", "PyTorch", "transformers", "FAISS", "Elasticsearch", "MLflow", "Docker"),
    ),
    # -- ML, prototype-only trap ---------------------------------------------
    CandidateProfile(
        candidate_id="c06",
        demographics=_d("Taylor Nunes", university="State University", location="Lisbon"),
        headline="ML enthusiast",
        github="taylornunes",
        target_role="ml",
        quality=QualityTier.MIXED,
        notes=(
            "TRAP: lots of LLM vocabulary but everything is a notebook or a demo. Nothing "
            "reached users and nothing was evaluated. Should rank well below c05."
        ),
        roles=(
            Role(
                title="Junior Data Analyst",
                company="Vellum Retail",
                start="2023",
                bullets=(
                    "Built dashboards in Looker for the merchandising team.",
                    "Explored using GPT for product description generation in a prototype.",
                ),
            ),
        ),
        projects=(
            ProfileProject(
                name="chat-with-pdf",
                description="RAG chatbot over PDFs using LangChain and Chroma",
                bullets=("Followed a tutorial and extended it with a Streamlit interface.",),
            ),
            ProfileProject(
                name="sentiment-analyzer",
                description="Sentiment classification of tweets using scikit-learn",
            ),
        ),
        skills=(
            "Python",
            "LangChain",
            "OpenAI API",
            "Chroma",
            "pandas",
            "scikit-learn",
            "Streamlit",
        ),
    ),
    # -- ML, research-heavy, production-light --------------------------------
    CandidateProfile(
        candidate_id="c07",
        demographics=_d("Morgan Ellis", university="State University", location="Edinburgh"),
        headline="ML researcher",
        target_role="ml",
        quality=QualityTier.SOLID,
        notes=(
            "Strong on modelling and evaluation, weak on production deployment. Tests whether "
            "the system distinguishes a genuine partial match from a full one."
        ),
        roles=(
            Role(
                title="Research Assistant",
                company="Institute for Language Technology",
                start="2020",
                bullets=(
                    "Trained and evaluated multilingual sequence labelling models across 14 languages.",
                    "Designed the evaluation protocol including significance testing and ablation studies.",
                    "Published two first-author papers at ACL workshops.",
                ),
            ),
        ),
        projects=(
            ProfileProject(
                name="polyglot-ner",
                description="Multilingual named entity recognition benchmark and baselines",
                url="github.com/morganellis/polyglot-ner",
            ),
        ),
        skills=("Python", "PyTorch", "transformers", "NumPy", "LaTeX"),
        awards=("Best paper award, ACL workshop on multilingual NLP, 2022",),
    ),
    # -- Frontend, strong ----------------------------------------------------
    CandidateProfile(
        candidate_id="c08",
        demographics=_d("Devin Park", university="State University", location="Seoul"),
        headline="Frontend engineer, design systems and accessibility",
        github="devinpark",
        target_role="frontend",
        quality=QualityTier.STRONG,
        notes="Top pick for frontend. Accessibility evidence is specific rather than claimed.",
        roles=(
            Role(
                title="Senior Frontend Engineer",
                company="Copperleaf Software",
                start="2021",
                bullets=(
                    "Built and maintained the company design system: 74 React components in TypeScript, used by nine product teams.",
                    "Took the component library from WCAG 2.1 AA failures on 31 components to full compliance, verified with axe and manual screen reader testing.",
                    "Introduced design tokens shared between Figma and code, cutting design-to-implementation drift.",
                    "Reduced main bundle size by 38% through code splitting and dependency auditing.",
                ),
            ),
            Role(
                title="Frontend Engineer",
                company="Marlowe Digital",
                start="2019",
                end="2021",
                bullets=(
                    "Rebuilt the checkout flow in React, raising conversion by 12% in an A/B test.",
                ),
            ),
        ),
        projects=(
            ProfileProject(
                name="a11y-primitives",
                description="Accessible unstyled React primitives",
                url="github.com/devinpark/a11y-primitives",
                bullets=(
                    "1.2k stars. Full keyboard and screen reader support with automated tests.",
                ),
            ),
        ),
        skills=("TypeScript", "React", "CSS", "Storybook", "Playwright", "Figma", "Vite"),
    ),
    # -- Frontend, solid but no design system --------------------------------
    CandidateProfile(
        candidate_id="c09",
        demographics=_d("Casey Adeyemi", university="State University", location="Dublin"),
        headline="Frontend developer",
        target_role="frontend",
        quality=QualityTier.SOLID,
        notes="Good React work, but nothing about component libraries or accessibility.",
        roles=(
            Role(
                title="Frontend Developer",
                company="Ridgeway Media",
                start="2021",
                bullets=(
                    "Built customer-facing pages in React and TypeScript for a news platform with 400k monthly readers.",
                    "Improved Largest Contentful Paint from 4.1s to 1.6s through image optimisation and lazy loading.",
                    "Wrote end-to-end tests in Cypress covering the subscription funnel.",
                ),
            ),
        ),
        skills=("JavaScript", "TypeScript", "React", "Next.js", "CSS", "Cypress"),
    ),
    # -- Career changer, weak everywhere -------------------------------------
    CandidateProfile(
        candidate_id="c10",
        demographics=_d("Robin Vasquez", university="State University", location="Madrid"),
        headline="Career changer moving into software",
        target_role="",
        quality=QualityTier.WEAK,
        notes="Genuine effort, little relevant evidence. Should rank low for all three jobs.",
        roles=(
            Role(
                title="Operations Coordinator",
                company="Alderman Freight",
                start="2016",
                end="2023",
                bullets=(
                    "Coordinated scheduling for a fleet of 40 vehicles.",
                    "Built spreadsheet automation that saved the team roughly 5 hours per week.",
                ),
            ),
        ),
        projects=(
            ProfileProject(name="todo-app", description="Task tracker built while learning React"),
            ProfileProject(name="weather-dashboard", description="Weather app using a public API"),
        ),
        skills=("JavaScript", "React", "HTML", "CSS", "Python"),
    ),
    # -- Strong engineer, badly written resume -------------------------------
    CandidateProfile(
        candidate_id="c11",
        demographics=_d("Kim Fontaine", university="State University", location="Montreal"),
        headline="",
        github="kimfontaine",
        target_role="backend",
        quality=QualityTier.MIXED,
        notes=(
            "TRAP: genuinely strong backend experience described in vague, passive language "
            "with no metrics. Tests whether the system penalises poor writing more than poor "
            "ability. A human screener would probably interview this person."
        ),
        roles=(
            Role(
                title="Software Engineer",
                company="Halden Systems",
                start="2019",
                bullets=(
                    "Worked on the core transaction processing platform.",
                    "Involved in migrating services to Kubernetes.",
                    "Helped with the Kafka-based event pipeline.",
                    "Participated in the on-call rotation.",
                ),
            ),
        ),
        skills=("Go", "Kubernetes", "Kafka", "PostgreSQL"),
    ),
    # -- Generalist, moderate fit for two roles ------------------------------
    CandidateProfile(
        candidate_id="c12",
        demographics=_d("Avery Lindqvist", university="State University", location="Stockholm"),
        headline="Full stack engineer",
        github="averylindqvist",
        target_role="",
        quality=QualityTier.MIXED,
        notes=(
            "Deliberately in-between: real but shallow evidence for backend and frontend, "
            "none for ML. Tests that rankings differ sensibly across the three jobs."
        ),
        roles=(
            Role(
                title="Full Stack Engineer",
                company="Sundial Booking",
                start="2020",
                bullets=(
                    "Built booking flows in React and TypeScript used by 60k monthly users.",
                    "Wrote the Node.js API behind them and deployed it on Kubernetes.",
                    "Cut page load time by 30% by moving rendering server-side.",
                ),
            ),
        ),
        projects=(
            ProfileProject(
                name="sundial-ui",
                description="Internal React component library, 20 components",
                url="github.com/averylindqvist/sundial-ui",
            ),
        ),
        skills=("TypeScript", "React", "Node.js", "PostgreSQL", "Kubernetes", "Docker"),
    ),
)


def build_golden_set() -> GoldenSet:
    """The full golden set: 12 profiles times 3 jobs, so 36 labelled pairs."""
    return GoldenSet(profiles=PROFILES, jobs=JOBS)
