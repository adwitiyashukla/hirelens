# Setup, step by step

Written for Windows. Every command goes in **PowerShell**, opened inside the
project folder. To open it there: open the `hiring_bot` folder in File Explorer,
click the address bar, type `powershell`, press Enter.

---

## Step 1: Check Python is installed

```powershell
python --version
```

You need 3.10 or higher. If you see an error or a version below 3.10, install
Python from [python.org/downloads](https://www.python.org/downloads/) and **tick
"Add python.exe to PATH"** on the first screen of the installer.

---

## Step 2: Install the project

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,api]"
```

After `activate`, your prompt shows `(.venv)` at the start. That means it worked.

You will need to run `.venv\Scripts\activate` again each time you open a new
PowerShell window.

---

## Step 3: Get a free Google API key

1. Go to [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys).
2. Click **Create API key** (top right).
3. If it asks about a project, pick any existing one or let it create a new one.
4. A long string starting with `AIza...` appears. Click the copy icon.

This is free. There is no card and no charge.

**Keep the key private.** Do not paste it into a chat, a screenshot, or a public
repository. If you ever do by accident, delete it on that page and make a new one.

---

## Step 4: Put the key in the project

```powershell
copy .env.example .env
notepad .env
```

Notepad opens. Find this line:

```
HIRELENS_GEMINI_API_KEY=
```

Paste your key straight after the `=`, with no spaces and no quotes:

```
HIRELENS_GEMINI_API_KEY=AIzaSy...your...key...here
```

Save (Ctrl+S) and close Notepad.

`.env` is already in `.gitignore`, so it will never be committed.

---

## Step 5: Check it works

```powershell
hirelens doctor
```

You want a green **Provider OK** panel. If instead you see:

- *No credentials*: the key did not save. Reopen `.env` and check it is on the
  `HIRELENS_GEMINI_API_KEY=` line with nothing before or after it.
- *Provider unreachable*: check your internet connection, then confirm the key is
  still listed on the API keys page.

---

## Step 6: Try it on one resume

```powershell
hirelens score samples\priya_narayanan.pdf --jd samples\senior_backend_engineer.txt
```

This is the whole system on one candidate: a score, a confidence band, the exact
resume lines behind each requirement, risk flags, and interview questions.

---

## Step 7: Label the golden set (about 40 minutes)

This is the only part nobody else can do. It creates the human ground truth every
quality number is measured against.

```powershell
make golden
make label
```

If `make` is not available on your machine, use these instead:

```powershell
python -m hirelens.evals.cli generate
python -m hirelens.evals.cli label
```

You will see a job description, then a resume, then one question:

> Would you interview them for this role? (1-5, s, q)

Answer with a number:

| Key | Meaning |
|---|---|
| `1` | Clear reject, not close to the role |
| `2` | Would not interview, misses things that matter |
| `3` | Borderline, only if the pipeline were thin |
| `4` | Would interview, meets the requirements |
| `5` | Clear interview, meets the bar with room to spare |

Then type a short reason, and press Enter.

Answer as a busy screener, not a perfectionist. Your first instinct is the right
one. 36 questions in total.

It saves after every answer, so you can press `q` to stop and rerun `make label`
later to pick up exactly where you left off.

**Do not look up the system's score first.** Seeing the machine's answer before
giving yours would bias the comparison and inflate every number in the report.

---

## Step 8: Produce the real numbers

```powershell
make eval
python -m hirelens.audit.cli run --budget tiny --gate
```

The first prints how well the system agrees with your rankings, next to three
baselines. The second produces `docs/BIAS_AUDIT.md`.

Both write files you can then paste into the README.

---

## Step 9: Run the dashboard

The dashboard is already built and sitting in `web\dist`, so this needs nothing
except the API:

```powershell
pip install -e ".[api]"
python -m uvicorn hirelens.api.app:app --port 8000
```

Open <http://localhost:8000> in a browser. That one address serves both the
dashboard and the API, because the API serves the built files itself.

<http://localhost:8000/docs> is the interactive API documentation, if you want to
click through the endpoints directly.

**What to do in it**, in order:

1. Click **Use sample** to fill in the job description, then **Compile rubric**.
   You will see the description turn into weighted requirements.
2. Drop `samples\priya_narayanan.pdf` onto the upload area.
3. Click **Screen 1 candidate** and watch the progress bar.
4. Click the row in the shortlist.
5. On the candidate page, click any requirement on the left. The exact lines of
   the resume it was scored from light up on the right, and everything else
   dims. That is the screenshot worth taking.

### Only if you want to change the dashboard code

You need [Node.js](https://nodejs.org) (the LTS installer) for this. Not needed
otherwise.

```powershell
cd web
npm install
npm run dev
```

That serves the dashboard on <http://localhost:5173> with hot reload, and passes
API calls through to port 8000, so run `python -m uvicorn hirelens.api.app:app`
in a second terminal at the same time.

After changing anything, rebuild so the API picks it up:

```powershell
cd web
npm run build
```

---

## Step 10: Take the screenshots

`docs\screenshots\README.md` lists the exact five images the README expects, what
each one has to show, and what not to put in frame. Read it before capturing:
two of the five are easy to get subtly wrong in a way that undercuts the point
they exist to make.

---

## If something breaks

- `'make' is not recognized`: use the `python -m ...` version of the command.
- `'hirelens' is not recognized`: run `.venv\Scripts\activate` first.
- `ModuleNotFoundError`: run `pip install -e ".[dev,api]"` again.
- `404 ... model is not found`: run `hirelens models` to see what your key can
  actually call, then set `HIRELENS_GEMINI_MODEL` in `.env` to one of those.
- **Every request returns 429**: this is the daily free-tier cap, not a settings
  problem. The error message says so explicitly. Lowering
  `HIRELENS_REQUESTS_PER_MINUTE` will not help, because the pacing is already
  working. Wait for the reset, or switch to Groq or Ollama.
- The dashboard loads but says it cannot reach the API: the API is not running,
  or is on a different port. Start it and reload.
- Anything else: run the command again with `--verbose` and read the last few lines.
