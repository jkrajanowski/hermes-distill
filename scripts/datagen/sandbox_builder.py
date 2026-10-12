"""Build a realistic sandbox for Hermes sessions: repo copy + synthetic data + a small buggy package."""

from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]  # repo root; copied into every sandbox

MANIFEST = """\
Sandbox layout (relative to cwd):
- README.md, pyproject.toml, LICENSE, src/hermes_distill/*.py (a Python CLI package), tests/test_pipeline.py
- data/sales.csv (order_id,date,region,product,quantity,unit_price,revenue; 300 rows)
- data/users.csv (id,name,email,age,country; 80 rows, some invalid emails/ages, duplicate emails)
- data/inventory.csv (sku,name,stock,reorder_at)
- data/orders.csv (order_id,customer_id,status,total,created_at; statuses: paid/refunded/pending/cancelled; 500 rows)
- data/products.json (list of {sku,name,category,price,tags})
- data/events.json (list of {id,type,ts,user}; types click/view/purchase/refund/signup)
- data/api_response.json (nested JSON: {"page":1,"results":[{"id","title","owner":{"name","email"},"metrics":{"views","likes"}}]})
- data/config.yaml (app config: server host/port, database url, feature flags, log level)
- data/settings.ini ([server] host/port, [db] url, [features] flags)
- data/requirements.txt (pinned python deps, one has an obviously wrong version)
- data/schema.sql (CREATE TABLE statements for users, orders, products)
- data/logs/app.log (600 lines: timestamp LEVEL [component] message; levels INFO/WARN/ERROR/DEBUG)
- data/logs/access.log (800 nginx-style lines: ip, time, request, status, bytes)
- data/notes/meeting.md (attendees, discussion, decisions with owners and dates), data/notes/ideas.md, data/notes/roadmap.md
- data/docs/api.md (REST endpoint docs), data/docs/onboarding.md
- data/buggy/average.py, slugify.py, parse_date.py, slow.py (each has one bug or perf problem), data/buggy/config.ini
- data/minipkg/ (tiny python package: __init__.py, money.py [currency rounding bug], dates.py [timezone bug],
  textutils.py [unicode bug], stats.py, cli.py); data/minipkg_tests/test_minipkg.py (4 tests, 3 currently fail)
- out/ (empty, for anything the agent writes)
Tools available to the agent: terminal, read_file, write_file, patch, search_files, execute_code (python).
"""


AGENTS_MD = """\
# Sandbox rules

This directory is a self-contained sandbox and the ONLY place you may work.

- Never read, list, search, or modify anything outside the current working directory.
- Never use `~`, `/`, `C:\\`, `..`, or any absolute path. Use paths relative to this directory only.
- Never run git commands against directories outside this one.
- Never install, uninstall, or reinstall Python packages (no pip/uv install, no `pip install -e .`).
- If a task refers to a file, value, or feature that does not exist here, say so and stop.
  Do not go looking for it elsewhere on the machine.
- Write new files under `out/`.
"""


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def build(sandbox: Path, seed: int = 7) -> None:
    if sandbox.exists():
        shutil.rmtree(sandbox)
    sandbox.mkdir(parents=True)
    for item in ("README.md", "pyproject.toml", "LICENSE", "src", "tests"):
        src = PROJECT / item
        if src.is_dir():
            shutil.copytree(src, sandbox / item, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
        else:
            shutil.copy2(src, sandbox / item)
    (sandbox / "out").mkdir()
    _write(sandbox / "AGENTS.md", AGENTS_MD)
    data = sandbox / "data"
    data.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    products = ["Widget", "Gadget", "Gizmo", "Doohickey", "Sprocket", "Flange", "Bracket", "Valve"]
    regions = ["EU", "US", "APAC", "LATAM"]
    with (data / "sales.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "date", "region", "product", "quantity", "unit_price", "revenue"])
        for i in range(1, 301):
            q = rng.randint(1, 40)
            p = rng.choice([4.5, 9.99, 12.0, 19.5, 25.0, 49.0, 99.0])
            w.writerow([1000 + i, f"2026-{rng.randint(1, 9):02d}-{rng.randint(1, 28):02d}", rng.choice(regions),
                        rng.choice(products), q, p, round(q * p, 2)])

    names = ["ana", "bob", "cleo", "dan", "eva", "finn", "gus", "hana", "ivo", "jo", "kim", "leo"]
    with (data / "users.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "email", "age", "country"])
        for i in range(1, 81):
            n = rng.choice(names)
            email = f"{n}{rng.randint(1, 12)}@example.com"
            age = rng.choice([rng.randint(18, 70), rng.randint(18, 70), rng.randint(18, 70), "n/a", 140, -3])
            w.writerow([i, n.title(), email if rng.random() > 0.05 else f"{n}.example.com", age,
                        rng.choice(["PL", "DE", "US", "NL", "FR"])])

    with (data / "inventory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sku", "name", "stock", "reorder_at"])
        for i, p in enumerate(products * 3):
            w.writerow([f"SKU-{100 + i}", f"{p} {i}", rng.randint(0, 120), rng.choice([10, 20, 30])])

    with (data / "orders.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order_id", "customer_id", "status", "total", "created_at"])
        for i in range(1, 501):
            w.writerow([5000 + i, rng.randint(1, 80), rng.choices(["paid", "refunded", "pending", "cancelled"],
                                                                  weights=[70, 10, 12, 8])[0],
                        round(rng.uniform(5, 900), 2), f"2026-{rng.randint(1, 10):02d}-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00"])

    prods = [{"sku": f"SKU-{100 + i}", "name": f"{p} {i}", "category": rng.choice(["tools", "parts", "kits"]),
              "price": rng.choice([4.5, 9.99, 12.0, 19.5, 25.0, 49.0, 99.0]),
              "tags": rng.sample(["new", "sale", "bulk", "fragile", "eu-only", "bestseller"], k=rng.randint(0, 3))}
             for i, p in enumerate(products * 3)]
    _write(data / "products.json", json.dumps(prods, indent=1))

    t0 = 1_760_000_000
    events = [{"id": i, "type": rng.choice(["click", "view", "purchase", "refund", "signup"]),
               "ts": t0 + rng.randint(0, 86_400 * 10), "user": rng.randint(1, 80)} for i in range(300)]
    _write(data / "events.json", json.dumps(events, indent=1))

    api = {"page": 1, "per_page": 25, "total": 61, "results": [
        {"id": i, "title": f"Item {i}", "owner": {"name": rng.choice(names).title(), "email": f"{rng.choice(names)}@example.com"},
         "metrics": {"views": rng.randint(0, 5000), "likes": rng.randint(0, 500)}, "archived": rng.random() < 0.2}
        for i in range(1, 26)]}
    _write(data / "api_response.json", json.dumps(api, indent=1))

    _write(data / "config.yaml",
           "server:\n  host: 0.0.0.0\n  port: 8080\n  workers: 4\ndatabase:\n  url: postgres://app:app@localhost:5432/app\n"
           "  pool_size: 10\nfeatures:\n  new_checkout: true\n  beta_search: false\n  dark_mode: true\nlogging:\n  level: INFO\n"
           "  file: logs/app.log\n")
    _write(data / "settings.ini", "[server]\nhost = 0.0.0.0\nport = 8080\n\n[db]\nurl = sqlite:///app.db\n\n[features]\n"
                                  "new_checkout = true\nbeta_search = false\n")
    _write(data / "requirements.txt", "requests==2.32.3\nrich==13.9.4\ntyper==0.15.1\nnumpy==99.0.0\npandas==2.2.3\n")
    _write(data / "schema.sql",
           "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE, age INTEGER, country TEXT);\n"
           "CREATE TABLE products (sku TEXT PRIMARY KEY, name TEXT, category TEXT, price REAL);\n"
           "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES users(id), status TEXT,\n"
           "  total REAL, created_at TEXT);\nCREATE INDEX idx_orders_customer ON orders(customer_id);\n")

    comps = ["auth", "billing", "search", "mailer", "sync", "api"]
    msgs = ["connection reset by peer", "timeout after 30s", "invalid token", "refund failed: card declined",
            "index rebuild skipped", "latency 123ms", "rate limited", "disk usage 91%", "retrying job 4412",
            "user 17 logged in", "cache miss for key products:all"]
    lines = []
    for i in range(600):
        lvl = rng.choices(["INFO", "WARN", "ERROR", "DEBUG"], weights=[60, 15, 15, 10])[0]
        msg = rng.choice(msgs)
        if "latency" in msg:
            msg = f"latency {rng.randint(20, 900)}ms"
        hh, mm, ss = 8 + i // 240, (i // 4) % 60, (i * 7) % 60
        lines.append(f"2026-10-11 {hh:02d}:{mm:02d}:{ss:02d} {lvl} [{rng.choice(comps)}] {msg}")
    _write(data / "logs" / "app.log", "\n".join(lines) + "\n")
    ips = [f"10.0.{rng.randint(0, 3)}.{rng.randint(1, 40)}" for _ in range(25)]
    acc = []
    for i in range(800):
        status = rng.choices([200, 301, 404, 500, 503], weights=[70, 5, 15, 6, 4])[0]
        acc.append(f'{rng.choice(ips)} - - [11/Oct/2026:{8 + i // 400:02d}:{(i // 7) % 60:02d}:{i % 60:02d} +0000] '
                   f'"GET /{rng.choice(["", "api/items", "api/users", "health", "static/app.js", "login"])} HTTP/1.1" '
                   f'{status} {rng.randint(100, 9000)}')
    _write(data / "logs" / "access.log", "\n".join(acc) + "\n")

    _write(data / "notes" / "meeting.md",
           "# Weekly sync 2026-10-10\n\nAttendees: Maria, Tomasz, Lena\n\n## Discussion\n"
           "- Refund rate is up 3% month over month; Maria suspects the new checkout flow.\n"
           "- Search latency regressed after the index rebuild. Tomasz to investigate.\n"
           "- Lena proposed moving the mailer to a queue; agreed in principle.\n\n## Decisions\n"
           "- Roll back the checkout change until refund cause is known (owner: Maria, by 2026-10-14).\n"
           "- Add a latency alert at 500ms p95 (owner: Tomasz, by 2026-10-12).\n"
           "- Write an RFC for the mailer queue (owner: Lena, by 2026-10-20).\n\n## Parking lot\n- Pricing page copy refresh.\n")
    _write(data / "notes" / "ideas.md", "# Ideas\n- cache product pages\n- nightly refund report\n- dark mode for admin\n")
    _write(data / "notes" / "roadmap.md", "# Roadmap Q4\n\n## October\n- checkout rollback\n- latency alerts\n\n## November\n"
                                          "- mailer queue\n- beta search GA\n\n## December\n- pricing refresh\n")
    _write(data / "docs" / "api.md",
           "# API\n\n## GET /api/items?page=N\nReturns a page of items. Fields: id, title, owner{name,email}, metrics{views,likes}, archived.\n\n"
           "## POST /api/orders\nBody: {customer_id, items:[{sku, qty}]}. Returns 201 with order_id. Errors: 400 invalid sku, 402 payment.\n\n"
           "## GET /health\nReturns {\"status\":\"ok\"}.\n")
    _write(data / "docs" / "onboarding.md", "# Onboarding\n\n1. Install Python 3.11\n2. pip install -r data/requirements.txt\n"
                                            "3. Copy data/config.yaml to config.local.yaml and set the database url\n4. Run the tests\n")

    _write(data / "buggy" / "average.py", "def average(xs):\n    return sum(xs) / len(xs)\n\n\nif __name__ == '__main__':\n"
                                          "    print(average([1, 2, 3]))\n    print(average([]))\n")
    _write(data / "buggy" / "slugify.py", "import re\n\n\ndef slugify(text):\n    text = text.lower().strip()\n"
                                          "    text = re.sub(r'[^a-z0-9]', '-', text)\n    return text\n\n\n"
                                          "if __name__ == '__main__':\n    for s in ['Hello World', 'hello   world', 'A--B']:\n"
                                          "        print(repr(slugify(s)))\n")
    _write(data / "buggy" / "parse_date.py", "import re\n\n\ndef parse_date(s):\n    m = re.fullmatch(r'(\\d{4})-(\\d{2})-(\\d{2})', s)\n"
                                             "    if not m:\n        raise ValueError(f'bad date: {s}')\n"
                                             "    return tuple(int(g) for g in m.groups())\n\n\nif __name__ == '__main__':\n"
                                             "    print(parse_date('2026-10-05'))\n    print(parse_date('2026-1-5'))\n")
    _write(data / "buggy" / "slow.py", "import time\n\n\ndef has_dupes(xs):\n    for i in range(len(xs)):\n        for j in range(len(xs)):\n"
                                       "            if i != j and xs[i] == xs[j]:\n                return True\n    return False\n\n\n"
                                       "if __name__ == '__main__':\n    data = list(range(6000))\n    t = time.perf_counter()\n"
                                       "    print(has_dupes(data))\n    print(f'{time.perf_counter() - t:.3f}s')\n")
    _write(data / "buggy" / "config.ini", "[server]\nhost = 0.0.0.0\nport = 8080\n\n[db]\nurl = sqlite:///app.db\n")

    pkg = data / "minipkg"
    _write(pkg / "__init__.py", "from .money import to_cents, format_money\nfrom .dates import days_between\n"
                                "from .textutils import truncate\nfrom .stats import mean, median\n")
    _write(pkg / "money.py", "def to_cents(amount):\n    # BUG: float truncation; 19.99 -> 1998\n    return int(amount * 100)\n\n\n"
                             "def format_money(cents, currency='USD'):\n    return f'{cents / 100:.2f} {currency}'\n")
    _write(pkg / "dates.py", "from datetime import datetime\n\n\ndef days_between(a, b):\n    # BUG: ignores timezone suffix and fails on 'Z'\n"
                             "    da = datetime.fromisoformat(a)\n    db = datetime.fromisoformat(b)\n    return (db - da).days\n")
    _write(pkg / "textutils.py", "def truncate(s, n):\n    # BUG: counts bytes, not characters; breaks on unicode\n"
                                 "    b = s.encode('utf-8')\n    return b[:n].decode('utf-8', 'ignore') + ('...' if len(b) > n else '')\n")
    _write(pkg / "stats.py", "def mean(xs):\n    return sum(xs) / len(xs) if xs else 0.0\n\n\ndef median(xs):\n    s = sorted(xs)\n"
                             "    n = len(s)\n    if n == 0:\n        return 0.0\n    mid = n // 2\n    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2\n")
    _write(pkg / "cli.py", "import sys\nfrom .stats import mean, median\n\n\ndef main(argv=None):\n    nums = [float(x) for x in (argv or sys.argv[1:])]\n"
                           "    print('mean', mean(nums), 'median', median(nums))\n\n\nif __name__ == '__main__':\n    main()\n")
    _write(data / "minipkg_tests" / "test_minipkg.py",
           "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))\n"
           "from minipkg import to_cents, days_between, truncate, mean, median\n\n\n"
           "def test_to_cents():\n    assert to_cents(19.99) == 1999\n\n\ndef test_days_between_z():\n"
           "    assert days_between('2026-01-01T00:00:00Z', '2026-01-03T00:00:00Z') == 2\n\n\n"
           "def test_truncate_unicode():\n    assert truncate('zażółć gęślą', 5) == 'zażół...'\n\n\n"
           "def test_stats():\n    assert mean([1, 2, 3]) == 2 and median([3, 1, 2]) == 2\n")


if __name__ == "__main__":
    import sys

    build(Path(sys.argv[1]))
    print("built", sys.argv[1])
