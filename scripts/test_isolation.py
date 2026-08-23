"""Tenant-isolation test (ADR-008 verification requirement). Runnable without
pytest: `docker compose exec api python -m scripts.test_isolation`.

Proves that with Row-Level Security, org A cannot read org B's rows, and cannot
insert a row owned by another org (WITH CHECK).
"""
from __future__ import annotations
import sys
import uuid

from app.db import admin_tx, tenant_tx, wait_ready

failures = []


def check(name, cond):
    print(("PASS" if cond else "FAIL"), "-", name)
    if not cond:
        failures.append(name)


def main():
    wait_ready(60)
    sfx = uuid.uuid4().hex[:8]
    with admin_tx() as conn:
        a = str(conn.execute("INSERT INTO organization (name) VALUES (%s) RETURNING id", (f"A-{sfx}",)).fetchone()["id"])
        b = str(conn.execute("INSERT INTO organization (name) VALUES (%s) RETURNING id", (f"B-{sfx}",)).fetchone()["id"])

    try:
        with tenant_tx(a) as conn:
            conn.execute("INSERT INTO shop (organization_id, name) VALUES (%s,%s)", (a, f"ShopA-{sfx}"))
        with tenant_tx(b) as conn:
            conn.execute("INSERT INTO shop (organization_id, name) VALUES (%s,%s)", (b, f"ShopB-{sfx}"))

        # A sees only its own shop
        with tenant_tx(a) as conn:
            rows = conn.execute("SELECT name FROM shop").fetchall()
            names = {r["name"] for r in rows}
        check("A sees ShopA", f"ShopA-{sfx}" in names)
        check("A does NOT see ShopB (RLS read isolation)", f"ShopB-{sfx}" not in names)

        # A cannot even target B's shop by name
        with tenant_tx(a) as conn:
            r = conn.execute("SELECT count(*) AS n FROM shop WHERE name=%s", (f"ShopB-{sfx}",)).fetchone()
        check("A gets 0 rows querying B's shop directly", int(r["n"]) == 0)

        # A cannot insert a row owned by B (RLS WITH CHECK)
        blocked = False
        try:
            with tenant_tx(a) as conn:
                conn.execute("INSERT INTO shop (organization_id, name) VALUES (%s,%s)", (b, f"Sneaky-{sfx}"))
        except Exception:
            blocked = True
        check("A cannot insert a row owned by B (WITH CHECK)", blocked)

        # confirm the sneaky row never landed
        with tenant_tx(b) as conn:
            r = conn.execute("SELECT count(*) AS n FROM shop WHERE name=%s", (f"Sneaky-{sfx}",)).fetchone()
        check("Sneaky row absent in B", int(r["n"]) == 0)
    finally:
        with admin_tx() as conn:
            conn.execute("DELETE FROM organization WHERE id IN (%s,%s)", (a, b))

    if failures:
        print(f"\n{len(failures)} check(s) FAILED")
        sys.exit(1)
    print("\nAll isolation checks passed ✅")


if __name__ == "__main__":
    main()
