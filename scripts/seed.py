"""Seed a demo tenant so the product is usable immediately after `up`.

Creates:
  - a platform admin:  admin@omnishop.local / admin12345
  - a demo merchant:   demo@omnishop.local  / demo12345  (org "Demo Shop Co")
  - a shop, a website widget channel, a few products (+variants), a policy doc
Embeddings are computed inline so retrieval works even without the worker.

Run:  docker compose exec api python -m scripts.seed
Idempotent: re-running reuses existing demo records and reprints the info.
"""
from __future__ import annotations
import json
import secrets

from app.db import no_tenant, tenant_tx, wait_ready
from app.providers.embeddings import to_pgvector
from app.providers.registry import get_embedder
from app.security import hash_password

ADMIN_EMAIL = "admin@omnishop.local"
DEMO_EMAIL = "demo@omnishop.local"


def _user(conn, email, pw, admin=False):
    row = conn.execute("SELECT id FROM app_user WHERE email=%s", (email,)).fetchone()
    if row:
        if admin:
            conn.execute("UPDATE app_user SET is_platform_admin=true WHERE id=%s", (row["id"],))
        return str(row["id"]), False
    row = conn.execute(
        "INSERT INTO app_user (email, password_hash, full_name, is_platform_admin) VALUES (%s,%s,%s,%s) RETURNING id",
        (email, hash_password(pw), email.split("@")[0], admin),
    ).fetchone()
    return str(row["id"]), True


def main():
    wait_ready(60)
    emb = get_embedder()

    with no_tenant() as conn:
        _user(conn, ADMIN_EMAIL, "admin12345", admin=True)
        demo_uid, _ = _user(conn, DEMO_EMAIL, "demo12345")
        org = conn.execute(
            "SELECT id FROM organization WHERE name=%s", ("Demo Shop Co",)
        ).fetchone()
        if org:
            org_id = str(org["id"])
        else:
            org_id = str(conn.execute(
                "INSERT INTO organization (name) VALUES ('Demo Shop Co') RETURNING id"
            ).fetchone()["id"])
            conn.execute("INSERT INTO subscription (organization_id, plan_code) VALUES (%s,'starter') ON CONFLICT DO NOTHING", (org_id,))

    with tenant_tx(org_id) as conn:
        conn.execute(
            "INSERT INTO membership (organization_id, user_id, role) VALUES (%s,%s,'owner') ON CONFLICT DO NOTHING",
            (org_id, demo_uid),
        )
        shop = conn.execute("SELECT id FROM shop WHERE organization_id=%s LIMIT 1", (org_id,)).fetchone()
        if shop:
            shop_id = str(shop["id"])
        else:
            shop_id = str(conn.execute(
                "INSERT INTO shop (organization_id, name) VALUES (%s,'Boutique Demo') RETURNING id", (org_id,)
            ).fetchone()["id"])

        # a demo bot with a custom persona
        bot = conn.execute("SELECT id FROM bot WHERE shop_id=%s LIMIT 1", (shop_id,)).fetchone()
        if bot:
            bot_id = str(bot["id"])
        else:
            bot_id = str(conn.execute(
                """INSERT INTO bot (organization_id, shop_id, name, persona, greeting, accent_color)
                   VALUES (%s,%s,%s,%s,%s,%s) RETURNING id""",
                (org_id, shop_id, "Trợ lý Boutique",
                 "Bạn là trợ lý bán hàng thân thiện của Boutique Demo, xưng \"mình\" và gọi khách là \"bạn\". "
                 "Tư vấn size và phối đồ ngắn gọn, luôn gợi ý thêm một sản phẩm phù hợp.",
                 "Xin chào! Mình là trợ lý của Boutique Demo, bạn cần tư vấn gì ạ?", "#6d7cff"),
            ).fetchone()["id"])

        ch = conn.execute("SELECT id, public_key FROM channel WHERE shop_id=%s AND kind='website' LIMIT 1", (shop_id,)).fetchone()
        if ch:
            public_key = ch["public_key"]
            conn.execute("UPDATE channel SET bot_id=coalesce(bot_id,%s) WHERE id=%s", (bot_id, ch["id"]))
        else:
            public_key = "web_" + secrets.token_urlsafe(12)
            conn.execute(
                """INSERT INTO channel (organization_id, shop_id, kind, name, public_key, config, bot_id)
                   VALUES (%s,%s,'website','Website widget',%s,%s,%s)""",
                (org_id, shop_id, public_key, json.dumps({"greeting": "Xin chào! Mình là trợ lý của Boutique Demo, bạn cần tư vấn gì ạ?"}), bot_id),
            )

        # products (only if none yet)
        has = conn.execute("SELECT count(*) AS n FROM product WHERE shop_id=%s", (shop_id,)).fetchone()["n"]
        if not has:
            products = [
                ("Áo thun cotton basic", "Áo thun cotton 100%, form regular, thoáng mát, nhiều màu.", 199000,
                 [("Size S / Trắng", 12), ("Size M / Trắng", 8), ("Size L / Đen", 3), ("Size M / Đen", 0)]),
                ("Quần jean slim-fit", "Quần jean nam co giãn nhẹ, dáng slim, màu xanh đậm.", 399000,
                 [("Size 29", 5), ("Size 30", 6), ("Size 31", 2), ("Size 32", 0)]),
                ("Váy hoa mùa hè", "Váy hoa nhẹ nhàng, chất liệu voan, phù hợp đi biển.", 349000,
                 [("Size S", 4), ("Size M", 7)]),
            ]
            for name, desc, price, variants in products:
                pid = str(conn.execute(
                    """INSERT INTO product (organization_id, shop_id, name, description, price, currency)
                       VALUES (%s,%s,%s,%s,%s,'VND') RETURNING id""",
                    (org_id, shop_id, name, desc, price),
                ).fetchone()["id"])
                vec = emb.embed_one(f"{name} {desc}")
                conn.execute("UPDATE product SET embedding=%s::vector WHERE id=%s", (to_pgvector(vec), pid))
                for vname, stock in variants:
                    conn.execute(
                        "INSERT INTO product_variant (organization_id, product_id, name, stock) VALUES (%s,%s,%s,%s)",
                        (org_id, pid, vname, stock),
                    )

        # knowledge (only if none yet)
        kb = conn.execute("SELECT id FROM knowledge_base WHERE shop_id=%s LIMIT 1", (shop_id,)).fetchone()
        if not kb:
            kb_id = str(conn.execute(
                "INSERT INTO knowledge_base (organization_id, shop_id, name) VALUES (%s,%s,'Default') RETURNING id",
                (org_id, shop_id),
            ).fetchone()["id"])
            docs = [
                ("Chính sách đổi trả", "Cửa hàng hỗ trợ đổi trả trong vòng 7 ngày kể từ khi nhận hàng, "
                 "với điều kiện sản phẩm còn nguyên tem mác và chưa qua sử dụng. Phí đổi trả do khách chịu nếu đổi vì lý do cá nhân."),
                ("Chính sách vận chuyển", "Giao hàng toàn quốc 2-4 ngày. Miễn phí ship cho đơn từ 500.000đ. "
                 "Nội thành Hà Nội và TP.HCM giao trong 24 giờ."),
                ("Phương thức thanh toán", "Chấp nhận COD (thanh toán khi nhận hàng), chuyển khoản ngân hàng, MoMo và ZaloPay."),
            ]
            for title, text in docs:
                did = str(conn.execute(
                    "INSERT INTO document (organization_id, knowledge_base_id, title, status) VALUES (%s,%s,%s,'ready') RETURNING id",
                    (org_id, kb_id, title),
                ).fetchone()["id"])
                vec = emb.embed_one(f"{title}. {text}")
                conn.execute(
                    "INSERT INTO chunk (organization_id, knowledge_base_id, document_id, ordinal, content, embedding) VALUES (%s,%s,%s,0,%s,%s::vector)",
                    (org_id, kb_id, did, text, to_pgvector(vec)),
                )

    print("\n=== OmniShop AI seed complete ===")
    print(f"Platform admin : {ADMIN_EMAIL} / admin12345")
    print(f"Merchant login : {DEMO_EMAIL} / demo12345")
    print(f"Dashboard      : http://localhost:8000/")
    print(f"Widget (thử)   : http://localhost:8000/widget.html?key={public_key}")
    print("Thử hỏi widget : 'Áo thun size M màu đen còn không?' hoặc 'Chính sách đổi trả thế nào?'\n")


if __name__ == "__main__":
    main()
