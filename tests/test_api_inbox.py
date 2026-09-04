"""Shared inbox: agent replies are delivered to the channel (website => widget),
recorded with the sender, and conversations can be assigned/claimed."""
import uuid

from conftest import requires_db


def _website_conversation(client, tenant):
    h, shop = tenant["headers"], tenant["shop_id"]
    ch = client.post("/api/channels", json={"shop_id": shop, "kind": "website", "name": "Web"}, headers=h).json()
    pk = ch["public_key"]
    sid = "sess_" + uuid.uuid4().hex[:8]
    client.post(f"/api/widget/{pk}/message", json={"session_id": sid, "text": "cho mình hỏi giá"})
    convs = client.get(f"/api/conversations?shop_id={shop}", headers=h).json()
    return convs[0]


@requires_db
def test_agent_reply_delivered_and_records_sender(client, tenant):
    h = tenant["headers"]
    conv = _website_conversation(client, tenant)
    r = client.post(f"/api/conversations/{conv['id']}/reply", json={"text": "Giá 200k bạn nhé"}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["delivered"] is True   # website => widget poll delivery
    msgs = client.get(f"/api/conversations/{conv['id']}/messages", headers=h).json()
    agent_msg = [m for m in msgs if m["role"] == "agent"][-1]
    assert agent_msg["content"] == "Giá 200k bạn nhé"
    assert agent_msg["sender"] == tenant["email"]        # who replied is recorded
    # replying auto-claims the conversation to the replier
    conv2 = next(c for c in client.get(f"/api/conversations?shop_id={tenant['shop_id']}", headers=h).json() if c["id"] == conv["id"])
    assert conv2["assignee"] == tenant["email"]


@requires_db
def test_assign_conversation(client, tenant):
    h = tenant["headers"]
    conv = _website_conversation(client, tenant)
    a = client.post(f"/api/conversations/{conv['id']}/assign", json={}, headers=h)
    assert a.status_code == 200 and a.json()["assignee"] == tenant["email"]
