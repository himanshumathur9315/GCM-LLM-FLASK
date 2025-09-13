from getpass import getpass

BASE_URL = "https://work.greyorange.com/confluence"   # <- your Confluence base
ROOT_ID  = "35987115"                                 # <- from your screenshot

AUTH_MODE = "bearer"   # "bearer" (Personal Access Token) or "basic"
VERIFY_TLS = False      # set to False if your Confluence uses self-signed certs

CONFLUENCE_TOKEN = "CONFLUENCE_TOKEN"
USER = PASSWORD = None

# if AUTH_MODE == "bearer":
#     CONFLUENCE_TOKEN = getpass("Paste Confluence Personal Access Token: ")
#     USER = PASSWORD = None
# else:
#     USER = input("Username: ").strip()
#     PASSWORD = getpass("Password / API token: ")
#     CONFLUENCE_TOKEN = None


import requests, html2text, json, re, time, math
from typing import Dict, Any, Iterable, Optional, List

def build_session(auth_mode: str, token: Optional[str], user: Optional[str], password: Optional[str], verify_tls: bool):
    s = requests.Session()
    s.verify = verify_tls
    s.headers.update({"Accept": "application/json"})
    if auth_mode == "bearer":
        s.headers.update({"Authorization": f"Bearer {token}"})
    elif auth_mode == "basic":
        s.auth = (user, password)
    else:
        raise ValueError("AUTH_MODE must be 'bearer' or 'basic'")
    return s

def get_page(base: str, sess: requests.Session, page_id: str) -> Dict[str, Any]:
    # Prefer storage (canonical), fall back to view/export_view
    expansions = "body.storage,body.view,body.export_view,version,_links"
    url = f"{base}/rest/api/content/{page_id}"
    r = sess.get(url, params={"expand": expansions})
    r.raise_for_status()
    return r.json()

def iter_descendants(base: str, sess: requests.Session, ancestor_id: str, limit: int = 5000) -> Iterable[Dict[str, Any]]:
    """
    Uses CQL: ancestor=<ID> and type=page
    """
    start = 0
    expansions = "body.storage,version,_links"
    cql = f"ancestor={ancestor_id} and type=page"
    while True:
        url = f"{base}/rest/api/content/search"
        params = {"cql": cql, "expand": expansions, "limit": limit, "start": start}
        r = sess.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if not results:
            break
        for item in results:
            yield item
        start = data.get("start", 0) + len(results)
        if start >= data.get("size", 0) and not data.get("_links", {}).get("next"):
            break
        time.sleep(0.05)  # gentle pacing

def html_to_markdown(html: str) -> str:
    conv = html2text.HTML2Text()
    conv.ignore_images = True
    conv.ignore_emphasis = False
    conv.body_width = 0
    md = conv.handle(html or "")
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md

def extract_body_md(obj: Dict[str, Any]) -> str:
    body = obj.get("body", {}) or {}
    html = None
    if "storage" in body and body["storage"] and "value" in body["storage"]:
        html = body["storage"]["value"]
    elif "view" in body and body["view"] and "value" in body["view"]:
        html = body["view"]["value"]
    elif "export_view" in body and body["export_view"] and "value" in body["export_view"]:
        html = body["export_view"]["value"]
    return html_to_markdown(html or "")

# def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
#     """
#     Simple char-based chunking with overlap; good enough for first pass.
#     """
#     text = text.strip()
#     if not text:
#         return []
#     if len(text) <= max_chars:
#         return [text]
#     chunks = []
#     i = 0
#     step = max_chars - overlap
#     while i < len(text):
#         chunk = text[i:i+max_chars]
#         chunks.append(chunk)
#         i += step
#     return chunks

def chunk_text(text: str, max_chars: int = None, overlap: int = 0) -> List[str]:
    text = text.strip()
    return [text] if text else []

sess = build_session(AUTH_MODE, CONFLUENCE_TOKEN, USER, PASSWORD, VERIFY_TLS)
print("Session ready.")

import os, datetime

OUT_PATH = "confluence_dump.jsonl"
total_pages = 0
total_chunks = 0

with open(OUT_PATH, "w", encoding="utf-8") as out:
    # root page
    root = get_page(BASE_URL, sess, ROOT_ID)
    root_id = str(root.get("id"))
    root_title = (root.get("title") or "").strip() or "Untitled"
    root_url = (root.get("_links", {}) or {}).get("webui", "")
    root_md = extract_body_md(root)

    for idx, piece in enumerate(chunk_text(root_md, max_chars=2000, overlap=200)):
        rec = {"id": f"{root_id}-{idx}", "title": root_title, "url": root_url, "text": piece}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total_chunks += 1
    total_pages += 1

    # descendants
    for obj in iter_descendants(BASE_URL, sess, ROOT_ID):
        pid = str(obj.get("id"))
        title = (obj.get("title") or "").strip() or "Untitled"
        url = (obj.get("_links", {}) or {}).get("webui", "")
        md = extract_body_md(obj)
        pieces = chunk_text(md, max_chars=2000, overlap=200)
        for idx, piece in enumerate(pieces):
            rec = {"id": f"{pid}-{idx}", "title": title, "url": url, "text": piece}
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total_chunks += 1
        total_pages += 1

print(f"✅ Wrote {total_chunks} chunks from {total_pages} pages → {OUT_PATH}")



#########################################################


# import json, re, math, hashlib, random
# from collections import Counter

# IN_PATH  = "confluence_dump.jsonl"
# OUT_PATH = "sft_instructions.jsonl"

# # --- basic sentence/word split ---
# _SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
# _WORD = re.compile(r"[A-Za-z0-9']+")
# _STOP = set("a an and are as at be by for from has have if in into is it its of on or such "
#             "that the their this to was were will with without within while about can not "
#             "we you your our they he she i them those these there here than then over under".split())

# def sent_tokenize(text): 
#     paras = [p.strip() for p in text.split("\n") if p.strip()]
#     sents = []
#     for p in paras: sents.extend(_SENT_SPLIT.split(p))
#     return [re.sub(r"\s+", " ", s.strip(" •-\t")) for s in sents if len(s.split()) >= 4]

# def word_tokenize(text): return [w.lower() for w in _WORD.findall(text)]

# def score_sentences(sentences):
#     words = [w for s in sentences for w in word_tokenize(s) if w not in _STOP]
#     if not words: return [1.0]*len(sentences)
#     freqs = Counter(words)
#     maxf = max(freqs.values())
#     for k in freqs: freqs[k] /= maxf
#     scores = []
#     for s in sentences:
#         toks = [w for w in word_tokenize(s) if w not in _STOP]
#         base = sum(freqs.get(t,0) for t in toks)/(math.log(len(toks)+1) or 1)
#         if re.search(r"\d|:", s): base*=1.1
#         scores.append(base)
#     return scores

# def top_k_sentences(sents,k=5,max_chars=2000):
#     if not sents: return []
#     scores = score_sentences(sents)
#     order = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)
#     chosen,total=[],0
#     for i in order:
#         s = sents[i]
#         if s in chosen: continue
#         if total+len(s)>max_chars and chosen: break
#         chosen.append(s); total+=len(s)
#         if len(chosen)>=k: break
#     return [s for s in sents if s in chosen]

# # --- builders ---
# def build_summary_example(title, text):
#     sents = sent_tokenize(text)
#     summary = " ".join(top_k_sentences(sents))
#     if not summary: return None
#     return {"instruction":"Summarize the following documentation section clearly and concisely.",
#             "input":text,"output":summary,"meta":{"type":"summary","title":title}}

# def build_takeaways_example(title, text):
#     sents = sent_tokenize(text)
#     bullets = top_k_sentences(sents)
#     if not bullets: return None
#     return {"instruction":"List the 3–5 most important takeaways from the context.",
#             "input":text,"output":"- "+"\n- ".join(bullets),"meta":{"type":"key_points","title":title}}

# def build_titleqa_example(title, text):
#     sents = sent_tokenize(text)
#     summary = " ".join(top_k_sentences(sents))
#     if not summary: return None
#     return {"instruction":f'What does the Confluence page section titled "{title}" cover?',
#             "input":text,"output":summary,"meta":{"type":"title_qa","title":title}}

# # --- process ---
# builders=[build_summary_example, build_takeaways_example, build_titleqa_example]
# rng=random.Random(1337)
# seen=set(); n_in=n_out=0

# with open(IN_PATH,"r",encoding="utf-8") as f, open(OUT_PATH,"w",encoding="utf-8") as g:
#     for line in f:
#         rec=json.loads(line); n_in+=1
#         title=(rec.get("title") or "Untitled").strip()
#         text =(rec.get("text") or "").strip()
#         if not text or len(text.split())<40: continue
#         order=list(range(len(builders))); rng.shuffle(order)
#         built=[]
#         for idx in order:
#             ex=builders[idx](title,text)
#             if ex:
#                 key=hashlib.md5(json.dumps([ex["instruction"],ex["input"],ex["output"]],ensure_ascii=False).encode()).hexdigest()
#                 if key not in seen:
#                     seen.add(key); built.append(ex)
#             if len(built)>=2: break
#         for ex in built:
#             g.write(json.dumps(ex,ensure_ascii=False)+"\n"); n_out+=1

# print(f"✅ Read {n_in} pages | Wrote {n_out} instruction examples → {OUT_PATH}")
