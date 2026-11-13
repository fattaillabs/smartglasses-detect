"""aside 브라우저로 빙 이미지 검색 결과를 모아 원본 사진을 받는다.

사용: python src/web_collect.py <라벨> "검색어1" "검색어2" ...
결과: data/raw/web/<라벨>/<해시>.jpg, data/raw/web/<라벨>/urls.jsonl
검색 결과에는 엉뚱한 사진이 섞이므로 자른 뒤 시트로 검수해야 한다.
"""
import hashlib
import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"

JS = r'''
const q_%(n)s = %(q)s;
const t_%(n)s = await openTab("https://www.bing.com/images/search?q=" + encodeURIComponent(q_%(n)s) + "&qft=+filterui:imagesize-large+filterui:photo-photo&form=IRFLTR");
await sleep(2500);
for (let i = 0; i < 14; i++) { await t_%(n)s.evaluate(() => window.scrollBy(0, 3000)); await sleep(800); }
const r_%(n)s = await t_%(n)s.evaluate(() => Array.from(document.querySelectorAll("a.iusc")).map(a => { try { const m = JSON.parse(a.getAttribute("m")); return { u: m.murl, t: m.t, p: m.purl }; } catch (e) { return null; } }).filter(Boolean));
console.log("RESULT_JSON " + JSON.stringify(r_%(n)s));
await closeTab(t_%(n)s);
'''


def search(q: str, n: int):
    js = JS % {"n": n, "q": json.dumps(q)}
    out = subprocess.run(["aside", "repl", js], capture_output=True, text=True, timeout=180).stdout
    for line in out.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON "):])
    print("no result for", q, out[-300:], file=sys.stderr)
    return []


def fetch(job):
    url, dst = job
    name = hashlib.sha1(url.encode()).hexdigest()[:16]
    for f in dst.glob(name + ".*"):
        return f.name
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=25) as r:
            data = r.read(30_000_000)
    except Exception:  # noqa: BLE001
        return None
    if data[:3] == b"\xff\xd8\xff":
        ext = ".jpg"
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        ext = ".png"
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        ext = ".webp"
    else:
        return None
    if len(data) < 20_000:
        return None
    (dst / (name + ext)).write_bytes(data)
    return name + ext


def main():
    label, queries = sys.argv[1], sys.argv[2:]
    dst = ROOT / "data/raw/web" / label
    dst.mkdir(parents=True, exist_ok=True)
    seen, items = set(), []
    for n, q in enumerate(queries):
        res = search(q, n)
        new = [x for x in res if x["u"] not in seen]
        seen.update(x["u"] for x in new)
        for x in new:
            x["q"] = q
        items += new
        print(f"{q!r}: {len(res)} found, {len(new)} new", flush=True)
    with ThreadPoolExecutor(12) as ex:
        names = list(ex.map(fetch, [(x["u"], dst) for x in items]))
    ok = 0
    with open(dst / "urls.jsonl", "a") as f:
        for x, nm in zip(items, names):
            if nm:
                x["file"] = nm
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
                ok += 1
    print(f"downloaded {ok} / {len(items)}")


if __name__ == "__main__":
    main()
