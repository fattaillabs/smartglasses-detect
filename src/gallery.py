"""테스트 결과(eval/test_<모델>.json)를 브라우저로 넘겨 보는 갤러리로 만든다.

사용: python src/gallery.py eval/test_run2_band.json <출력 폴더> [--thr 0.5]
출력 폴더에 index.html 과 img/ 가 생긴다. 사진은 남의 얼굴이므로 이 폴더는 공유하지 않는다.
"""
import argparse
import html
import json
import shutil
from pathlib import Path

NAMES = {"none": "스마트안경 아님", "wayfarer": "레이밴 메타 웨이페어러", "headliner": "레이밴 메타 헤드라이너",
         "skyler": "레이밴 메타 스카일러", "adventurer": "메타 어드밴처러", "fury": "메타 퓨리", "nova": "메타 노바",
         "oakley_hstn": "오클리 메타 HSTN", "vanguard": "오클리 메타 뱅가드", "display": "메타 레이밴 디스플레이",
         "rokid": "로키드", "samsung": "삼성 인텔리전트 아이웨어", "xiaomi": "샤오미", "eveng2": "이븐 G2(카메라 없음)",
         "xreal": "엑스리얼(카메라 없음)"}
KIND = {**NAMES, "rayban_meta": "레이밴 메타", "oakley_meta": "오클리 메타 HSTN", "google_glass": "구글 글래스",
        "plain_glasses": "일반 안경", "no_glasses": "안경 없음", "other_smart": "스마트안경(제품 미상)",
        "scriber": "레이밴 메타 스크라이버", "blazer": "레이밴 메타 블레이저"}
PROD = NAMES

PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>스마트안경 판별 테스트 표본</title>
<style>
:root{--bg:#f4f3ef;--card:#fff;--ink:#1c1b19;--sub:#6b675f;--line:#dcd8cf;--ok:#2f6f4f;--bad:#b3261e;--accent:#1f4e79}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--card:#211f1c;--ink:#ece8df;--sub:#a39d91;--line:#38342e;--ok:#7cc29b;--bad:#ff8a80;--accent:#8db8e0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,"Apple SD Gothic Neo",sans-serif}
header{padding:20px 16px 12px;max-width:1500px;margin:0 auto}
h1{font-size:20px;margin:0 0 6px}p.sum{margin:0;color:var(--sub)}
.bar{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);z-index:2}
.bar div{max-width:1500px;margin:0 auto;padding:10px 16px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button{font:inherit;padding:6px 12px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:6px;cursor:pointer}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
.grid{max-width:1500px;margin:0 auto;padding:16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.c{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.c.wrong{border-color:var(--bad);border-width:2px}.c.flag{outline:3px solid #d9a400}
.c img{display:block;width:100%;height:auto;cursor:zoom-in}
.m{padding:8px 10px;display:flex;justify-content:space-between;gap:8px;align-items:center;font-size:13px}
.m b{font-weight:600}.p{font-variant-numeric:tabular-nums}.ok{color:var(--ok)}.bad{color:var(--bad)}
.s{color:var(--sub);font-size:12px;padding:0 10px 8px;display:flex;justify-content:space-between;gap:8px}
.s label{cursor:pointer;white-space:nowrap}
#out{width:100%;max-width:1468px;margin:0 16px 24px;height:90px;font:12px ui-monospace,monospace;display:none}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:5;cursor:zoom-out}
#zoom img{max-width:96vw;max-height:90vh;image-rendering:auto}
</style></head><body>
<header><h1>스마트안경 판별 테스트 표본</h1><p class="sum">__SUMMARY__</p></header>
<div class="bar"><div id="filters"></div></div>
<div class="grid" id="grid"></div>
<textarea id="out" readonly></textarea>
<div id="zoom"><img alt=""></div>
<script>
const D=__DATA__, THR=__THR__, K=__KIND__;
const flags=new Set(JSON.parse(localStorage.getItem('sg_flags')||'[]'));
const F=[['all','전체',r=>true],['wrong','틀린 것',r=>r.w],['smart','스마트안경',r=>r.y==1],['plain','일반 안경',r=>r.k=='plain_glasses'],['bare','안경 없음',r=>r.k=='no_glasses'],['nocam','카메라 없는 스마트안경',r=>r.k=='eveng2'||r.k=='xreal'],['near','아슬아슬(0.2~0.8)',r=>r.p>=.2&&r.p<=.8],['flag','내가 표시한 것',r=>flags.has(r.id)]];
let cur='wrong';
const fb=document.getElementById('filters'), grid=document.getElementById('grid'), out=document.getElementById('out');
function draw(){
  fb.innerHTML='';
  for(const [k,t,f] of F){const b=document.createElement('button');b.textContent=t+' '+D.filter(f).length;b.className=k==cur?'on':'';b.onclick=()=>{cur=k;draw()};fb.appendChild(b)}
  const ex=document.createElement('button');ex.textContent='표시한 목록 내보내기';ex.onclick=()=>{out.style.display='block';out.value=[...flags].join('\\n');out.select()};fb.appendChild(ex);
  const f=F.find(x=>x[0]==cur)[2];
  const rows=D.filter(f).sort((a,b)=>(b.w-a.w)||(Math.abs(a.p-THR)-Math.abs(b.p-THR)));
  grid.innerHTML=rows.map(r=>`<div class="c ${r.w?'wrong':''} ${flags.has(r.id)?'flag':''}" data-id="${r.id}">
<img loading="lazy" src="img/${r.id}.jpg" alt="">
<div class="m"><span>정답 <b>${K[r.k]||r.k}</b></span><span class="p ${r.w?'bad':'ok'}">스마트안경 확률 ${(r.p*100).toFixed(1)}% ${r.w?'· 틀림':''}</span></div>
<div class="s"><span>${r.src} · ${r.g}${r.pp?' · 제품 추정 '+r.pp+(r.tp?(r.tp==r.pp?' ✓':' (정답 '+r.tp+')'):''):''}</span><label><input type="checkbox" ${flags.has(r.id)?'checked':''}> 정답표 이상함</label></div></div>`).join('');
}
grid.addEventListener('change',e=>{const c=e.target.closest('.c');if(!c)return;const id=c.dataset.id;e.target.checked?flags.add(id):flags.delete(id);c.classList.toggle('flag',e.target.checked);localStorage.setItem('sg_flags',JSON.stringify([...flags]))});
const z=document.getElementById('zoom');
grid.addEventListener('click',e=>{if(e.target.tagName=='IMG'){z.firstChild.src=e.target.src;z.style.display='flex'}});
z.onclick=()=>z.style.display='none';
draw();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("out")
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    rows = json.load(open(a.result))
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    data = []
    for r in rows:
        src = Path(r["dir"]) / "band" / f"{r['id']}.jpg"
        shutil.copy(src, out / "img" / f"{r['id']}.jpg")
        wrong = (r["p"] >= a.thr) != bool(r["y"])
        data.append({"id": r["id"], "y": r["y"], "k": r["kind"], "p": r["p"], "w": int(wrong),
                     "src": "영상" if r["source"] == "yt_test" else "웹 사진", "g": r["group"][:16],
                     "pp": PROD.get(r.get("pred_prod")) if (r["p"] >= a.thr or r.get("pred_prod") in ("eveng2", "xreal")) else None,
                     "tp": PROD.get(r.get("prod")) if r.get("prod") not in (None, "none") else None})
    pos = [d for d in data if d["y"]]
    neg = [d for d in data if not d["y"]]
    miss = sum(d["w"] for d in pos)
    fa = sum(d["w"] for d in neg)
    summary = (f"학습·검증에 쓰지 않은 표본 {len(data)}장 · 스마트안경 {len(pos)}장 중 {miss}장 놓침 · "
               f"그 밖 {len(neg)}장 중 {fa}장 오탐 · 판정 기준 {a.thr:.2f} · 사진을 누르면 크게 보입니다")
    page = (PAGE.replace("__SUMMARY__", html.escape(summary)).replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__THR__", str(a.thr)).replace("__KIND__", json.dumps(KIND, ensure_ascii=False)))
    (out / "index.html").write_text(page, encoding="utf-8")
    print(out / "index.html", len(data), "miss", miss, "false alarm", fa)


if __name__ == "__main__":
    main()
