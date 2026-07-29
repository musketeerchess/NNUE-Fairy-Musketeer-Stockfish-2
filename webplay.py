"""
Web interface to play Musketeer against any trained net, with a model selector.

Loads every available model (symmetric model1-4 and asymmetric model1-4_asym)
and lets you switch which one you play against from a dropdown -- for testing
and comparing them. Includes the Musketeer placement phase and a 2-ply search.

The board is the standard Hawk/Unicorn Musketeer game. The symmetric models were
trained on it; the asymmetric models were trained for the custom-piece variant,
so selecting them here is a cross-domain test (they will play, but not at their
best -- their home is the asymmetric game).

    python webplay.py            # then open http://localhost:8000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train"))

import torch                                                         # noqa: E402
from features import (encode_fen, encode_fen_model3,                # noqa: E402
                      set_model3_types)
from arena import Arbiter, NetPlayer                                 # noqa: E402
from model1 import Model1                                           # noqa: E402
from model2 import Model2                                           # noqa: E402
from model3 import Model3                                           # noqa: E402
from model4 import Model4                                           # noqa: E402

LOCK = threading.Lock()
SETUP_FEN = ("********/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/******** "
             "w KQkq - 0 1")
SYM_M3 = ("P", "N", "B", "R", "Q", "K", "H", "U")
ASYM_M3 = ("P", "N", "B", "R", "Q", "K", "I", "J")

ARB: Arbiter | None = None
PLAYERS: dict = {}
CURRENT: str = ""
GAME_START = ""
DEPTH = 2


# --------------------------------------------------------------------------- #
# Model loading (all available checkpoints)
# --------------------------------------------------------------------------- #
def _load_one(path, cls, encoder, gating, m3types):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if cls is Model2:
        model = cls(hidden=ck.get("hidden", 1))
    else:
        model = cls(width=ck.get("width", 256), hidden=ck.get("hidden", 2))
    model.load_state_dict(ck["model"])
    name = os.path.basename(path)[:-3]
    p = NetPlayer(name, model, encoder, gating=gating, depth=DEPTH)
    p.m3types = m3types                       # set before encoding (model3 only)
    return name, p


def load_all_players() -> dict:
    reg = [
        ("models/model1.pt",      Model1, encode_fen,        False, None),
        ("models/model2.pt",      Model2, encode_fen,        True,  None),
        ("models/model3.pt",      Model3, encode_fen_model3, False, SYM_M3),
        ("models/model4.pt",      Model4, encode_fen,        False, None),
        ("models/model1_asym.pt", Model1, encode_fen,        False, None),
        ("models/model2_asym.pt", Model2, encode_fen,        True,  None),
        ("models/model3_asym.pt", Model3, encode_fen_model3, False, ASYM_M3),
        ("models/model4_asym.pt", Model4, encode_fen,        False, None),
    ]
    players: dict = {}
    for path, cls, enc, gating, m3 in reg:
        if os.path.exists(path):
            name, p = _load_one(path, cls, enc, gating, m3)
            players[name] = p
    # fall back to smoke checkpoints if no full models exist
    if not players:
        for path in glob.glob("models/model*_smoke.pt"):
            cls = {"1": Model1, "2": Model2, "3": Model3, "4": Model4}[
                os.path.basename(path)[5]]
            enc = encode_fen_model3 if cls is Model3 else encode_fen
            name, p = _load_one(path, cls, enc, cls is Model2,
                                SYM_M3 if cls is Model3 else None)
            players[name] = p
    return players


# --------------------------------------------------------------------------- #
# Board / rules helpers (placement phase preserved)
# --------------------------------------------------------------------------- #
def _row(hfile, ufile, upper):
    cells = ["*"] * 8
    cells[hfile] = "H" if upper else "h"
    cells[ufile] = "U" if upper else "u"
    return "".join(cells)


def build_start(white, black):
    wr = _row(white["h"], white["u"], True)
    br = _row(black["h"], black["u"], False)
    return (f"{br}/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/{wr} w KQkq - 0 1")


def validate(h, u):
    if h == u:
        return False, "Hawk and Unicorn must go on different files."
    if (h == 4 and u in (0, 7)) or (u == 4 and h in (0, 7)):
        return False, ("Can't place one piece behind the king and the other "
                       "behind a rook. Pick again.")
    return True, ""


def current_player():
    return PLAYERS[CURRENT]


def state(stack):
    fen, in_check = ARB.board(GAME_START, stack)
    legal = ARB.legal_moves(GAME_START, stack)
    stm = fen.split()[1]
    grouped: dict = {}
    if stm == "w":
        for m in legal:
            grouped.setdefault(m[:2], []).append(m)
    status, msg = "ok", ""
    if not legal:
        status = "over"
        msg = ("Checkmate — " + ("you lose" if stm == "w" else "you win") + "!") \
            if in_check else "Stalemate — draw."
    elif in_check and stm == "w":
        msg = "You are in check!"
    return {"phase": "play", "fen": fen, "moves": stack, "legal": grouped,
            "turn": stm, "status": status, "msg": msg, "model": CURRENT}


def do_move(stack, frm, to, promo):
    legal = ARB.legal_moves(GAME_START, stack)
    cands = [m for m in legal if m[:2] == frm and m[2:4] == to]
    if not cands:
        return None, "illegal"
    mv = cands[0]
    if promo:
        pc = [m for m in cands if m.endswith(promo)]
        if pc:
            mv = pc[0]
    stack = stack + [mv]
    if not ARB.legal_moves(GAME_START, stack):
        return stack, mv
    fen, _ = ARB.board(GAME_START, stack)
    if fen.split()[1] == "b":
        bl = ARB.legal_moves(GAME_START, stack)
        if bl:
            p = current_player()
            if getattr(p, "m3types", None):
                set_model3_types(p.m3types)         # correct geometry per model
            stack = stack + [p.choose(ARB, GAME_START, stack, bl)]
    return stack, mv


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/models":
            self._send(200, json.dumps({"models": sorted(PLAYERS), "current": CURRENT}))
        elif self.path == "/api/new":
            self._send(200, json.dumps({
                "phase": "setup", "fen": SETUP_FEN, "model": CURRENT,
                "msg": "Place your Hawk: click a square on your back row."}))
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        global GAME_START, CURRENT
        if self.path == "/api/select":
            name = self._body().get("name")
            if name in PLAYERS:
                CURRENT = name
                self._send(200, json.dumps({"ok": True, "current": CURRENT}))
            else:
                self._send(200, json.dumps({"error": "unknown model"}))
            return
        if self.path == "/api/start":
            req = self._body()
            h, u = int(req["h"]), int(req["u"])
            ok, why = validate(h, u)
            if not ok:
                self._send(200, json.dumps({"error": why})); return
            with LOCK:
                GAME_START = build_start({"h": h, "u": u}, {"h": 2, "u": 5})
                st = state([])
            self._send(200, json.dumps(st)); return
        if self.path == "/api/move":
            req = self._body()
            with LOCK:
                new_stack, info = do_move(req.get("moves", []), req["from"],
                                          req["to"], req.get("promo"))
                if new_stack is None:
                    self._send(200, json.dumps({"error": "illegal move"})); return
                out = state(new_stack)
            out["last"] = info
            self._send(200, json.dumps(out)); return
        self._send(404, "not found", "text/plain")


PAGE = r"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Musketeer Chess — model tester</title>
<style>
 :root{--l:#efe7d2;--d:#b28a5c;--sel:#f6e05e;}
 *{box-sizing:border-box} body{margin:0;font-family:system-ui,Arial;background:#232323;color:#eee;
   display:flex;flex-direction:column;align-items:center;gap:8px;padding:14px}
 h1{font-size:18px;margin:4px} .sub{color:#aaa;font-size:13px;margin:-4px 0 4px;text-align:center;max-width:600px}
 .bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;justify-content:center}
 select{padding:6px 10px;border-radius:6px;background:#333;color:#fff;border:1px solid #555;font-size:14px}
 .tag{font-size:12px;padding:2px 8px;border-radius:10px} .sym{background:#2a6}.asym{background:#a63}
 #board{display:grid;grid-template-columns:repeat(8,56px);
   grid-template-rows:46px repeat(8,56px) 46px;border:3px solid #111;border-radius:6px;overflow:hidden}
 .sq{width:56px;height:56px;display:flex;align-items:center;justify-content:center;
   font-size:40px;cursor:pointer;position:relative;user-select:none}
 .light{background:var(--l)} .dark{background:var(--d)}
 .sq.sel{box-shadow:inset 0 0 0 4px var(--sel)}
 .sq.tgt::after{content:"";position:absolute;width:20px;height:20px;border-radius:50%;background:rgba(30,120,30,.55)}
 .wp{color:#fff;text-shadow:0 0 2px #000,0 1px 2px #000} .bp{color:#111}
 .stage{height:46px;background:repeating-linear-gradient(45deg,#2a2a2a,#2a2a2a 6px,#333 6px,#333 12px);
   display:flex;align-items:center;justify-content:center;font-size:26px;position:relative}
 .stage.pick{cursor:pointer} .stage.pick:hover{outline:3px solid #f6e05e;outline-offset:-3px}
 .stage .filemark{position:absolute;bottom:1px;right:3px;font-size:9px;color:#777}
 .extra{width:44px;height:44px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:26px}
 .we{background:#f3f3f0;box-shadow:0 0 0 3px #3aa06a inset,0 1px 3px #000}
 .be{background:#20242c;box-shadow:0 0 0 3px #4aa3ff inset,0 1px 3px #000}
 #msg{min-height:22px;font-weight:600;text-align:center}
 .btns{display:flex;gap:10px} button{padding:8px 14px;border:0;border-radius:6px;background:#4a7;color:#fff;font-weight:600;cursor:pointer}
 button.sec{background:#555} #startBtn{display:none}
</style></head><body>
<h1>♞ Musketeer Chess — Model Tester</h1>
<div class=bar>
  <label>Opponent model: <select id=modelsel onchange=selectModel()></select></label>
  <span id=modeltag class=tag></span>
</div>
<div class=sub id=hint></div>
<div id=board></div>
<div id=msg></div>
<div class=btns>
  <button id=startBtn onclick=startGame()>Start game ▶</button>
  <button class=sec onclick=newGame()>New game / re-place</button>
</div>
<script>
let S=null, sel=null, phase='setup', place={h:null,u:null}, models=[], current='';
const G={p:'♟',r:'♜',n:'♞',b:'♝',q:'♛',k:'♚'}, EMO={h:'🦅',u:'🦄'};
function parseFEN(f){const rows=f.split(' ')[0].split('/');return {black_wait:rows[0],board:rows.slice(1,9),white_wait:rows[9]};}
function expand(s){const a=[];for(const ch of s){if(/\d/.test(ch)){for(let k=0;k<+ch;k++)a.push(null);}else if(ch==='*')a.push(null);else a.push(ch);}while(a.length<8)a.push(null);return a;}
function pieceHTML(p){const w=p===p.toUpperCase(),low=p.toLowerCase();if(EMO[low])return `<span class="extra ${w?'we':'be'}">${EMO[low]}</span>`;return `<span class="${w?'wp':'bp'}">${G[low]||p}</span>`;}
function tag(){const t=document.getElementById('modeltag');const asym=current.includes('asym');
  t.className='tag '+(asym?'asym':'sym'); t.textContent=asym?'asymmetric (cross-domain here)':'symmetric (in-domain)';}
async function loadModels(){const j=await(await fetch('/api/models')).json();models=j.models;current=j.current;
  const s=document.getElementById('modelsel');s.innerHTML=models.map(m=>`<option ${m===current?'selected':''}>${m}</option>`).join('');tag();}
async function selectModel(){current=document.getElementById('modelsel').value;
  await fetch('/api/select',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:current})});
  tag();document.getElementById('msg').textContent='Now playing against: '+current;}
function draw(){
  const P=parseFEN(S.fen),bd=document.getElementById('board');bd.innerHTML='';
  const bw=expand(P.black_wait);
  for(let c=0;c<8;c++){const d=document.createElement('div');d.className='stage';if(bw[c])d.innerHTML=pieceHTML(bw[c]);
    d.innerHTML+=`<span class=filemark>${String.fromCharCode(97+c)}9</span>`;bd.appendChild(d);}
  for(let r=0;r<8;r++){const cells=expand(P.board[r]);
    for(let c=0;c<8;c++){const sqName=String.fromCharCode(97+c)+(8-r);const d=document.createElement('div');
      d.className='sq '+((r+c)%2?'dark':'light');d.dataset.sq=sqName;if(cells[c])d.innerHTML=pieceHTML(cells[c]);
      if(phase==='play'&&sel){if(sel===sqName)d.classList.add('sel');if((S.legal[sel]||[]).some(m=>m.slice(2,4)===sqName))d.classList.add('tgt');}
      d.onclick=()=>click(sqName);bd.appendChild(d);}}
  const ww=expand(P.white_wait);
  for(let c=0;c<8;c++){const d=document.createElement('div');d.className='stage';let show=ww[c];
    if(phase==='setup'){d.classList.add('pick');if(place.h===c)show='H';else if(place.u===c)show='U';else show=null;d.onclick=()=>placeClick(c);}
    if(show)d.innerHTML=pieceHTML(show);d.innerHTML+=`<span class=filemark>${String.fromCharCode(97+c)}0</span>`;bd.appendChild(d);}
  updateHint();
}
function updateHint(){const hint=document.getElementById('hint'),sb=document.getElementById('startBtn');
  if(phase==='setup'){let step=place.h===null?'Click a bottom-row square to place your 🦅 Hawk.':place.u===null?'Now place your 🦄 Unicorn.':'Both placed — press “Start game”.';
    hint.innerHTML='<b>Placement phase.</b> '+step;sb.style.display=(place.h!==null&&place.u!==null)?'inline-block':'none';}
  else{hint.textContent='Click a piece, then its destination. Switch the opponent model anytime from the dropdown.';
    document.getElementById('msg').textContent=S.msg||(S.turn==='w'?'Your move.':'');}}
function placeClick(f){if(place.h===null)place.h=f;else if(place.u===null){if(f===place.h)return;place.u=f;}else place={h:f,u:null};draw();}
async function startGame(){const r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({h:place.h,u:place.u})});
  const j=await r.json();if(j.error){document.getElementById('msg').textContent=j.error;return;}S=j;phase='play';sel=null;draw();}
function click(sq){if(phase!=='play'||S.status==='over')return;
  if(sel&&(S.legal[sel]||[]).some(m=>m.slice(2,4)===sq)){move(sel,sq);sel=null;return;}sel=(S.legal[sq]?sq:null);draw();}
async function move(frm,to){let promo=null;const ms=(S.legal[frm]||[]).filter(m=>m.slice(2,4)===to);
  if(ms.length>1&&ms.some(m=>m.length>4))promo=(prompt('Promote to (q,r,b,n,h,u):','q')||'q').toLowerCase();
  document.getElementById('msg').textContent=current+' thinking…';
  const res=await fetch('/api/move',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({moves:S.moves,from:frm,to:to,promo:promo})});
  const j=await res.json();if(j.error){document.getElementById('msg').textContent=j.error;return;}S=j;draw();}
async function newGame(){S=await(await fetch('/api/new')).json();phase='setup';place={h:null,u:null};sel=null;draw();}
(async()=>{await loadModels();await newGame();})();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="model3")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()
    global ARB, PLAYERS, CURRENT, DEPTH
    DEPTH = args.depth
    PLAYERS = load_all_players()
    if not PLAYERS:
        raise SystemExit("no models found in models/")
    CURRENT = args.net if args.net in PLAYERS else sorted(PLAYERS)[0]
    ARB = Arbiter()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"Musketeer model tester on http://localhost:{args.port}  "
          f"({len(PLAYERS)} models: {', '.join(sorted(PLAYERS))})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
