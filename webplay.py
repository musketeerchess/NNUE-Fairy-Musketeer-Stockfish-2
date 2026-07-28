"""
Web interface to play Musketeer against a trained net.

Includes the Musketeer **placement phase**: before the game, you place your Hawk
and Unicorn on your back row (choosing which pieces they gate behind); the model
places its two; then play begins.

Runs a tiny stdlib HTTP server (no dependencies): the browser shows a clickable
board; the server uses the private engine for legal moves/FENs and the chosen
net (1-ply) to reply.  Human plays White.

    python webplay.py            # then open http://localhost:8000
    python webplay.py --net model1 --port 8000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "train"))

from arena import Arbiter, load_players     # noqa: E402

LOCK = threading.Lock()
VARIANT_MEN = ("P:fmWfceFifmnD;N:N;B:B;R:R;Q:Q;E:FWDA;C:FWDsN;A:BN;F:B3vND;"
               "M:RN;H:ADGH;S:B2ND;U:NC;D:QN;L:B2N;K:KO2")
SETUP_FEN = ("********/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/******** "
             "w KQkq - 0 1")

ARB: Arbiter | None = None
PLAYER = None
GAME_START = ""          # the configured start FEN for the current game


def _row(hfile, ufile, upper):
    cells = ["*"] * 8
    cells[hfile] = "H" if upper else "h"
    cells[ufile] = "U" if upper else "u"
    return "".join(cells)


def build_start(white, black):
    """white/black = {'h': file, 'u': file} -> full 10-rank Musketeer FEN."""
    wr = _row(white["h"], white["u"], True)
    br = _row(black["h"], black["u"], False)
    return (f"{br}/rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR/{wr} "
            "w KQkq - 0 1")


def validate(h, u):
    if h == u:
        return False, "Hawk and Unicorn must go on different files."
    # Musketeer rule: not behind the king (e/file 4) AND a rook (a/h) together.
    if (h == 4 and u in (0, 7)) or (u == 4 and h in (0, 7)):
        return False, ("Can't place one piece behind the king and the other "
                       "behind a rook (would gate two in one castle). Pick again.")
    return True, ""


def state(stack):
    fen, in_check = ARB.board(GAME_START, stack)
    legal = ARB.legal_moves(GAME_START, stack)
    stm = fen.split()[1]
    grouped: dict[str, list[str]] = {}
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
            "turn": stm, "status": status, "msg": msg}


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
            stack = stack + [PLAYER.choose(ARB, GAME_START, stack, bl)]
    return stack, mv


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
        elif self.path == "/api/new":
            # enter placement phase
            self._send(200, json.dumps({
                "phase": "setup", "fen": SETUP_FEN,
                "msg": "Place your Hawk: click a square on your back row (bottom)."}))
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        global GAME_START
        if self.path == "/api/start":
            req = self._body()
            h, u = int(req["h"]), int(req["u"])
            ok, why = validate(h, u)
            if not ok:
                self._send(200, json.dumps({"error": why})); return
            with LOCK:
                # model places its two pieces (safe default: c & f files)
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
<title>Musketeer Chess — you vs your model</title>
<style>
 :root{--l:#efe7d2;--d:#b28a5c;--sel:#f6e05e;}
 *{box-sizing:border-box} body{margin:0;font-family:system-ui,Arial;background:#232323;color:#eee;
   display:flex;flex-direction:column;align-items:center;gap:8px;padding:14px}
 h1{font-size:18px;margin:4px} .sub{color:#aaa;font-size:13px;margin:-4px 0 4px;text-align:center;max-width:560px}
 #board{display:grid;grid-template-columns:repeat(8,56px);
   grid-template-rows:46px repeat(8,56px) 46px;border:3px solid #111;border-radius:6px;overflow:hidden}
 .sq{width:56px;height:56px;display:flex;align-items:center;justify-content:center;
   font-size:40px;cursor:pointer;position:relative;user-select:none}
 .light{background:var(--l)} .dark{background:var(--d)}
 .sq.sel{box-shadow:inset 0 0 0 4px var(--sel)}
 .sq.tgt::after{content:"";position:absolute;width:20px;height:20px;border-radius:50%;
   background:rgba(30,120,30,.55)}
 .wp{color:#fff;text-shadow:0 0 2px #000,0 1px 2px #000} .bp{color:#111}
 .stage{height:46px;background:repeating-linear-gradient(45deg,#2a2a2a,#2a2a2a 6px,#333 6px,#333 12px);
   display:flex;align-items:center;justify-content:center;font-size:26px;position:relative}
 .stage.pick{cursor:pointer} .stage.pick:hover{outline:3px solid #f6e05e;outline-offset:-3px}
 .stage .filemark{position:absolute;bottom:1px;right:3px;font-size:9px;color:#777}
 .extra{width:44px;height:44px;border-radius:50%;display:flex;align-items:center;justify-content:center;
   font-size:26px;line-height:1}
 .we{background:#f3f3f0;box-shadow:0 0 0 3px #3aa06a inset,0 1px 3px #000}
 .be{background:#20242c;box-shadow:0 0 0 3px #4aa3ff inset,0 1px 3px #000}
 .legend{display:flex;gap:16px;align-items:center;font-size:13px;color:#ccc}
 .legend .extra{width:30px;height:30px;font-size:18px}
 #msg{min-height:22px;font-weight:600;text-align:center}
 .btns{display:flex;gap:10px} button{padding:8px 14px;border:0;border-radius:6px;
   background:#4a7;color:#fff;font-weight:600;cursor:pointer} button.sec{background:#555}
 button:disabled{opacity:.4;cursor:default}
</style></head><body>
<h1>♞ Musketeer Chess — You (White) vs <span id=netname>model</span></h1>
<div class=sub id=hint></div>
<div class=legend>
  <span><span class="extra we">🦅</span> Hawk</span>
  <span><span class="extra we">🦄</span> Unicorn</span>
  <span style="color:#888">green ring = yours, blue ring = model's</span>
</div>
<div id=board></div>
<div id=msg></div>
<div class=btns>
  <button id=startBtn onclick=startGame() style="display:none">Start game ▶</button>
  <button class=sec onclick=newGame()>New game / re-place</button>
</div>
<script>
let S=null, sel=null, phase='setup', place={h:null,u:null};
const G={p:'♟',r:'♜',n:'♞',b:'♝',q:'♛',k:'♚'}, EMO={h:'🦅',u:'🦄'};
function parseFEN(f){const rows=f.split(' ')[0].split('/');
  return {black_wait:rows[0],board:rows.slice(1,9),white_wait:rows[9]};}
function expand(s){const a=[];for(const ch of s){
  if(/\d/.test(ch)){for(let k=0;k<+ch;k++)a.push(null);}
  else if(ch==='*')a.push(null); else a.push(ch);} while(a.length<8)a.push(null); return a;}
function pieceHTML(p){const w=p===p.toUpperCase(),low=p.toLowerCase();
  if(EMO[low])return `<span class="extra ${w?'we':'be'}">${EMO[low]}</span>`;
  return `<span class="${w?'wp':'bp'}">${G[low]}</span>`;}

function draw(){
  const P=parseFEN(S.fen),bd=document.getElementById('board');bd.innerHTML='';
  // top waiting = Black
  const bw=expand(P.black_wait);
  for(let c=0;c<8;c++){const d=document.createElement('div');d.className='stage';
    if(bw[c])d.innerHTML=pieceHTML(bw[c]);
    d.innerHTML+=`<span class=filemark>${String.fromCharCode(97+c)}9</span>`;bd.appendChild(d);}
  // 8 board rows
  for(let r=0;r<8;r++){const cells=expand(P.board[r]);
    for(let c=0;c<8;c++){const sqName=String.fromCharCode(97+c)+(8-r);
      const d=document.createElement('div');d.className='sq '+((r+c)%2?'dark':'light');d.dataset.sq=sqName;
      if(cells[c])d.innerHTML=pieceHTML(cells[c]);
      if(phase==='play'&&sel){if(sel===sqName)d.classList.add('sel');
        if((S.legal[sel]||[]).some(m=>m.slice(2,4)===sqName))d.classList.add('tgt');}
      d.onclick=()=>click(sqName);bd.appendChild(d);}}
  // bottom waiting = White  (clickable during setup)
  const ww=expand(P.white_wait);
  for(let c=0;c<8;c++){const d=document.createElement('div');d.className='stage';
    let show=ww[c];
    if(phase==='setup'){ d.classList.add('pick');
      if(place.h===c)show='H'; else if(place.u===c)show='U'; else show=null;
      d.onclick=()=>placeClick(c); }
    if(show)d.innerHTML=pieceHTML(show);
    d.innerHTML+=`<span class=filemark>${String.fromCharCode(97+c)}0</span>`;bd.appendChild(d);}
  updateHint();
}
function updateHint(){
  const hint=document.getElementById('hint'), sb=document.getElementById('startBtn');
  if(phase==='setup'){
    let step = place.h===null ? 'Click a square on your back row (bottom, striped) to place your 🦅 Hawk.'
             : place.u===null ? 'Now place your 🦄 Unicorn on another back-row square.'
             : 'Both placed! Press “Start game”. (Click a square to change, or “New game” to redo.)';
    hint.innerHTML='<b>Placement phase.</b> '+step+' The piece gates onto that file when the piece in front of it first moves.';
    sb.style.display=(place.h!==null&&place.u!==null)?'inline-block':'none';
    document.getElementById('msg').textContent='';
  } else {
    hint.textContent='Click a piece, then its destination. Your Hawk/Unicorn gate in automatically.';
    document.getElementById('msg').textContent=S.msg||(S.turn==='w'?'Your move.':'');
  }
}
function placeClick(f){
  if(place.h===null){place.h=f;}
  else if(place.u===null){ if(f===place.h){place.h=f;return;} place.u=f;}
  else { place={h:f,u:null}; }   // start over
  draw();
}
async function startGame(){
  const r=await fetch('/api/start',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({h:place.h,u:place.u})});
  const j=await r.json();
  if(j.error){document.getElementById('msg').textContent=j.error;return;}
  S=j; phase='play'; sel=null; draw();
}
function click(sq){ if(phase!=='play'||S.status==='over')return;
  if(sel&&(S.legal[sel]||[]).some(m=>m.slice(2,4)===sq)){move(sel,sq);sel=null;return;}
  sel=(S.legal[sq]?sq:null);draw();}
async function move(frm,to){let promo=null;
  const ms=(S.legal[frm]||[]).filter(m=>m.slice(2,4)===to);
  if(ms.length>1&&ms.some(m=>m.length>4)){promo=(prompt('Promote to (q,r,b,n,h,u):','q')||'q').toLowerCase();}
  document.getElementById('msg').textContent='Model thinking…';
  const res=await fetch('/api/move',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({moves:S.moves,from:frm,to:to,promo:promo})});
  const j=await res.json();if(j.error){document.getElementById('msg').textContent=j.error;return;}
  S=j;draw();}
async function newGame(){ S=await(await fetch('/api/new')).json(); phase='setup'; place={h:null,u:null}; sel=null; draw(); }
newGame();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="model3")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    global ARB, PLAYER, PAGE
    players = {p.name: p for p in load_players()}
    if args.net not in players:
        raise SystemExit(f"net '{args.net}' not found; have {list(players)}")
    PLAYER = players[args.net]
    PAGE = PAGE.replace("<span id=netname>model</span>",
                        f"<span id=netname>{args.net}</span>")
    ARB = Arbiter()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    print(f"Musketeer web UI on http://localhost:{args.port}  (net={args.net})",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
