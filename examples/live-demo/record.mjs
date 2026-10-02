// Drive the live demo page over CDP, inject annotations, capture screencast frames.
// Usage: node record.mjs [--dry]   (--dry: no model calls, overlay/capture check only)
// TYPEVET_DEMO_UPLOAD: image for the upload step (default: the blurred PNG beside this file).
// TYPEVET_DEMO_CHROME: Chrome binary (default: google-chrome on PATH).
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const DRY = process.argv.includes("--dry");
const REC = path.dirname(new URL(import.meta.url).pathname);
const FR = path.join(REC, DRY ? "frames-dry" : "frames");
const PORT = 9333;
const UPLOAD = path.resolve(process.env.TYPEVET_DEMO_UPLOAD ?? path.join(REC, "insufficient-R01-blurred.png"));
fs.rmSync(FR, { recursive: true, force: true });
fs.mkdirSync(FR, { recursive: true });
const profile = path.join(REC, "profile-" + Date.now());

const chrome = spawn(process.env.TYPEVET_DEMO_CHROME ?? "google-chrome", [
  "--headless=new", `--remote-debugging-port=${PORT}`, `--user-data-dir=${profile}`,
  "--window-size=1920,1080", "--hide-scrollbars", "--force-device-scale-factor=1",
  "--no-first-run", "--no-default-browser-check", "--disable-extensions", "--mute-audio",
  "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });
chrome.stderr.on("data", () => {});

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function getTarget() {
  for (let i = 0; i < 100; i++) {
    try {
      const l = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json();
      const p = l.find((t) => t.type === "page");
      if (p) return p.webSocketDebuggerUrl;
    } catch {}
    await sleep(100);
  }
  throw new Error("chrome did not start");
}

const ws = new WebSocket(await getTarget());
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let id = 0; const pend = new Map(); const handlers = {};
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pend.has(m.id)) { const { res, rej } = pend.get(m.id); pend.delete(m.id); m.error ? rej(new Error(JSON.stringify(m.error))) : res(m.result); }
  else if (m.method && handlers[m.method]) handlers[m.method](m.params);
});
const send = (method, params = {}) => new Promise((res, rej) => { const i = ++id; pend.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); });
async function ev(expr) {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error("eval: " + JSON.stringify(r.exceptionDetails).slice(0, 400));
  return r.result.value;
}

// frames
const frames = []; let recording = false;
handlers["Page.screencastFrame"] = (p) => {
  send("Page.screencastFrameAck", { sessionId: p.sessionId }).catch(() => {});
  if (!recording) return;
  const f = path.join(FR, `f${String(frames.length).padStart(5, "0")}.jpg`);
  fs.writeFileSync(f, Buffer.from(p.data, "base64"));
  frames.push({ f, t: p.metadata.timestamp });
};

await send("Page.enable"); await send("Runtime.enable"); await send("DOM.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1920, height: 1080, deviceScaleFactor: 1, mobile: false });
await send("Network.enable"); await send("Network.setCacheDisabled", { cacheDisabled: true });
await send("Page.navigate", { url: `http://127.0.0.1:8765/?v=${Date.now()}` });
for (let i = 0; i < 100; i++) { if (await ev(`document.readyState==='complete' && !!document.querySelector('.thumb[data-id=R01]') && /Gemma 4/.test(document.getElementById('ident').textContent)`)) break; await sleep(100); }
await ev(`Promise.all([...document.images].map(i=>i.complete?1:new Promise(r=>{i.onload=i.onerror=r})))`);

// ---------- overlay ----------
const OVERLAY = String.raw`(() => {
const st = document.createElement('style');
st.textContent = ` + "`" + String.raw`
#ov-cap{position:fixed;left:50%;bottom:34px;transform:translate(-50%,20px);max-width:1380px;width:max-content;text-wrap:balance;
  background:rgba(12,17,28,.88);color:#fff;font:600 40px/1.3 Inter,system-ui,sans-serif;
  padding:22px 42px;border-radius:18px;box-shadow:0 10px 40px rgba(0,0,0,.35);opacity:0;transition:opacity .5s,transform .5s;
  z-index:2147483646;text-align:center;pointer-events:none;letter-spacing:.2px}
#ov-cap.on{opacity:1;transform:translate(-50%,0)}
#ov-step{position:fixed;right:28px;bottom:34px;background:rgba(12,17,28,.78);color:#fff;font:700 24px system-ui,sans-serif;
  padding:10px 20px;border-radius:12px;z-index:2147483646;opacity:0;transition:opacity .5s;pointer-events:none}
#ov-step.on{opacity:1}
#ov-ring{position:fixed;border:5px solid #f59e0b;border-radius:14px;z-index:2147483645;pointer-events:none;opacity:0;
  transition:opacity .35s;animation:ovpulse 1.1s ease-in-out infinite}
#ov-ring.on{opacity:1}
@keyframes ovpulse{0%{box-shadow:0 0 0 0 rgba(245,158,11,.75)}70%{box-shadow:0 0 0 22px rgba(245,158,11,0)}100%{box-shadow:0 0 0 0 rgba(245,158,11,0)}}
#ov-card{position:fixed;inset:0;background:linear-gradient(135deg,#0b1220 0%,#123a2e 100%);color:#fff;z-index:2147483647;
  display:flex;flex-direction:column;align-items:center;justify-content:center;gap:28px;opacity:0;transition:opacity .8s;pointer-events:none;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;text-align:center;padding:0 120px}
#ov-card.on{opacity:1}
#ov-card h1{font-size:76px;margin:0;font-weight:800;line-height:1.15}
#ov-card p{font-size:38px;margin:0;color:#b7f0d5;font-weight:500}
#ov-click{position:fixed;width:34px;height:34px;margin:-17px 0 0 -17px;border-radius:50%;background:rgba(245,158,11,.55);
  z-index:2147483646;pointer-events:none;transform:scale(.3);opacity:0}
#ov-click.go{animation:ovclick .6s ease-out}
@keyframes ovclick{0%{transform:scale(.3);opacity:1}100%{transform:scale(2.6);opacity:0}}
` + "`" + String.raw`;
document.head.appendChild(st);
const mk=(id)=>{const d=document.createElement('div');d.id=id;document.body.appendChild(d);return d};
const cap=mk('ov-cap'), step=mk('ov-step'), ring=mk('ov-ring'), card=mk('ov-card'), clk=mk('ov-click');
const Z=parseFloat(getComputedStyle(document.documentElement).zoom)||1;
[cap,step,ring,card,clk].forEach(d=>d.style.zoom=String(1/Z));
let target=null, pad=10;
function place(){ if(target){const r=target.getBoundingClientRect();
  ring.style.left=(r.left-pad)+'px';ring.style.top=(r.top-pad)+'px';ring.style.width=(r.width+2*pad)+'px';ring.style.height=(r.height+2*pad)+'px';}
  requestAnimationFrame(place);}
requestAnimationFrame(place);
const q=(s)=>typeof s==='string'?document.querySelector(s):s;
window.__ov={
  cap(t){ if(cap.classList.contains('on')&&cap.textContent!==t){cap.classList.remove('on');setTimeout(()=>{cap.textContent=t;cap.classList.add('on')},350)} else {cap.textContent=t;cap.classList.add('on')} },
  capOff(){cap.classList.remove('on')},
  step(n,m){step.textContent='Step '+n+' of '+m;step.classList.add('on')},
  stepOff(){step.classList.remove('on')},
  ring(s,p){target=q(s);pad=p==null?10:p;ring.classList.add('on')},
  ringOff(){ring.classList.remove('on')},
  card(h,p){card.innerHTML='<h1>'+h+'</h1>'+(p?'<p>'+p+'</p>':'');card.classList.add('on')},
  cardOff(){card.classList.remove('on')},
  click(s){const e=q(s);const r=e.getBoundingClientRect();clk.style.left=(r.left+r.width/2)+'px';clk.style.top=(r.top+r.height/2)+'px';
    clk.classList.remove('go');void clk.offsetWidth;clk.classList.add('go');e.click();},
  scrollTo(y,ms){return new Promise(res=>{const y0=scrollY,t0=performance.now();const f=(t)=>{const k=Math.min(1,(t-t0)/ms);
    const e=k<.5?2*k*k:1-Math.pow(-2*k+2,2)/2;scrollTo(0,y0+(y-y0)*e);k<1?requestAnimationFrame(f):res()};requestAnimationFrame(f)})},
  scrollToEl(s,off,ms){const e=q(s);const r0=e.getBoundingClientRect().top;const y0=scrollY;scrollTo(0,y0+40);const r1=e.getBoundingClientRect().top;const d=scrollY-y0;scrollTo(0,y0);const k=d>0?(r0-r1)/d:1;const y=y0+(r0-off)/(k||1);return this.scrollTo(Math.max(0,y),ms)},
};
})()`;
await ev(OVERLAY);

const O = (js) => ev(`window.__ov.${js}`);
const cap = (t) => O(`cap(${JSON.stringify(t)})`);
const ring = (s, p) => O(`ring(${JSON.stringify(s)}${p != null ? "," + p : ""})`);
const click = (s) => O(`click(${JSON.stringify(s)})`);
const step = (n) => O(`step(${n},8)`);
const log = []; let calls = 0;
const idle = async () => { for (let i = 0; i < 600; i++) { if (await ev(`!document.querySelector('#out .wait')`)) return; await sleep(100); } throw new Error("timeout"); };
async function answer() {
  await idle();
  return ev(`(() => { const o = document.getElementById('out');
    const v = o.querySelector('.verdict'); const err = o.querySelector('.err,.guard h3');
    const qs = [...o.querySelectorAll('.q')].map(q => ({ title: q.querySelector('.qh span:last-child')?.textContent,
      win: q.querySelector('.opt.win .name span:last-child')?.textContent, p: q.querySelector('.opt.win .pct')?.textContent }));
    return { verdict: v ? v.firstChild.textContent : null, p: v?.querySelector('.p')?.textContent, err: err?.textContent || null,
      calls: o.querySelector('.guard .calls b')?.textContent, meta: o.querySelector('.meta')?.textContent, qs }; })()`);
}
async function ask(btn, label) {
  if (DRY) { log.push({ label, dry: true }); return null; }
  await click(btn); calls++;
  const a = await answer(); log.push({ label, ...a }); console.error(label, JSON.stringify(a)); return a;
}

// ---------- script ----------
await send("Page.startScreencast", { format: "jpeg", quality: 90, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 });
recording = true; const tStart = Date.now();

// 1. title card, then badge
await O(`card("Typed decisions on Gemma 4 — live on local hardware","typevet · Gemma 4 31B · NVIDIA RTX 4090 · no cloud")`);
await sleep(4500); await O("cardOff()"); await sleep(900);
await step(1); await ring(".local", 8); await cap("Runs on our own GPU. No cloud, no vendor API.");
await sleep(3800); await O("ringOff()");

// 2. R01 supported
await step(2); await cap("Ask: does this receipt support an 80,500 claim?");
await ring('.thumb[data-id="R01"]', 6); await sleep(2800); await click('.thumb[data-id="R01"]'); await sleep(700);
await ring("#pTrue", 8); await sleep(1200); await click("#pTrue"); await sleep(700);
await ring("#askImg", 8); await sleep(1200);
await O("ringOff()"); await cap("Gemma 4 is reading the receipt image — live, on this machine…");
let a = await ask("#askImg", "R01 true total");
await ring("#out .verdict", 12);
await cap(a && a.verdict !== "supported" ? `The model answered “${a.verdict}” (${a.p}) for R01.` : "Supported — the model read the receipt image.");
await sleep(4200); await O("ringOff()");

// 3. swapped receipt
await step(3); await cap("Same claim, different receipt…");
await ring("#pSwap", 8); await sleep(2800); await click("#pSwap"); await sleep(700);
await ring("#askImg", 8); await sleep(1200); await O("ringOff()");
a = await ask("#askImg", "R02 swapped");
await ring("#out .verdict", 12);
await cap(a && a.verdict !== "contradicted" ? `The model answered “${a.verdict}” (${a.p}) for the swapped receipt.` : "Contradicted — the answer follows the picture, not the text.");
await sleep(4200);
await ring("#hist", 10); await cap("Recent answers: every result stays on screen, each one a live call.");
await sleep(3500); await O("ringOff()");

// 4. upload unreadable receipt
await step(4); await cap("An unreadable receipt…");
await ring("#upTile", 6); await sleep(2800);
await O(`click("#upTile .plus")`).catch(() => {});
{ const doc = await send("DOM.getDocument", { depth: 1 });
  const n = await send("DOM.querySelector", { nodeId: doc.root.nodeId, selector: "#file" });
  await send("DOM.setFileInputFiles", { nodeId: n.nodeId, files: [UPLOAD] }); }
for (let i = 0; i < 50; i++) { if (await ev(`!!document.querySelector('#upTile.on')`)) break; await sleep(100); }
await sleep(1000);
await ring("#askImg", 8); await sleep(1200); await O("ringOff()");
a = await ask("#askImg", "upload blurred");
await ring("#out .verdict", 12);
await cap(a && a.verdict !== "insufficient_evidence" ? `The model answered “${a.verdict}” (${a.p}) for the blurred receipt.` : "Insufficient evidence — it says when it can't tell.");
await sleep(4200); await O("ringOff()");

// 5. behind the scenes
await step(5);
if (!DRY) {
  await ring("#out details.bts summary", 6);
  await cap("Behind the scenes: the exact prompt, and the answer start typevet writes itself.");
  await sleep(2800); await click("#out details.bts summary"); await sleep(600); await O("ringOff()");
  await O(`scrollToEl("#out pre.prompt", 140, 2200)`); await sleep(500);
  await ring("#out pre.prompt .pf", 8); await sleep(2600);
  await O(`scrollToEl("#out .pfline", 200, 1500)`); await ring("#out .pfline", 8); await sleep(2200);
  await cap("Every allowed answer gets a probability from the model's own scores. No free text to parse.");
  await O(`scrollToEl("#out .bts table", 160, 2200)`); await ring("#out .bts .scroll", 8); await sleep(4500);
  await O("ringOff()"); await O(`scrollTo(0, 1500)`);
  await ring("#out details.bts summary", 6); await sleep(900); await click("#out details.bts summary"); await sleep(600); await O("ringOff()");
}

// 6. text
await step(6); await cap("One customer message, three typed questions at once: yes/no, pick one, and a score.");
await ring("#tTxt", 8); await sleep(2800); await click("#tTxt"); await sleep(700);
await ring("#msg", 8); await sleep(1800); await ring("#askTxt", 8); await sleep(1200); await O("ringOff()");
a = await ask("#askTxt", "text three questions");
await ring("#out .txtres", 10);
if (a) await cap(`Answers: ${a.qs.map((q) => `${q.win} ${q.p}`).join(" · ")}`);
await sleep(3000);
{ const h = await ev(`document.documentElement.scrollHeight - innerHeight`); if (h > 20) { await O(`scrollTo(${h}, 2000)`); await sleep(2200); await O(`scrollTo(0, 1500)`); } }
await sleep(1500); await O("ringOff()");

// 7. GIF guarantee
await step(7); await ring("#tImg", 8); await cap("Typed guarantee: send a GIF instead of a receipt image.");
await sleep(1800); await click("#tImg"); await sleep(700);
await ring("#guar", 8); await cap("Bad input is rejected by the type system — before the model is ever called.");
await sleep(2800); await click("#guar"); await sleep(300);
a = await answer(); log.push({ label: "gif guard", ...a }); console.error("gif", JSON.stringify(a));
await O("ringOff()"); await ring("#out .guard", 10); await sleep(4500); await O("ringOff()");

// 8. end card
await O("capOff()"); await O("stepOff()"); await sleep(400);
await O(`card("Typed answers · confidence on every answer · images and text · local","Gemma 4 on our own GPU, driven by typevet")`);
await sleep(4800);

recording = false; const tEnd = Date.now();
await send("Page.stopScreencast").catch(() => {});
fs.writeFileSync(path.join(REC, DRY ? "log-dry.json" : "log.json"), JSON.stringify({ calls, seconds: (tEnd - tStart) / 1000, log }, null, 2));
// concat list with per-frame durations
let lines = ["ffconcat version 1.0"];
const lastT = frames.length ? frames[frames.length - 1].t + 0.5 : 0;
frames.forEach((fr, i) => { const d = (i + 1 < frames.length ? frames[i + 1].t : lastT) - fr.t;
  lines.push(`file '${fr.f}'`, `duration ${Math.max(0.001, d).toFixed(4)}`); });
lines.push(`file '${frames[frames.length - 1].f}'`);
fs.writeFileSync(path.join(REC, DRY ? "frames-dry.ffconcat" : "frames.ffconcat"), lines.join("\n") + "\n");
console.error(`frames=${frames.length} span=${(lastT - frames[0].t).toFixed(1)}s calls=${calls}`);
await send("Browser.close").catch(() => {});
ws.close(); await sleep(500); try { chrome.kill(); } catch {}
fs.rmSync(profile, { recursive: true, force: true });
process.exit(0);
