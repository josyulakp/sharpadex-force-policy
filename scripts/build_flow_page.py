"""Page for the flow-policy diagnosis.

    python scripts/build_flow_page.py     -> reports/flow_diagnosis/index.html   (needs scripts/flow_diagnose.py first)

Adds the direct tests (runs/flow_diagnose/*/seed0) when their evaluation exists.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sharpa.config import REPO_ROOT  # noqa: E402

OUT = REPO_ROOT / "reports" / "flow_diagnosis"
TESTS = {"flow (reference)": "runs/flow/full/seed0", "regression (reference)": "runs/flow/regression_25k/seed0",
         "flow, width 512": "runs/flow_diagnose/wide/seed0", "flow, force channels rescaled": "runs/flow_diagnose/force_scale/seed0",
         "flow, 50,000 steps": "runs/flow_diagnose/steps_50k/seed0"}


def main():
    d = json.loads((OUT / "diagnosis.json").read_text())
    tests = []
    for name, run in TESTS.items():
        p = REPO_ROOT / run / "eval_heldout.json"
        if p.exists():
            e = json.loads(p.read_text())
            z = REPO_ROOT / run / "eval_heldout__flow_noise_scale-0.json"
            tests.append({"name": name, "act": 1000 * e["act"]["all"], "force": e["force"]["all"], "persist": e["force_persist"]["all"],
                          "act0": 1000 * json.loads(z.read_text())["act"]["all"] if z.exists() else None})
        else:
            tests.append({"name": name, "act": None})
    html = TEMPLATE.replace("__DATA__", json.dumps(d)).replace("__TESTS__", json.dumps(tests))
    (OUT / "index.html").write_text(html)
    print("wrote", OUT / "index.html", f"({len(html) / 1e3:.0f} kB)")


TEMPLATE = r"""<title>Flow Policy Diagnosis</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap">
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--line:#e3e2dd;--track:#eeede8;--flow:#2a78d6;--reg:#eb6834;--train:#1baf7a;--bad:#d03b3b;--good:#0ca30c}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--line:#34342f;--track:#2a2a27;--flow:#3987e5;--reg:#d95926;--train:#199e70;--bad:#e66767;--good:#3fbf3f;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0d0d0d;--card:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--line:#34342f;--track:#2a2a27;--flow:#3987e5;--reg:#d95926;--train:#199e70;--bad:#e66767;--good:#3fbf3f;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.55 Inter,system-ui,sans-serif;padding-block:28px 60px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:920px;margin:0 auto;display:flex;flex-direction:column;gap:26px}
h1,h2{font-family:"Space Grotesk",Inter,sans-serif;text-wrap:balance;margin:0}
h1{font-size:30px;line-height:1.15}h2{font-size:20px}
p{margin:0;max-width:68ch}.sub{color:var(--fg2)}
section{display:flex;flex-direction:column;gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;display:flex;flex-direction:column;gap:10px;min-width:0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.tile b{font:600 28px/1.1 "Space Grotesk",sans-serif;display:block}
.tile span{color:var(--fg2);font-size:13px}
.lab{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--fg2)}
.bars{display:grid;grid-template-columns:minmax(120px,max-content) 1fr max-content;gap:6px 10px;align-items:center;font-size:13px}
.bars .t{background:var(--track);border-radius:4px;height:14px;position:relative;min-width:40px}
.bars .t i{position:absolute;left:0;top:0;bottom:0;border-radius:0 4px 4px 0}
.bars .v{font-variant-numeric:tabular-nums}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--fg2)}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;font-size:13px;width:100%}
th,td{text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--fg2);font-weight:500}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.btns{display:flex;gap:6px;flex-wrap:wrap}
button{font:inherit;font-size:13px;padding:4px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button[aria-pressed="true"]{background:var(--fg);color:var(--bg);border-color:var(--fg)}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
ol,ul{margin:0;padding-left:20px;max-width:68ch}li{margin:4px 0}
code{font-size:12.5px}
</style>
<main>
<header style="display:flex;flex-direction:column;gap:8px">
  <div class="lab">SharpaDex · three tasks · seed 0 · <span id="nframes"></span> held-out frames</div>
  <h1>Why the flow policy trails the regression policy</h1>
  <p class="sub">Both models have the same inputs, size, data and 25,000 training steps. The flow model's action error is about 3.8 mrad higher. Most of that is the price of drawing one random sample; the rest is a weaker fit, and its force output is broken by a scaling choice. Direct tests at the bottom: a wider flow model closes most of the gap, twice the steps closes part of it, and rescaling the force channels fixes the force output.</p>
</header>

<section>
  <h2>How close are the predictions to the real data?</h2>
  <p class="sub">"Explained" compares the model's error with doing nothing: 100% is a perfect prediction, 0% is no better than holding the current joints (or assuming the force stays as it is), and a negative value is worse than that.</p>
  <div class="tiles" id="tiles"></div>
  <div class="card scroll"><table id="align"></table></div>
</section>

<section>
  <h2>Where the action gap comes from</h2>
  <p class="sub">Action error on held-out frames, mean over the 16 predicted steps and 58 joints (mrad).</p>
  <div class="card"><div class="legend"><span><i style="background:var(--flow)"></i>flow</span><span><i style="background:var(--reg)"></i>regression</span></div><div class="bars" id="ladder"></div></div>
  <ul>
    <li><b>Sampling, about 3 of the 3.8 mrad.</b> A flow model returns one random draw from the actions it considers plausible; draws differ from each other by <span id="spread"></span> mrad. Averaging 16 draws, or starting from zero noise, removes most of the gap.</li>
    <li><b>More integration steps make it worse, not better.</b> So the 10-step sampler is not the problem; finer integration only reproduces the spread more faithfully.</li>
    <li><b>A weaker fit, about 1 mrad.</b> Even without sampling noise the flow model is behind, on training frames too (see below).</li>
  </ul>
  <div class="cols">
    <div class="card"><div class="lab">Flow error against integration steps</div><div class="bars" id="steps"></div></div>
    <div class="card"><div class="lab">Flow error against starting-noise size</div><div class="bars" id="noise"></div></div>
  </div>
</section>

<section>
  <h2>Under-fitting or over-fitting?</h2>
  <p class="sub">The same error on frames the model was trained on. Both models are far better on training frames, so both are limited by the amount of data. The flow model also fits the training frames worse than regression does, so it has learned less in the same number of steps.</p>
  <div class="card"><div class="legend"><span><i style="background:var(--flow)"></i>flow</span><span><i style="background:var(--reg)"></i>regression</span><span><i style="background:var(--train)"></i>training frames</span></div><div class="bars" id="fit"></div></div>
  <div class="card"><div class="lab">Flow: chunk recovered in one step from each noise level (mrad)</div>
    <p class="sub">At noise level 1 the input is pure noise, so this is the flow network used as a plain regressor. It should match the regression model there, and it does not.</p>
    <div class="legend"><span><i style="background:var(--flow)"></i>held-out</span><span><i style="background:var(--train)"></i>training</span></div><div class="bars" id="denoise"></div></div>
</section>

<section>
  <h2>Action-layer activations: is the network too narrow?</h2>
  <p class="sub">For each layer on the path from the latent to the joint commands: how many independent directions carry 99% of the activation variance, as a share of the layer's width. A layer near 100% is full and would benefit from more width. No layer has unused units.</p>
  <div class="btns" id="taus"></div>
  <div class="card"><div class="legend"><span><i style="background:var(--flow)"></i>flow velocity network</span><span><i style="background:var(--reg)"></i>regression</span></div><div class="bars" id="layers"></div></div>
  <div class="card scroll"><table id="layertab"></table></div>
  <p class="sub">The flow network uses more directions than the regression network in every 256-wide layer (it must also represent the noisy chunk and the noise level), but none is full. Read alone, this suggests width is not the limit. The direct test below says otherwise: the 512-wide flow model is clearly better. So this activation measure did not detect the shortage, and a wider run is the reliable test.</p>
</section>

<section>
  <h2>How to diagnose it</h2>
  <div class="card scroll"><table>
    <tr><th>Question</th><th>Test</th><th>Result here</th></tr>
    <tr><td>Is it the sampler?</td><td>Vary integration steps; start from zero noise</td><td>No. More steps raise the error; zero noise lowers it</td></tr>
    <tr><td>Is it sampling spread?</td><td>Average K draws; measure spread between draws</td><td>Yes, the largest part</td></tr>
    <tr><td>Is it capacity or training time?</td><td>Error on training frames; one-step recovery at noise level 1; a wider model; a longer run</td><td>Yes, both. Width 512 and 50,000 steps each lower the error</td></tr>
    <tr><td>Is it too little data?</td><td>Gap between training and held-out error</td><td>Yes for both models (about 5–6 mrad)</td></tr>
    <tr><td>Is the width used up?</td><td>Unused units and directions in use per layer</td><td>No layer is full, yet the wider model is better: this check is not sufficient</td></tr>
    <tr><td>Is an output badly scaled?</td><td>Size of each target next to the unit-size noise it is mixed with</td><td>Yes for force: a typical force change is about 0.03 units, the noise is 1</td></tr>
  </table></div>
  <p class="sub">Why the force output fails: the flow mixes its targets with unit-size noise. Joint changes are scaled to about unit size, but fingertip forces are divided by 10 N, so a 0.3 N change is 0.03 units and the leftover noise swamps it. With zero starting noise the flow's force error was 0.29 N instead of 0.5 N.</p>
</section>

<section>
  <h2>Direct tests</h2>
  <p class="sub">One seed each, same data. Action error in mrad (as sampled / zero starting noise), force error in N next to "force stays as it is". Regression was not retrained at width 512 or 50,000 steps, so these runs show what limits the flow model, not which head is better at equal size.</p>
  <div class="card scroll"><table id="tests"></table></div>
</section>
</main>
<script>
const D=__DATA__, T=__TESTS__;
const $=id=>document.getElementById(id), f=(x,n=1)=>x.toFixed(n);
const H=m=>D[m]["alignment held-out"], TR=m=>D[m]["alignment training"];
$("nframes").textContent=D.frames.toLocaleString();
function bars(el,rows,max,unit){el.innerHTML=rows.map(r=>`<div>${r[0]}</div><div class="t" title="${r[0]}: ${f(r[1],r[3]??1)} ${unit}"><i style="width:${Math.max(0,100*r[1]/max)}%;background:var(--${r[2]})"></i></div><div class="v">${f(r[1],r[3]??1)}</div>`).join("")}
const pct=x=>(x>=0?"":"−")+Math.abs(Math.round(100*x))+"%";
$("tiles").innerHTML=[["Joint commands, flow",H("flow").joints.explained],["Joint commands, regression",H("regression").joints.explained],["Fingertip force, flow",H("flow").force.explained],["Fingertip force, regression",H("regression").force.explained]]
 .map(([n,v])=>`<div class="card tile"><b style="color:${v<0?"var(--bad)":"inherit"}">${pct(v)}</b><span>${n}: share of the real change explained</span></div>`).join("");
const rowsA=["joints","arm joints","hand joints","joints, in contact","joints, free","force","force, in contact","force, free"];
$("align").innerHTML=`<tr><th>Held-out frames</th><th>Model</th><th class="n">Error</th><th class="n">Doing nothing</th><th class="n">Explained</th><th class="n">Correlation</th><th class="n">R²</th></tr>`+
 rowsA.flatMap(k=>["flow","regression"].map(m=>{const a=H(m)[k],j=k.includes("joint"),u=j?1000:1;return `<tr><td>${m=="flow"?k[0].toUpperCase()+k.slice(1):""}</td><td>${m}</td><td class="n">${f(u*a.error,j?1:2)} ${j?"mrad":"N"}</td><td class="n">${f(u*a.baseline,j?1:2)}</td><td class="n" style="color:${a.explained<0?"var(--bad)":"inherit"}">${pct(a.explained)}</td><td class="n">${f(a.corr,2)}</td><td class="n">${f(a.r2,2)}</td></tr>`})).join("");
const fl=D.flow, s=n=>fl.steps.find(x=>x.steps==n).error, nz=x=>fl.noise.find(y=>y.scale==x);
const k16=fl.samples.find(x=>x.k==16);
$("spread").textContent=f(k16.spread,0);
bars($("ladder"),[["Flow, one draw (as evaluated)",1000*H("flow").joints.error,"flow"],["Flow, 2 integration steps",s(2),"flow"],["Flow, zero starting noise",nz(0).error,"flow"],["Flow, mean of 16 draws",k16.error,"flow"],["Regression",1000*H("regression").joints.error,"reg"]],22,"mrad");
bars($("steps"),fl.steps.map(x=>[x.steps+(x.steps==1?" step":" steps"),x.error,"flow"]),22,"mrad");
bars($("noise"),fl.noise.map(x=>["noise × "+x.scale,x.error,"flow"]),22,"mrad");
bars($("fit"),[["Flow, held-out",1000*H("flow").joints.error,"flow"],["Flow, training",1000*TR("flow").joints.error,"train"],["Flow zero noise, held-out",nz(0).error,"flow"],["Flow zero noise, training",nz(0).error_training,"train"],["Regression, held-out",1000*H("regression").joints.error,"reg"],["Regression, training",1000*TR("regression").joints.error,"train"]],22,"mrad");
bars($("denoise"),fl["denoise held-out"].flatMap((x,i)=>[["noise level "+x.tau,x.one_step_error_mrad,"flow"],["",fl["denoise training"][i].one_step_error_mrad,"train"]]),22,"mrad");
const conds=Object.keys(fl.layers); let cur=conds[0];
function layers(){const L=fl.layers[cur],R=D.regression.layers.forward;
 $("taus").innerHTML=conds.map(c=>`<button aria-pressed="${c==cur}" data-c="${c}">flow at noise level ${c.split(" ")[1]}</button>`).join("");
 $("taus").querySelectorAll("button").forEach(b=>b.onclick=()=>{cur=b.dataset.c;layers()});
 const names=Object.keys(L).filter(n=>!n.includes("feed-forward"));
 bars($("layers"),names.flatMap(n=>[[n,100*L[n].dims_99/L[n].width,"flow",0]].concat(R[n]?[["",100*R[n].dims_99/R[n].width,"reg",0]]:[])),100,"% of width");
 $("layertab").innerHTML=`<tr><th>Layer</th><th class="n">Width</th><th class="n">Flow: 90% / 99% dirs</th><th class="n">Regr.: 90% / 99% dirs</th><th class="n">Flow size (RMS)</th><th class="n">Regr. size</th><th class="n">Unused units</th></tr>`+
  Object.keys(L).map(n=>`<tr><td>${n}</td><td class="n">${L[n].width}</td><td class="n">${L[n].dims_90} / ${L[n].dims_99}</td><td class="n">${R[n]?R[n].dims_90+" / "+R[n].dims_99:"–"}</td><td class="n">${f(L[n].rms,2)}</td><td class="n">${R[n]?f(R[n].rms,2):"–"}</td><td class="n">${Math.round(100*L[n].unused_share)}%</td></tr>`).join("")}
layers();
$("tests").innerHTML=`<tr><th>Run</th><th class="n">Action error</th><th class="n">Zero noise</th><th class="n">Force error</th><th class="n">Force stays</th></tr>`+
 T.map(t=>t.act==null?`<tr><td>${t.name}</td><td colspan="4" class="sub">not finished yet</td></tr>`:`<tr><td>${t.name}</td><td class="n">${f(t.act,2)}</td><td class="n">${t.act0==null?"–":f(t.act0,2)}</td><td class="n">${f(t.force,2)}</td><td class="n">${f(t.persist,2)}</td></tr>`).join("");
</script>
"""

if __name__ == "__main__":
    main()
