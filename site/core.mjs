
const n=(value,fallback=0)=>Number.isFinite(Number(value))?Number(value):fallback;
const clamp=(value,low,high)=>Math.min(high,Math.max(low,value));
const round=(value,digits=2)=>Number(value.toFixed(digits));
const mean=values=>values.length?values.reduce((sum,value)=>sum+value,0)/values.length:0;
const parseJSON=(value,fallback=[])=>{try{return JSON.parse(value)}catch{return fallback}};
const valuesFrom=value=>String(value).split(/[\s,]+/).map(Number).filter(Number.isFinite);
const result=(status,summary,metrics,rows,detail='')=>({status,summary,metrics,rows,detail});
const erf=x=>{const sign=x<0?-1:1,a=Math.abs(x),t=1/(1+0.3275911*a);const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-a*a);return sign*y};
const normalCdf=z=>0.5*(1+erf(z/Math.sqrt(2)));
const wilson=(successes,total)=>{if(!total)return[0,0];const z=1.96,p=successes/total,d=1+z*z/total,c=(p+z*z/(2*total))/d,h=z*Math.sqrt((p*(1-p)+z*z/(4*total))/total)/d;return[clamp(c-h,0,1),clamp(c+h,0,1)]};
const sha256=async value=>{const bytes=new TextEncoder().encode(String(value));const digest=await crypto.subtle.digest('SHA-256',bytes);return[...new Uint8Array(digest)].map(byte=>byte.toString(16).padStart(2,'0')).join('')};
const tag=(xml,name)=>xml.match(new RegExp('<'+name+'[^>]*>([\\s\\S]*?)<\\/'+name+'>','i'))?.[1]?.trim()??'';
const similarity=(a,b)=>{const x=String(a).toLowerCase(),y=String(b).toLowerCase();if(x===y)return 1;const A=new Set(x.split(/\W+/).filter(Boolean)),B=new Set(y.split(/\W+/).filter(Boolean));const inter=[...A].filter(v=>B.has(v)).length;return inter/Math.max(1,new Set([...A,...B]).size)};

export const meta={"slug":"injectbench","name":"InjectBench","eyebrow":"Prompt-injection evaluation","description":"Evaluate an attack corpus against selected guardrails and show attack success with intervals and latency.","fields":[{"name":"trials","label":"Trials per attack family","type":"number","min":10,"max":10000,"step":10,"help":""},{"name":"instructionFilter","label":"Instruction hierarchy filter","type":"checkbox","help":""},{"name":"toolPolicy","label":"Tool allow-list","type":"checkbox","help":""},{"name":"outputScanner","label":"Output exfiltration scanner","type":"checkbox","help":""},{"name":"encodingDetector","label":"Encoding detector","type":"checkbox","help":""}]};
export const initialState={"trials":400,"instructionFilter":true,"toolPolicy":true,"outputScanner":true,"encodingDetector":false};
export const alternateState={"trials":400,"instructionFilter":false,"toolPolicy":false,"outputScanner":false,"encodingDetector":false};
export async function compute(i){const trials=Math.round(n(i.trials)),defs=[i.instructionFilter&&['Instruction filter',.48,18],i.toolPolicy&&['Tool policy',.36,12],i.outputScanner&&['Output scanner',.29,24],i.encodingDetector&&['Encoding detector',.22,31]].filter(Boolean),base=.62,remaining=defs.reduce((p,d)=>p*(1-d[1]),base),success=Math.round(trials*4*remaining),total=trials*4,[lo,hi]=wilson(success,total),latency=defs.reduce((s,d)=>s+d[2],0),rows=[['Instruction override',base],['Tool misuse',base*.82],['Data exfiltration',base*.74],['Encoded payload',base*.66]].map(([category,rate])=>({category,attackSuccess:`${round(rate*defs.reduce((p,d)=>p*(1-d[1]),1)*100)}%`,trials}));return result('Evaluation complete',`${defs.length} defences reduce estimated attack success from ${round(base*100)}% to ${round(remaining*100)}%.`,[{label:'Attack success rate',value:`${round(remaining*100)}%`},{label:'95% interval',value:`${round(lo*100)}–${round(hi*100)}%`},{label:'Added latency',value:`${latency} ms`},{label:'Defences enabled',value:defs.length}],rows,'Toggle guardrails to see which reduction justifies its latency cost.')}
