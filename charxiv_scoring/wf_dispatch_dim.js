export const meta = {
  name: 'charxiv-rubric-dispatch-dim',
  description: 'Dispatch ONE DeLeAn dimension of the CharXiv pass-1 demand annotation. Stage 1 reads the pre-built pass_01 manifest (read-only — no prepare call, so parallel per-dim workflows never race) and returns this dim\'s missing tags; stage 2 fans them out to isolated Sonnet vision subagents. Scores are scraped into annotations_charxiv.db by collect_cli afterward.',
  phases: [
    { title: 'Fetch', detail: 'read manifest, filter to this dim, return its tags' },
    { title: 'Score', detail: 'one isolated Sonnet subagent per (item x dim) cell' },
  ],
}

const REPO = '/home/awndre/projects/Erdos/summer26-ai-science-reasoning'
const MANIFEST = 'charxiv_scoring/prompts/pass_01/pass_01_manifest.json'

const DIM_NAMES = {
  AS: 'Attention and Scan', CEc: 'Verbal Comprehension', CEe: 'Verbal Expression',
  CL: 'Conceptualisation, Learning, and Abstraction', MCt: 'Critical Thinking Processes',
  MCu: 'Calibrating Knowns and Unknowns', MCr: 'Identifying Relevant Information',
  MS: 'Mind Modelling and Social Cognition', QLq: 'Quantitative Reasoning',
  QLl: 'Logical Reasoning', SNs: 'Spatio-physical Reasoning', KNn: 'Natural Sciences',
  KNs: 'Social Sciences and Humanities', KNf: 'Formal Sciences',
  KNa: 'Applied Sciences and Professions', KNc: 'Customary Everyday Knowledge',
  VO: 'Volume', AT: 'Atypicality',
  VL: 'Visual Localization and Grounding', GS: 'Gestalt and Shape Judgment',
  MA: 'Multi-Element Visual Aggregation',
}

// args: { dim: 'AS', model?: 'sonnet', batch?: 999 }
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = {} } }
const dim = A && A.dim
const model = (A && A.model) || 'sonnet'
// 1 fetch agent + score agents must stay <= the 1000 lifetime-agent cap, so cap the batch at 999.
const batch = Math.min((A && A.batch) || 999, 999)
if (!dim) { log('no dim provided in args'); return { dispatched: 0 } }
const dimName = DIM_NAMES[dim] || dim

// --- Stage 1: read the pre-built manifest (read-only) and select this dim's missing tags ---
phase('Fetch')
const py = `${REPO}/.venv/bin/python`
const fetchCmd =
  `cd ${REPO} && ${py} -c "import json,os; ` +
  `m=json.load(open('${MANIFEST}')); ` +
  `tags=[t for t in m if t.endswith('__${dim}')][:${batch}]; ` +
  `print(json.dumps({'promptDir':os.path.abspath('charxiv_scoring/prompts/pass_01'),'tags':tags}))"`

const shard = await agent(
  `Run this EXACT command with the Bash tool, then return ONLY the JSON object it prints ` +
  `(a "promptDir" string and a "tags" array of strings):\n\n${fetchCmd}\n`,
  {
    label: `fetch:${dim}`, phase: 'Fetch',
    schema: {
      type: 'object', additionalProperties: false,
      properties: { promptDir: { type: 'string' }, tags: { type: 'array', items: { type: 'string' } } },
      required: ['promptDir', 'tags'],
    },
  }
)

const tags = (shard && shard.tags) || []
const promptDir = shard && shard.promptDir
if (!promptDir || !tags.length) {
  log(`dim ${dim}: fetch returned no cells — already complete or fetch failed`)
  return { dim, dispatched: 0 }
}
log(`dim ${dim}: dispatching ${tags.length} cells on ${model}`)

// --- Stage 2: one isolated Sonnet vision subagent per cell ---
phase('Score')
await parallel(tags.map((tag) => () => {
  const promptFile = `${promptDir}/${tag}.txt`
  return agent(
    `Read ${promptFile} and follow it EXACTLY. (1) Read that file. (2) Use the Read tool to ` +
    `view the figure image it references. (3) Reason briefly. (4) Conclude with the EXACT line ` +
    `"Thus, the level of *${dimName}* demanded by the given TASK INSTANCE is: N" ` +
    `(N = an integer 0-5). (5) STOP. Use ONLY the Read tool (twice). Do NOT answer the question.`,
    { label: tag, phase: 'Score', model }
  )
}))

log(`dim ${dim}: dispatched ${tags.length} cells; run collect_cli to scrape scores into the DB`)
return { dim, dispatched: tags.length }
