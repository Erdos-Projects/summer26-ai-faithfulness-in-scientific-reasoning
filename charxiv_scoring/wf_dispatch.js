export const meta = {
  name: 'charxiv-rubric-dispatch',
  description: 'Self-fetching CharXiv DeLeAn demand-rubric pass: stage 1 refreshes the resume-aware manifest and returns the next N missing cells; stage 2 fans them out to isolated Sonnet subagents. Scores are scraped into annotations_charxiv.db by collect_cli afterward.',
  phases: [
    { title: 'Fetch', detail: 'refresh manifest (resume-aware) + return next N missing tags' },
    { title: 'Score', detail: 'one isolated Sonnet subagent per (item x dim) cell' },
  ],
}

const REPO = '/home/awndre/projects/Erdos/summer26-ai-science-reasoning'
const DIMS = 'AS,QLq,QLl,MCr,AT,VO,VL,GS,MA,MCu,CL,KNf'

// Display names for every dim code (18 DeLeAn + 3 authored emergent). The agent's required
// conclusion line uses the rubric's display name, derived here from the dim_code.
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

// args: { batch?: number (default 300), model?: 'sonnet' }
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = {} } }
const batch = (A && A.batch) || 300
const model = (A && A.model) || 'sonnet'
const offset = (A && A.offset) || 0
// prepare:false reuses an already-written manifest (required when running several shards in
// parallel — each takes a disjoint [offset, offset+batch) slice; re-preparing would race on files).
const doPrepare = !(A && A.prepare === false)

// --- Stage 1: refresh the resume-aware manifest and fetch the next `batch` missing tags ---
phase('Fetch')
const py = `${REPO}/.venv/bin/python`
const prepareStep = doPrepare
  ? `${py} -m charxiv_scoring.prepare --pass 1 --dims ${DIMS} >/dev/null 2>&1 && `
  : ''
const fetchCmd =
  `cd ${REPO} && ${prepareStep}` +
  `${py} -c "import json,os; ` +
  `m=json.load(open('charxiv_scoring/prompts/pass_01/pass_01_manifest.json')); ` +
  `tags=list(m.keys())[${offset}:${offset}+${batch}]; ` +
  `print(json.dumps({'promptDir':os.path.abspath('charxiv_scoring/prompts/pass_01'),'tags':tags}))"`

const shard = await agent(
  `Run this EXACT command with the Bash tool, then return ONLY the JSON object it prints ` +
  `(a "promptDir" string and a "tags" array of strings):\n\n${fetchCmd}\n`,
  {
    label: 'fetch-shard', phase: 'Fetch',
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
  log('fetch returned no cells — pass may be complete (0 MISSING) or fetch failed')
  return { dispatched: 0 }
}
log(`fetched ${tags.length} missing cells; dispatching on ${model}`)

// --- Stage 2: one isolated subagent per cell ---
phase('Score')
await parallel(tags.map((tag) => () => {
  const dimCode = tag.split('__').pop()
  const dimName = DIM_NAMES[dimCode] || dimCode
  const promptFile = `${promptDir}/${tag}.txt`
  return agent(
    `Read ${promptFile} and follow it EXACTLY. (1) Read that file. (2) Use the Read tool to ` +
    `view the figure image it references. (3) Reason briefly. (4) Conclude with the EXACT line ` +
    `"Thus, the level of *${dimName}* demanded by the given TASK INSTANCE is: N" ` +
    `(N = an integer 0-5). (5) STOP. Use ONLY the Read tool (twice). Do NOT answer the question.`,
    { label: tag, phase: 'Score', model }
  )
}))

log(`dispatched ${tags.length} cells; run collect_cli to scrape scores + tokens into the DB`)
return { dispatched: tags.length }
