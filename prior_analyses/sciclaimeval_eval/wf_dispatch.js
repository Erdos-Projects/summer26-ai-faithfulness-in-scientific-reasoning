export const meta = {
  name: 'sciclaimeval-eval-dispatch',
  description: 'Self-fetching SciClaimEval Supported/Refuted prediction run: stage 1 refreshes the resume-aware manifest (eval prepare) and returns the next N missing (item x trial) cells; stage 2 fans them out to Haiku subagents. Verdicts are scraped into predictions_sciclaimeval.db by collect_cli afterward.',
  phases: [
    { title: 'Fetch', detail: 'eval prepare (resume-aware) + return next N missing tags' },
    { title: 'Predict', detail: 'one Haiku subagent per (item x trial) cell' },
  ],
}

const REPO = '/home/awndre/projects/Erdos/summer26-ai-science-reasoning'

// args: { run?: number (default 1), trials?: number (default 1), batch?: number (default 300), model?: 'haiku' }
let A = args
if (typeof A === 'string') { try { A = JSON.parse(A) } catch (e) { A = {} } }
const run = (A && A.run) || 1
const trials = (A && A.trials) || 1
const batch = (A && A.batch) || 300
const model = (A && A.model) || 'haiku'

// --- Stage 1: refresh the resume-aware manifest and fetch the next `batch` missing tags ---
phase('Fetch')
const py = `${REPO}/.venv/bin/python`
const runPad = String(run).padStart(2, '0')
const manifest = `sciclaimeval_eval/prompts/run_${runPad}/run_${runPad}_manifest.json`
const fetchCmd =
  `cd ${REPO} && ${py} -m sciclaimeval_eval.prepare --run ${run} --trials ${trials} >/dev/null 2>&1 && ` +
  `${py} -c "import json,os; ` +
  `m=json.load(open('${manifest}')); tags=list(m.keys())[:${batch}]; ` +
  `print(json.dumps({'promptDir':os.path.abspath('sciclaimeval_eval/prompts/run_${runPad}'),'tags':tags}))"`

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
  log('fetch returned no cells — run may be complete (0 MISSING) or fetch failed')
  return { dispatched: 0 }
}
log(`fetched ${tags.length} missing prediction cells; dispatching on ${model}`)

// --- Stage 2: one Haiku subagent per (item x trial) cell ---
phase('Predict')
await parallel(tags.map((tag) => () => {
  const promptFile = `${promptDir}/${tag}.txt`
  return agent(
    `Read ${promptFile} and follow it EXACTLY. Use ONLY the Read tool: (1) read that file, ` +
    `(2) view the figure image it references with the Read tool, (3) reason step by step, ` +
    `(4) end with your final line EXACTLY "Answer: yes" or "Answer: no" (lowercase). STOP after that line.`,
    { label: tag, phase: 'Predict', model }
  )
}))

log(`dispatched ${tags.length} cells; run collect_cli to scrape verdicts + tokens into the DB`)
return { dispatched: tags.length }
