export const meta = {
  name: 'sciver-verify-dispatch',
  description: 'Dispatch SciVer chart (item,trial) cells to verifier subagents (model from args.model, default haiku) for claim verification (yes/no). Each shard file is loaded by a Haiku agent, then verifier agents read figure + prompt and answer. Results recovered by sciver_eval.collect_cli via transcript tags.',
  phases: [
    { title: 'Load', detail: 'read shard cell-lists' },
    { title: 'Verify', detail: 'one verifier subagent per (item,trial): read figure, answer yes/no' },
  ],
}

const A = (typeof args === 'string') ? JSON.parse(args) : (args || {})
const SHARD_FILES = A.shardFiles || (A.shardFile ? [A.shardFile] : [])
const DIR = A.dir
const RUN = A.run || 1
// Verifier model alias ('haiku' | 'sonnet' | 'opus'); loaders stay 'haiku' (cheap JSON echo).
// Must match the model id recorded on the run row in predictions.db.
const VMODEL = A.model || 'haiku'
const t2 = (n) => String(n).padStart(2, '0')

phase('Load')
const loaded = await parallel(SHARD_FILES.map(sf => () => agent(
  `Read the JSON file ${sf}. It contains an object {"cells": [{"id": "...", "trial": N}, ...]}. ` +
  `Return the cells array exactly as found.`,
  {
    model: 'haiku',
    label: `load:${sf.split('/').pop()}`,
    schema: {
      type: 'object', additionalProperties: false, required: ['cells'],
      properties: {
        cells: {
          type: 'array',
          items: {
            type: 'object', additionalProperties: false, required: ['id', 'trial'],
            properties: { id: { type: 'string' }, trial: { type: 'integer' } },
          },
        },
      },
    },
  }
)))
const cells = loaded.filter(Boolean).flatMap(x => x.cells || [])
log(`${cells.length} cells to verify across ${SHARD_FILES.length} shard(s)`)

phase('Verify')
await parallel(cells.map(c => () => agent(
  `Read the file ${DIR}/se_r${t2(RUN)}__${c.id}__t${t2(c.trial)}.txt and follow it EXACTLY. ` +
  `(1) Read that file. (2) Use the Read tool to view EACH figure image path it lists — ` +
  `you must actually look at the chart. (3) Reason briefly through the task. ` +
  `(4) Conclude with a final line that is exactly "Answer: yes" or "Answer: no" (lowercase). ` +
  `(5) STOP. Use ONLY the Read tool.`,
  { model: VMODEL, phase: 'Verify', label: `${c.id}__t${t2(c.trial)}` }
)))

return { dispatched: cells.length, shards: SHARD_FILES.length }
