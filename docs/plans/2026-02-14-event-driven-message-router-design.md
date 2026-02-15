# Event-Driven Message Router Design

## Goal

Replace `session-manager.ts` (~400 lines) and `background-processor.ts` (~224 lines) with a single `message-router.ts` (~50-80 lines) that processes messages through a simple FIFO queue with no timers, no polling, no per-session state, and no batching.

## Architecture

A single queue drains sequentially. HTTP handlers enqueue messages; a drain function processes them one at a time through the local ONNX model. Session context (project, userPrompt, promptNumber) is read from the database, not held in memory.

This works because:
- The local ONNX model is CPU-bound — parallel inference provides no benefit
- The plugin runs within a single Claude Code session — no multi-session routing needed
- Hooks are fire-and-forget — the queue buffers faster-than-processing arrivals

## What Gets Deleted

| File | Lines | Reason |
|------|-------|--------|
| `src/worker/session-manager.ts` | ~400 | Replaced entirely by message-router |
| `src/worker/background-processor.ts` | ~224 | Replaced entirely by message-router |

## What Gets Created

### `src/worker/message-router.ts` (~50-80 lines)

```typescript
interface RouterDeps {
  readonly db: Database;
  readonly modelManager: ModelManager;
}

interface RouterMessage {
  readonly type: "observation" | "summarize" | "complete";
  readonly claudeSessionId: string;
  readonly data: ObservationData | SummarizeData | CompleteData;
}

interface MessageRouter {
  readonly enqueue: (msg: RouterMessage) => void;
  readonly shutdown: () => Promise<void>;
  readonly pending: () => number;
}
```

Core implementation:

```typescript
const createMessageRouter = (deps: RouterDeps): MessageRouter => {
  const queue: RouterMessage[] = [];
  let drainPromise: Promise<void> | null = null;

  const drain = async () => {
    while (queue.length > 0) {
      await processMessage(queue.shift()!, deps);
    }
    drainPromise = null;
  };

  return {
    enqueue: (msg) => {
      queue.push(msg);
      if (!drainPromise) drainPromise = drain();
    },
    shutdown: () => drainPromise ?? Promise.resolve(),
    pending: () => queue.length,
  };
};
```

`processMessage` dispatches on type:
- `observation` → read session from DB, run local model, store result + embedding
- `summarize` → read session from DB, run local model, store summary
- `complete` → update session status in DB

## What Gets Modified

### `src/worker/handlers.ts`

- Replace `SessionManager` dependency with `MessageRouter` in `WorkerDeps`
- `handleQueueObservation`: enqueue `{ type: "observation", ... }` instead of `sessionManager.queueObservation()`
- `handleQueueSummary`: enqueue `{ type: "summarize", ... }` instead of `sessionManager.queueSummarize()`
- `handleCompleteSession`: enqueue `{ type: "complete", ... }` instead of `sessionManager.closeSession()`
- `handleQueuePrompt`: remove SessionManager calls (DB operations are sufficient)
- `handleHealth`: replace `activeSessions` with `router.pending()`

### `src/worker/local-agent.ts`

Simplify from async generator pattern to two standalone pure functions:

```typescript
const processObservation = async (
  deps: RouterDeps,
  context: SessionContext,
  observation: ToolObservation,
): Promise<Result<number, Error>>

const processSummary = async (
  deps: RouterDeps,
  context: SessionContext,
  input: SummarizeInput,
): Promise<Result<number, Error>>
```

Where `SessionContext` is:

```typescript
interface SessionContext {
  readonly claudeSessionId: string;
  readonly project: string;
  readonly promptNumber: number;
}
```

No `ActiveSession`, no `AbortController`, no async generator, no `SDKAgent` interface.

### `src/worker/agent-types.ts`

Delete entirely. The `SDKAgent` interface, `PendingInputMessage`, and `SDKAgentMessage` types are no longer needed. Message types move into `message-router.ts`.

### `src/worker/main.ts`

Replace:
```typescript
const sessionManager = createSessionManager();
const sdkAgent = createLocalAgent({ db, modelManager });
const backgroundProcessor = createBackgroundProcessor({ sessionManager, sdkAgent });
backgroundProcessor.start();
sessionManager.startEvictionSweep();
```

With:
```typescript
const router = createMessageRouter({ db, modelManager });
```

Shutdown simplifies from "stop processor, await completion, close all sessions" to:
```typescript
await router.shutdown();
```

### `src/worker/service.ts`

Update `WorkerDeps` import and threading.

## Data Flow (Before vs After)

**Before:**
```
HTTP → handler → SessionManager.queue*() → Map<SessionState>
                                                    ↓
                         BackgroundProcessor polls (1s interval)
                                                    ↓
                              SDKAgent.processMessages (async generator)
                                                    ↓
                                              DB store
```

**After:**
```
HTTP → handler → router.enqueue() → queue[]
                                        ↓
                                  drain() (immediate)
                                        ↓
                                processMessage() → DB read context → local model → DB store
```

## What We Eliminated

| Concept | Lines | Why not needed |
|---------|-------|----------------|
| Session Map | ~80 | Single session, context from DB |
| Async iterator + promise resolvers | ~40 | Array + drain loop |
| Observation batching + merge | ~60 | No API cost savings for local model |
| TTL eviction + sweep timer | ~50 | No sessions to evict |
| AbortController | ~15 | No cancellation needed |
| Polling interval | ~20 | Drain triggered at enqueue time |
| SDKAgent interface + generator | ~40 | Direct function calls |
| Message transformation layer | ~30 | Messages go straight to processor |

## Error Handling

- If `processMessage` throws, log the error and continue draining. One failed message must not block the queue.
- Use the existing `Result` pattern for DB operations within `processMessage`.
- Model inference errors are caught and logged (same as current local-agent behavior).

## Testing Strategy

- Unit test `createMessageRouter`: enqueue/drain/shutdown behavior
- Unit test `processMessage`: dispatch to observation/summary/complete handlers
- Existing `local-agent.test.ts` tests adapt to test the new standalone functions
- Existing `worker-handlers.test.ts` tests update mock from SessionManager to MessageRouter
