import { describe, expect, it, mock } from "bun:test";
import {
  createMessageRouter,
  type RouterMessage,
} from "../../src/worker/message-router";

const makeMsg = (id: string): RouterMessage => ({
  type: "observation",
  claudeSessionId: id,
  data: { toolName: "Edit", toolInput: {}, toolResponse: "", cwd: "/tmp" },
});

describe("message-router", () => {
  it("processes enqueued messages in FIFO order", async () => {
    const processed: string[] = [];
    const processor = mock(async (msg: RouterMessage) => {
      processed.push(msg.claudeSessionId);
    });

    const router = createMessageRouter({ processMessage: processor });

    router.enqueue(makeMsg("a"));
    router.enqueue(makeMsg("b"));

    await router.shutdown();

    expect(processed).toEqual(["a", "b"]);
    expect(processor).toHaveBeenCalledTimes(2);
  });

  it("processes messages sequentially (not in parallel)", async () => {
    const timeline: string[] = [];
    const processor = mock(async (msg: RouterMessage) => {
      timeline.push(`start-${msg.claudeSessionId}`);
      await new Promise((r) => setTimeout(r, 10));
      timeline.push(`end-${msg.claudeSessionId}`);
    });

    const router = createMessageRouter({ processMessage: processor });

    router.enqueue(makeMsg("first"));
    router.enqueue(makeMsg("second"));

    await router.shutdown();

    expect(timeline).toEqual([
      "start-first",
      "end-first",
      "start-second",
      "end-second",
    ]);
  });

  it("shutdown resolves immediately when queue is empty", async () => {
    const processor = mock(async () => {});
    const router = createMessageRouter({ processMessage: processor });

    await router.shutdown();

    expect(processor).not.toHaveBeenCalled();
  });

  it("reports pending count", async () => {
    let resolveFirst!: () => void;
    const blockingPromise = new Promise<void>((r) => {
      resolveFirst = r;
    });
    const processor = mock(async () => {
      await blockingPromise;
    });

    const router = createMessageRouter({ processMessage: processor });

    router.enqueue(makeMsg("a"));
    router.enqueue(makeMsg("b"));

    // First message is being processed, second is pending
    expect(router.pending()).toBe(1);

    resolveFirst();
    await router.shutdown();

    expect(router.pending()).toBe(0);
  });

  it("continues processing after a message handler throws", async () => {
    const processed: string[] = [];
    let callCount = 0;
    const processor = mock(async (msg: RouterMessage) => {
      callCount++;
      if (callCount === 1) throw new Error("boom");
      processed.push(msg.claudeSessionId);
    });

    const router = createMessageRouter({ processMessage: processor });

    router.enqueue(makeMsg("fails"));
    router.enqueue(makeMsg("succeeds"));

    await router.shutdown();

    expect(processed).toEqual(["succeeds"]);
  });

  it("handles messages enqueued during processing", async () => {
    const processed: string[] = [];
    let enqueueMore: (() => void) | null = null;

    const processor = mock(async (msg: RouterMessage) => {
      processed.push(msg.claudeSessionId);
      if (enqueueMore) {
        const fn = enqueueMore;
        enqueueMore = null;
        fn();
      }
    });

    const router = createMessageRouter({ processMessage: processor });

    enqueueMore = () => {
      router.enqueue(makeMsg("dynamic"));
    };

    router.enqueue(makeMsg("initial"));

    await router.shutdown();

    expect(processed).toEqual(["initial", "dynamic"]);
  });
});
