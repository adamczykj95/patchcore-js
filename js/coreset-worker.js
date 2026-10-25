import { greedyCpu, argmaxRange, updateRange } from './coreset.js';

// Sense-reversing barrier on ctrl = [arrived, generation].
function barrier(ctrl, parties) {
  const gen = Atomics.load(ctrl, 1);
  if (Atomics.add(ctrl, 0, 1) === parties - 1) {
    Atomics.store(ctrl, 0, 0);
    Atomics.add(ctrl, 1, 1);
    Atomics.notify(ctrl, 1);
    return;
  }
  for (let spin = 0; spin < 2000; spin++) if (Atomics.load(ctrl, 1) !== gen) return;
  while (Atomics.load(ctrl, 1) === gen) Atomics.wait(ctrl, 1, gen);
}

function runShared({ w, workers, n, d, count, feats, minDist, bestVal, bestIdx, ctrl, selected }) {
  const lo = Math.floor((w * n) / workers);
  const hi = Math.floor(((w + 1) * n) / workers);
  for (let it = 0; it < count; it++) {
    // double-buffered by parity so a fast worker can't overwrite values still being read
    const slot = (it & 1) * workers;
    const [v, i] = argmaxRange(minDist, lo, hi);
    bestVal[slot + w] = v;
    bestIdx[slot + w] = i;
    barrier(ctrl, workers);
    let gv = -1, gi = 0;
    for (let k = 0; k < workers; k++) {
      if (bestVal[slot + k] > gv) { gv = bestVal[slot + k]; gi = bestIdx[slot + k]; }
    }
    if (w === 0) {
      selected[it] = gi;
      if (it % 64 === 0 || it === count - 1) self.postMessage({ progress: it + 1 });
    }
    updateRange(feats, minDist, lo, hi, d, gi);
  }
  self.postMessage({ done: true });
}

self.onmessage = ({ data }) => {
  if (data.mode === 'shared') {
    runShared(data);
    return;
  }
  const { feats, n, d, count, minDist } = data;
  const selected = greedyCpu(feats, n, d, count, minDist, (done) => self.postMessage({ progress: done }));
  self.postMessage({ selected }, [selected.buffer]);
};
