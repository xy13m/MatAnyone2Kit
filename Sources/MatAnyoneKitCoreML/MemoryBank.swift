import Accelerate
import Foundation

/// Single-object MatAnyone working memory + affinity/readout, in pure Swift/Accelerate.
///
/// Implements the upstream MatAnyone2 `SingleObjectMemory` / `MemoryOps` in Swift. Batch is
/// always 1 in the realtime path (no flip-aug), so everything is 2-D row-major `[Float]` and the
/// matmuls are `cblas_sgemm`. Keys are stored channel-first `[C, N]`, matching the PyTorch memory
/// boundary; values are stored token-major `[N, CV]` so the sparse readout reads one contiguous
/// row per selected token. `N = T * h * w` grows each memory frame and is FIFO-capped.
///
/// Validated end-to-end against the PyTorch reference by `scripts/dump_e2e_ref.py` + `e2e_validate.swift`.
final class MemoryBank {
    let maxMemFrames: Int
    let topK: Int
    let keyDim: Int      // CK
    let valueDim: Int    // CV

    private(set) var h = 0
    private(set) var w = 0

    // Row-major stores: key[CK,N] and shrinkage[N] channel-first, valueT[N,CV] token-major.
    private var key: [Float] = []
    private var shrinkage: [Float] = []
    private var valueT: [Float] = []
    private var n = 0
    private var permEnd = 0

    // Object memory running sum [Q, embedDim+1] and sensory [h, w, sensoryDim] (NHWC).
    private(set) var objV: [Float]?
    private(set) var objVShape: [Int]?

    init(maxMemFrames: Int = 4, topK: Int = 30, keyDim: Int = 64, valueDim: Int = 256) {
        self.maxMemFrames = maxMemFrames
        self.topK = topK
        self.keyDim = keyDim
        self.valueDim = valueDim
    }

    var hw: Int { h * w }
    var maxWorkTokens: Int { maxMemFrames * hw }
    var engaged: Bool { n > 0 }

    func clearTemp() {
        key = []; shrinkage = []; valueT = []
        simLHSValid = false
        n = 0; permEnd = 0
        objV = nil; objVShape = nil
    }

    // ----------------------------------------------------------------- write
    /// keyIn[CK,h,w], shrinkageIn[1,h,w], mskValue[CV,h,w] (channel-first, contiguous),
    /// objValue[Q, C+1]. Appends a memory frame and FIFO-caps the working tokens.
    func addMemory(key keyIn: [Float], shrinkage shrinkageIn: [Float], value mskValue: [Float],
                   objValue: [Float], objValueShape: [Int], h: Int, w: Int, asPermanent: Bool) {
        self.h = h; self.w = w
        let newN = h * w

        accumulateObj(objValue, objValueShape)
        appendTokens(into: &key, src: keyIn, rows: keyDim, addCols: newN)
        appendTokens(into: &shrinkage, src: shrinkageIn, rows: 1, addCols: newN)
        valueT.append(contentsOf: [Float](unsafeUninitializedCapacity: newN * valueDim) { t, count in
            vDSP_mtrans(mskValue, 1, t.baseAddress!, 1, vDSP_Length(newN), vDSP_Length(valueDim))
            count = newN * valueDim
        })
        n += newN
        simLHSValid = false
        if asPermanent && permEnd == 0 { permEnd = n }
        fifo()
    }

    private func accumulateObj(_ v: [Float], _ shape: [Int]) {
        if objV == nil {
            objV = v; objVShape = shape
        } else {
            vDSP.add(objV!, v, result: &objV!)
        }
    }

    /// Append `addCols` columns to a row-major `[rows, oldCols]` store → `[rows, oldCols+addCols]`.
    private func appendTokens(into store: inout [Float], src: [Float], rows: Int, addCols: Int) {
        let oldCols = rows == 0 ? 0 : (store.count / rows)
        let newCols = oldCols + addCols
        var out = [Float](repeating: 0, count: rows * newCols)
        for r in 0..<rows {
            for c in 0..<oldCols { out[r * newCols + c] = store[r * oldCols + c] }
            for c in 0..<addCols { out[r * newCols + oldCols + c] = src[r * addCols + c] }
        }
        store = out
    }

    /// Keep the permanent prefix + the most recent `maxWorkTokens` temporary tokens.
    private func fifo() {
        let nonPerm = n - permEnd
        if nonPerm <= maxWorkTokens { return }
        let keepStart = n - maxWorkTokens
        let keptCols = permEnd + (n - keepStart)
        func sieve(_ store: [Float], rows: Int) -> [Float] {
            var out = [Float](repeating: 0, count: rows * keptCols)
            for r in 0..<rows {
                var dst = r * keptCols
                for c in 0..<permEnd { out[dst] = store[r * n + c]; dst += 1 }
                for c in keepStart..<n { out[dst] = store[r * n + c]; dst += 1 }
            }
            return out
        }
        key = sieve(key, rows: keyDim)
        shrinkage = sieve(shrinkage, rows: 1)
        valueT.removeSubrange(permEnd * valueDim ..< keepStart * valueDim)
        n = keptCols
    }

    // ----------------------------------------------------------------- read
    nonisolated(unsafe) static var profile: [String: (ms: Double, calls: Int)] = [:]
    nonisolated(unsafe) static var profilingEnabled = false
    @inline(__always) private func lap(_ name: String, _ t0: CFAbsoluteTime) {
        guard Self.profilingEnabled else { return }
        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let p = Self.profile[name] ?? (0, 0)
        Self.profile[name] = (p.ms + ms, p.calls + 1)
    }

    // Reused across steps so a read allocates nothing proportional to N x HW.
    // simLHS depends only on the stored memory and is rebuilt after each write.
    private var simLHS: [Float] = []        // [2*CK+1, N]
    private var simLHSValid = false
    private var simRHS: [Float] = []        // [2*CK+1, HW]
    private var simT: [Float] = []          // [HW, N]
    private var readoutT: [Float] = []      // [HW, CV]

    /// qk,qe: [CK, h, w] channel-first. Returns the memory readout [CV, h, w] (channel-first).
    func readout(queryKey qk: [Float], querySelection qe: [Float]) -> [Float] {
        let t0 = CFAbsoluteTimeGetCurrent()
        if !simLHSValid {
            MemoryMath.similarityLHS(key: key, shrinkage: shrinkage, ck: keyDim, n: n, into: &simLHS)
            simLHSValid = true
        }
        MemoryMath.similarityRHS(qk: qk, qe: qe, ck: keyDim, hw: hw, into: &simRHS)
        MemoryMath.similarityT(lhs: simLHS, rhs: simRHS, ck: keyDim, n: n, hw: hw, into: &simT)
        lap("getSimilarity", t0)
        let t1 = CFAbsoluteTimeGetCurrent()
        let r = MemoryMath.topKReadout(simT: simT, valueT: valueT, n: n, hw: hw, cv: valueDim,
                                       k: min(topK, n), scratch: &readoutT)
        lap("topKSoftmax", t1)                             // top-k softmax + sparse readout
        return r
    }
}

/// Pure, stateless memory math (anisotropic-L2 similarity, top-k softmax, readout). Extracted so
/// it can be exercised on the Mac via the `scripts/` PyTorch parity harness independent of Core ML.
/// All arrays are row-major `[Float]`; batch is 1.
///
/// The similarity is upstream's
///
///     sim[i,j] = (-Σc mk²·qe + 2·Σc mk·qk·qe - Σc qe·qk²) · ms[i] / √CK
///
/// written as one matrix product `simT = rhsᵀ · lhs` over K = 2·CK+1 rows, with
/// lhs = [-mk²·s; 2·mk·s; -s] (s = ms/√CK, scaled per memory token) and
/// rhs = [qe; qk·qe; Σc qe·qk²]. It is computed transposed, `[HW, N]`, so the top-k over the
/// memory dimension reads contiguous rows.
enum MemoryMath {
    /// key[CK,N], shrinkage[N] → lhs[2*CK+1, N].
    static func similarityLHS(key mk: [Float], shrinkage ms: [Float], ck: Int, n: Int,
                              into lhs: inout [Float]) {
        let rows = 2 * ck + 1
        if lhs.count != rows * n { lhs = [Float](repeating: 0, count: rows * n) }
        var invSqrtCk = 1.0 / Float(ck).squareRoot()
        mk.withUnsafeBufferPointer { k in
        ms.withUnsafeBufferPointer { m in
        lhs.withUnsafeMutableBufferPointer { l in
            let s = l.baseAddress! + 2 * ck * n                     // last row: -s
            vDSP_vsmul(m.baseAddress!, 1, &invSqrtCk, s, 1, vDSP_Length(n))   // s = ms/√CK
            for c in 0..<ck {
                let kc = k.baseAddress! + c * n
                let sq = l.baseAddress! + c * n                    // -mk²·s
                let lin = l.baseAddress! + (ck + c) * n            // 2·mk·s
                vDSP_vmul(kc, 1, s, 1, lin, 1, vDSP_Length(n))     // mk·s
                vDSP_vmul(kc, 1, lin, 1, sq, 1, vDSP_Length(n))    // mk²·s
                var minusOne: Float = -1, two: Float = 2
                vDSP_vsmul(sq, 1, &minusOne, sq, 1, vDSP_Length(n))
                vDSP_vsmul(lin, 1, &two, lin, 1, vDSP_Length(n))
            }
            var minusOne: Float = -1
            vDSP_vsmul(s, 1, &minusOne, s, 1, vDSP_Length(n))
        }}}
    }

    /// qk,qe [CK,HW] → rhs[2*CK+1, HW].
    static func similarityRHS(qk: [Float], qe: [Float], ck: Int, hw: Int, into rhs: inout [Float]) {
        let rows = 2 * ck + 1
        if rhs.count != rows * hw { rhs = [Float](repeating: 0, count: rows * hw) }
        qk.withUnsafeBufferPointer { k in
        qe.withUnsafeBufferPointer { e in
        rhs.withUnsafeMutableBufferPointer { r in
            let base = r.baseAddress!
            base.update(from: e.baseAddress!, count: ck * hw)                    // qe
            let kq = base + ck * hw
            vDSP_vmul(k.baseAddress!, 1, e.baseAddress!, 1, kq, 1, vDSP_Length(ck * hw))   // qk·qe
            let b = base + 2 * ck * hw                                           // Σc qe·qk²
            b.update(repeating: 0, count: hw)
            for c in 0..<ck {
                vDSP_vma(kq + c * hw, 1, k.baseAddress! + c * hw, 1, b, 1, b, 1, vDSP_Length(hw))
            }
        }}}
    }

    /// simT[HW, N] = rhsᵀ · lhs.
    static func similarityT(lhs: [Float], rhs: [Float], ck: Int, n: Int, hw: Int,
                            into simT: inout [Float]) {
        if simT.count != hw * n { simT = [Float](repeating: 0, count: hw * n) }
        let k = 2 * ck + 1
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, Int32(hw), Int32(n), Int32(k),
                    1.0, rhs, Int32(hw), lhs, Int32(n), 0.0, &simT, Int32(n))
    }

    /// Per query pixel j (row of simT [HW, N]): softmax over the memory tokens whose similarity is
    /// ≥ the k-th largest of the row (ties at the threshold all survive, as upstream's `>=` mask),
    /// then `readout[:, j] = Σ w_i · valueT[i, :]` over those few tokens. This is the dense
    /// `value @ affinity` of upstream with the zero entries skipped. Returns [CV, HW].
    ///
    /// Pixels are independent, so the work is split across cores with `concurrentPerform`; each
    /// worker writes only its own rows of `scratch` [HW, CV].
    static func topKReadout(simT: [Float], valueT: [Float], n: Int, hw: Int, cv: Int, k: Int,
                            scratch outT: inout [Float]) -> [Float] {
        if outT.count != hw * cv { outT = [Float](repeating: 0, count: hw * cv) }
        // Small chunks so the faster cores pick up more of the work.
        let chunk = 8
        let workers = (hw + chunk - 1) / chunk
        simT.withUnsafeBufferPointer { simP in
        valueT.withUnsafeBufferPointer { valP in
        outT.withUnsafeMutableBufferPointer { outP in
            let sim = simP.baseAddress!, val = valP.baseAddress!, out = outP.baseAddress!
            DispatchQueue.concurrentPerform(iterations: workers) { t in
                let jStart = t * chunk
                guard jStart < hw else { return }
                let jEnd = min(jStart + chunk, hw)
                withUnsafeTemporaryAllocation(of: Float.self, capacity: k) { top in
                withUnsafeTemporaryAllocation(of: Int.self, capacity: n) { sIdx in
                withUnsafeTemporaryAllocation(of: Float.self, capacity: n) { sVal in
                    for j in jStart..<jEnd {
                        let row = sim + j * n
                        let dst = out + j * cv
                        let thr = kthLargest(UnsafeBufferPointer(start: row, count: n), k: k,
                                             scratch: top)
                        let maxV = top[k - 1]
                        var cnt = 0
                        for i in 0..<n where row[i] >= thr { sIdx[cnt] = i; sVal[cnt] = row[i]; cnt += 1 }
                        var sum: Float = 0
                        for s in 0..<cnt { let e = expf(sVal[s] - maxV); sVal[s] = e; sum += e }
                        dst.update(repeating: 0, count: cv)
                        guard sum > 0 else { continue }
                        let inv = 1.0 / sum
                        for s in 0..<cnt {
                            var wgt = sVal[s] * inv
                            vDSP_vsma(val + sIdx[s] * cv, 1, &wgt, dst, 1, dst, 1, vDSP_Length(cv))
                        }
                    }
                }}}
            }
        }}}
        return [Float](unsafeUninitializedCapacity: cv * hw) { o, count in
            vDSP_mtrans(outT, 1, o.baseAddress!, 1, vDSP_Length(cv), vDSP_Length(hw))
            count = cv * hw
        }
    }

    /// k-th largest value of `a` (counting duplicates), in one pass that keeps the k largest seen
    /// so far in `scratch`, sorted ascending. On return `scratch[k-1]` is the maximum.
    static func kthLargest(_ a: UnsafeBufferPointer<Float>, k: Int,
                           scratch top: UnsafeMutableBufferPointer<Float>) -> Float {
        for i in 0..<k {                                    // insertion-sort the first k
            let v = a[i]
            var j = i
            while j > 0 && top[j - 1] > v { top[j] = top[j - 1]; j -= 1 }
            top[j] = v
        }
        var floor = top[0]
        for i in k..<a.count {
            let v = a[i]
            if v <= floor { continue }
            var j = 1                                       // drop top[0], insert v
            while j < k && top[j] < v { top[j - 1] = top[j]; j += 1 }
            top[j - 1] = v
            floor = top[0]
        }
        return floor
    }
}
