# Contributing to Arithmancy ⚖️

Thank you for your interest in Arithmancy. Because this project involves high-stakes computational mathematics and long-running GPU tasks, we maintain a **High-Integrity Coding Standard** based on the Orthos Protocol.

---

## 📜 The Golden Rule: Straight Logic
Every contribution must be "Straight." This means:
1. **Verifiable:** Code must not "guess." Every optimization must have a mathematical proof or a benchmark showing it maintains bit-level accuracy.
2. **Stateless:** Kernels should be stateless wherever possible to allow for Gerbicz-driven rollbacks.
3. **Documented Rationale:** We do not accept "Fix stuff" commit messages. Every change must explain *why* it is necessary and *how* it preserves integrity.

---

## 🛠️ Pull Request Requirements

Before submitting a PR, ensure your branch meets these criteria:

### 1. Mathematical Integrity
* If you are modifying the **DWT** or **Carry Kernels**, you must run the `tests/test_arithmetic.cu` suite.
* New algorithms must include a **Gerbicz-compatibility** analysis. Does this change break the shadow-product logic?

### 2. Rationale Headers
All PR descriptions must start with an Orthos Rationale block:
> **U_RATIONALE:** [Description of change]
> **INTEGRITY_IMPACT:** [None / Low / High]
> **BENCHMARK_RESULT:** [e.g., +5% throughput on RTX 4090]

### 3. Clean Performance
* No branch divergence in GPU warps unless mathematically unavoidable.
* Minimize VRAM round-trips; prioritize L2 cache residency.

---

## 🧪 Testing Process
We use `pytest` for high-level logic and `nvcc` based unit tests for kernels.
\`\`\`bash
# Run the integrity suite
python3 -m pytest tests/
./bin/test_kernels
\`\`\`

---

## 🤝 Community Standards
* **Human-in-the-Loop:** AI-generated PRs (from Copilot/Claude) are welcome but **must** be verified by the human author. Never "blind-merge" agentic code.
* **Boring > Clever:** We prefer simple, readable code over "magic" optimizations that are difficult to audit.

---

## ⚖️ License
By contributing, you agree that your contributions will be licensed under the project's **MIT License**.
