---
name: 🚨 Mathematical or System Bug
about: Report a logic error, kernel crash, or Gerbicz failure.
title: "[BUG]: "
labels: bug, integrity-check
assignees: ''

---

## 🛡️ ORTHOS DIAGNOSTIC SUMMARY
**U_RATIONALE:** [Briefly describe what happened vs. what was expected]
**EXPONENT (p):** [e.g., M136279841]
**PROTOCOL_STATE:** [e.g., V10_OMEGA / Gerbicz_Rollback_Loop]

---

## 💻 SYSTEM SPECIFICATIONS
* **GPU:** [e.g., NVIDIA RTX 5090 / Latitude 5480 Pascal]
* **Driver Version:** [e.g., 550.xx]
* **CUDA Version:** [e.g., 12.4]

---

## 🔍 BUG DESCRIPTION
A clear and concise description of what the bug is.

### Steps to Reproduce
1. Start `arithmancy` with exponent `p = ...`
2. Wait for iteration `...`
3. Observe error: [e.g., "R:STRATEGIC(TENSION) - Gerbicz mismatch"]

### Observed Behavior
What actually happened? Did the system crash, or did it produce an incorrect 64-bit residue?

### Expected Behavior
What should have happened according to the Orthos protocol?

---

## 📂 LOG FRAGMENTS
Paste any relevant output from the **Arithmancy Dashboard** or the `logs/` directory here.
\`\`\`text
[PASTE LOGS HERE]
\`\`\`

---

## 🛠️ POSSIBLE FIX/RATIONALE (Optional)
If you have a theory on why the logic drifted (e.g., "L2 cache overflow on older Pascal cards"), provide it here.
