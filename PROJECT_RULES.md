# Project Rules & Protocols

These rules govern how the AI assistant (Antigravity) interacts with the codebase and external literature.

## Rule 1: Mandatory Changelog Updates
Whenever a major architectural change or significant code addition is made, the `CHANGELOG.md` file MUST be updated. This ensures a persistent, human-readable record of progress alongside standard Git commits.

## Rule 2: Research Paper (PDF) Analysis Protocol
At later stages, when reviewing research papers, the AI must evaluate the text using the following framework:
1. **Methodology Extraction**: Clearly summarize the novel technique or math proposed in the paper.
2. **Gap Analysis**: Compare what the paper is doing against what our current architecture (Semantic Entropy / Information Score) is doing. Identify the deltas.
3. **Cost/Benefit Assessment**: Determine if adopting their methodology would be beneficial (e.g., improves accuracy, reduces token cost) or if it is wrong/sub-optimal for our specific use case.
