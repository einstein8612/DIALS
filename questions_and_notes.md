# Questions and Notes

- Add convergence threshold
- Are you allowed to converge on rewards or only on loss?

- Managed to replicate results in the paper reliably

- Ditched GPU conversion since performance was worse and lots of maintainability issues
- Performance not really an issue as original benchmarks were done on an i7

- How do we improve DIALS for our project?
- Possible ideas:
  - Add more context like time of day, weather etc. -> see comment in paper
    - Local regions in paper are the same but can be made different
  - Add autonomous vehicles that regulate traffic as well

- Influence sources = External variables aka cars coming into the traffic section
- What is influence and an influence source in the context of traffic light intersections?
- What does the predictor approximate?

- The paper mentions currently influence sources are discrete but can be made continuous

- FNNPolicy is used since past doesnt affect future in traffic light control

- Rewards are different in DIALS paper than in IALS; Our benchmark works for IALS
