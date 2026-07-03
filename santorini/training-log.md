# Training Logs

1. Run #1 - 2026/06/28
   Platform: Google Colab (CPU)
   Iterations: 11
   Time: 8 hours
   Notes:
   Results:
    - vs previous local best, 64 sims: 68-32
    - vs previous local best, 128 sims: 71-29
    - vs greedy, 64 sims: 92-8
    - _promoted as new baseline_

2. Run #2 - 2026/06/29
   Platform: Google Colab (GPU)
   Iterations: 10
   Time: 30 minutes
   Notes: We made several performance improvements in code, and adjusted parameters of training to take advantage of hardware better, and switched to GPU instead of CPU
   Results:
    - vs previous run best, 128 sims: 66-34
    - vs greedy, 128 sims: 95-5
    - _promoted as new baseline_

3. Run #3 - 2026/06/29
   Platform: Google Colab (GPU)
   Iterations: 20
   Time: 57 minutes
   Notes: Increased iterations
   Results:
    - vs previous run best, 128 sims: 72-28
    - vs greedy, 128 sims: 97-3
    - _promoted as new baseline_

4. Run #4 - 2026/06/29
   Platform: Google Colab (CPU)
   Iterations: 8
   Time: 3 hours
   Notes:
   Results:
    - vs previous run best, 128 sims: 58-42
    - vs greedy, 128 sims: 100-0
    - _promoted as new baseline_

5. Run #5 - 2026/06/29
   Platform: Kaggle (GPU)
   Iterations: 20
   Time: 1 hour
   Notes:
   Results:
    - vs previous run best, 128 sims: 64-36
    - vs greedy, 128 sims: 100-0
    - _promoted as new baseline_

6. Run #6 - 2026/06/29
   Platform: Kaggle (GPU)
   Iterations: 20
   Time: 1 hour
   Notes: Use generated seed instead of static value of 9
   Results:
    - vs previous run best, 128 sims: 73-27
    - vs greedy, 128 sims: 98-2
    - _promoted as new baseline_

7. Run #7 - 2026/06/30
   Platform: Kaggle (GPU)
   Iterations: 60
   Time: 2 hours 11 minutes
   Notes:
   Results:
    - vs previous run best, 128 sims: 83-17
    - vs greedy, 128 sims: 99-1
    - _promoted as new baseline_

8. Run #8 - 2026/06/30
   Platform: Kaggle (GPU)
   Iterations: 40
   Time: 1 hour 25 minutes
   Notes:
   Results:
    - vs previous run best, 128 sims: 69-31
    - vs greedy, 128 sims: 100-0
    - _promoted as new baseline_

9. Run #9 - 2026/06/30
   Platform: Kaggle (GPU)
   Iterations: 20
   Time: 45 minutes
   Notes: Increased history_iters to 20, increased epochs to 5
   Results:
    - vs previous run best, 128 sims: 53-47
    - vs greedy, 128 sims: 100-0
    - Promoted tentatively, but on further testing later it seems at best it's even with run 8, or even a bit weaker

10. Run #10 - 2026/06/30
    Platform: Kaggle (GPU)
    Iterations: 40
    Time: 1 hour 45 minutes
    Notes: increased epochs to 6
    Results:
    - vs previous run best, 128 sims: 67-33
    - vs greedy, 128 sims: 100-0
    - against run 8 it went 56-44 and 52-48 in two 100 game matches, so not significantly better as the record against run 9 might suggest
    - _promoted as new baseline_

11. !! Transition to V2 architecture - 2026/07/01
    Platform: Kaggle (GPU)
    Notes: see santorini_ai_architecture_v2.md
    Results:
    - vs previous run best (V1 run 10), 128 sims: 82-18
    - vs greedy, 128 sims: 100-0
    - Surprising jump in strength, beats previous models easily
    - _promoted as new baseline_

12. Run #12 - 2026/07/02
    Platform: Kaggle (GPU)
    Iterations: 13
    Time:
    Notes: opening position changes
    Results:
    - vs previous run best, 128 sims:
    - _promoted as new baseline_
