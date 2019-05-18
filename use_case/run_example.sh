mkdir -p temp

# Start optimization
PYTHONPATH=$(realpath ..) python3 run_use_case.py mode_share opdyts

# Plot progress (also possible while running)
PYTHONPATH=$(realpath ..) python3 plot_opdyts.py temp/history_opdyts.p
