mkdir -p temp

# Start optimization
PYTHONPATH=$(realpath ..) python3 run_use_case.py --problem mode_share --algorithm opdyts --log_path temp/log.p

# Plot progress for opdyts or SPSA (also possible while running)
PYTHONPATH=$(realpath ..) python3 plotting/plot_opdyts.py temp/log.p
#PYTHONPATH=$(realpath ..) python3 plotting/plot_spsa.py temp/log.p

# Plot objective
PYTHONPATH=$(realpath ..) python3 plotting/plot_mode_share.py temp/log.p
#PYTHONPATH=$(realpath ..) python3 plotting/plot_travel_time.py temp/log.p
