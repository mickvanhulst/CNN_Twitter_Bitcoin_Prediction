import sys
import subprocess

procs = []
for i in range(5):
    proc = subprocess.Popen([sys.executable, 'scrape_user.py',  '--file', '../data/usernames/usernames_{}.npy'.format(i)], shell=True)
    procs.append(proc)

for proc in procs:
    proc.wait()