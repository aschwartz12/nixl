#!/bin/bash
# Diagnostic for the connection-setup hang. Run (from the milestone directory) next to
# an ep-modes.sbatch job (EP_MODES=new) of a build with hsdiag.patch applied: on a run's
# first handshake timeout it finds the agent the others wait for (its HSDIAG lines name
# it; the main thread's tid is its pid) and sends it signal 44 (SIGRTMIN+10), on which
# that process prints every thread's stack to stderr, one thread at a time.
# Usage: hang-dump-watch.sh <job id>   (env LOG: the job's log, default logs/efa-cpu-proxy/*-<job>.log)
J=$1
LOG=${LOG:-$(ls -t logs/efa-cpu-proxy/*-$J.log | head -1)}
done_reps=""
while squeue -h -j "$J" 2>/dev/null | grep -q .; do
    if [[ -f $LOG ]]; then
        rep=$(grep -E '^=== mode=new rep=' "$LOG" | tail -1 | sed -E 's/.*rep=([0-9]+).*/\1/')
        sect=$(awk -v r="$rep" '$0 ~ "^=== mode=new rep="r" " {p=1} p' "$LOG")
        peer=$(grep -m1 -oE "Handshake from peer '[0-9]+' not received" <<<"$sect" | grep -oE "[0-9]+")
        if [[ -n $rep && -n $peer && " $done_reps " != *" $rep "* ]]; then
            pid=$(grep -E "HSDIAG $peer load-begin " <<<"$sect" | awk '{print $3}' | head -1)
            echo "rep $rep hung on agent $peer (pid $pid); dumping it at $(date +%T)"
            srun --jobid="$J" --overlap --nodes=4 --ntasks-per-node=1 --time=00:02:00 bash -c "
                if [[ -d /proc/$pid ]] && grep -q libplugin_LIBFABRIC /proc/$pid/maps 2>/dev/null; then kill -44 $pid && echo \$(hostname) dumped $pid; fi" 2>&1 | tail -4
            done_reps="$done_reps $rep"
        fi
    fi
    sleep 5
done
echo "job $J finished"
