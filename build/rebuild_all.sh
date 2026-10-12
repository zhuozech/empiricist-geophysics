#!/bin/zsh
# Rebuild every notebook one at a time (clean timings), logging each; resumable (skips those already done in this pass).
cd "$(dirname "$0")/.."
PY=${PY:-python}
LOG=build/rebuild_all.log
DONE=build/rebuild_all.done
touch $DONE
for f in $(ls build/src | grep -E '^[0-9]-[0-9][0-9]_.*\.py$' | sort); do
  stem=${f%.py}
  if grep -qx "$stem" $DONE; then continue; fi
  echo "$(date +%H:%M:%S) start $stem" >> $LOG
  if $PY build/build.py "$stem" >> $LOG 2>&1; then echo "$stem" >> $DONE; echo "$(date +%H:%M:%S) ok $stem" >> $LOG
  else echo "$(date +%H:%M:%S) FAILED $stem" >> $LOG; fi
done
echo "$(date +%H:%M:%S) ALL DONE" >> $LOG
