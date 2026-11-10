#!/usr/bin/env bash
# Process each session NWB once its download has finished (final file exists).
cd /c/pianshi/invivo
for s in 754312389 732592105 791319847 798911424 754829445 760693773; do
  until [ -f /e/pianshi_invivo/nwb/session_$s.nwb ]; do sleep 30; done
  [ -f processed/ns_$s.npz ] || /c/ProgramData/anaconda3/python.exe process_nwb.py $s
done
echo ALL_PROCESSED
