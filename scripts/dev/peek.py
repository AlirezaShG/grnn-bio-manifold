import json, glob, os, sys
pat = sys.argv[1] if len(sys.argv) > 1 else 'results/v11/dev/*/runs/*/*/history.json'
for f in sorted(glob.glob(pat)):
    h = json.load(open(f)); p = f.replace(os.sep, '/').split('/')
    last = [r for r in h if 'validation_accuracy' in r]
    print(p[-5], p[-3], p[-2], h[-1]['step'], round(h[-1]['training_seconds']),
          round(last[-1]['validation_accuracy'], 3) if last else None)
