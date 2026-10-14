#!/usr/bin/env python3
# Install a Container Device Interface (CDI) spec on a Dragonwing board.
#
# Vendored, near-verbatim, from the official Qualcomm / Edge Impulse example:
#   github.com/edgeimpulse/qc-ai-test-docs-examples  ->  docker/cdi/install_cdi.py
# (referenced by dragonwingdocs.qualcomm.com/Ubuntu/tools/docker). Kept here so
# this repo is self-contained. Only change vs upstream: the "Missing ..." log
# line avoids a nested same-quote f-string so it runs on Python < 3.12 too.
#
# It copies the given CDI file to /etc/cdi/cdi-hw-acc.json, dropping any mount
# whose hostPath is absent on this board (so a spec that lists optional libs is
# safe). Needs sudo because /etc/cdi is root-owned.
#
# Usage:
#   sudo python3 install_cdi.py --file cdi/cdi-hw-acc-8275.json

import argparse, os, json

parser = argparse.ArgumentParser(description='Install CDI on Dragonwing IoT boards')
parser.add_argument('--file', type=str, required=True, help='CDI input file')
args, unknown = parser.parse_known_args()

if os.path.exists('/etc/cdi/cdi-hw-acc.json'):
    print('/etc/cdi/cdi-hw-acc.json already exists. Remove it first before continuing.')
    exit(1)

if not os.path.exists(args.file):
    print(f'{args.file} (via --file) does not exist')
    exit(1)

# Check if directory exists
if not os.path.exists('/etc/cdi'):
    # Check if we can create it in /etc
    if not os.access(os.path.dirname('/etc/cdi'), os.W_OK):
        print(f"{os.path.dirname('/etc/cdi')} is not writable. Re-run this script with sudo.")
        exit(1)
    os.mkdir('/etc/cdi')
else:
    # Directory exists -> check if writable
    if not os.access('/etc/cdi', os.W_OK):
        print(f"/etc/cdi is not writable. Re-run this script with sudo.")
        exit(1)

with open(args.file, 'r') as f:
    cdi = json.loads(f.read())

print('Finding missing mount paths...')
for device in cdi['devices']:
    new_mounts = []

    for mount in device['containerEdits']['mounts']:
        host_path = mount['hostPath']
        if not os.path.exists(host_path):
            print(f'    Missing {host_path}')
        else:
            new_mounts.append(mount)

    device['containerEdits']['mounts'] = new_mounts

print('')
print('Writing to /etc/cdi/cdi-hw-acc.json...')
with open('/etc/cdi/cdi-hw-acc.json', 'w') as f:
    f.write(json.dumps(cdi, indent=4))
print('Writing to /etc/cdi/cdi-hw-acc.json OK')
