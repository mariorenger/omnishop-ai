#!/usr/bin/env python3
"""Builds resource_windows_amd64.syso from assets/app.ico so `go build` embeds the exe icon.
Pure Python, no third-party code.   Usage:  python3 assets/make_syso.py   (run from desktop-agent/)

Result: a COFF object with one .rsrc section (resource tree: RT_ICON images + RT_GROUP_ICON #1).
Go's linker picks up any *_windows_amd64.syso automatically; other platforms ignore it.
"""
import struct, sys

ICO, OUT = 'assets/app.ico', 'resource_windows_amd64.syso'
RT_ICON, RT_GROUP_ICON, LANG = 3, 14, 0x0409

data = open(ICO, 'rb').read()
reserved, typ, count = struct.unpack_from('<HHH', data, 0)
assert typ == 1 and count >= 1, 'not an .ico file'
images, groups = [], []
for i in range(count):
    w, h, colors, res, planes, bits, size, off = struct.unpack_from('<BBBBHHII', data, 6 + 16 * i)
    images.append(data[off:off + size])
    groups.append((w, h, colors, res, planes, bits, size, i + 1))      # resource id = i+1
grp = struct.pack('<HHH', 0, 1, count) + b''.join(struct.pack('<BBBBHHIH', *g) for g in groups)

# ---- resource tree. Each leaf = (data bytes). Layout: dirs first, then data entries, then raw data.
leaves = [(RT_ICON, i + 1, images[i]) for i in range(count)] + [(RT_GROUP_ICON, 1, grp)]
types = sorted({t for t, _, _ in leaves})
DIR, ENT, DENT = 16, 8, 16
def dirhdr(n_id): return struct.pack('<IIHHHH', 0, 0, 0, 0, 0, n_id)

# compute offsets
root_off = 0
type_dir_off, off = {}, DIR + ENT * len(types)
for t in types:
    type_dir_off[t] = off
    off += DIR + ENT * sum(1 for x in leaves if x[0] == t)
lang_dir_off = {}
for t, rid, _ in leaves:
    lang_dir_off[(t, rid)] = off
    off += DIR + ENT
dent_off = {}
for t, rid, _ in leaves:
    dent_off[(t, rid)] = off
    off += DENT
raw_off, pos = {}, off
for t, rid, blob in leaves:
    pos = (pos + 3) & ~3
    raw_off[(t, rid)] = pos
    pos += len(blob)
total = (pos + 3) & ~3

sec = bytearray(total)
def put(o, b): sec[o:o + len(b)] = b
put(root_off, dirhdr(len(types)))
for i, t in enumerate(types):
    put(DIR + ENT * i, struct.pack('<II', t, 0x80000000 | type_dir_off[t]))
for t in types:
    ids = sorted(rid for tt, rid, _ in leaves if tt == t)
    put(type_dir_off[t], dirhdr(len(ids)))
    for j, rid in enumerate(ids):
        put(type_dir_off[t] + DIR + ENT * j, struct.pack('<II', rid, 0x80000000 | lang_dir_off[(t, rid)]))
relocs = []
for t, rid, blob in leaves:
    put(lang_dir_off[(t, rid)], dirhdr(1))
    put(lang_dir_off[(t, rid)] + DIR, struct.pack('<II', LANG, dent_off[(t, rid)]))
    # data entry: RVA (filled by relocation: section start + addend), size, codepage, reserved
    put(dent_off[(t, rid)], struct.pack('<IIII', raw_off[(t, rid)], len(blob), 0, 0))
    put(raw_off[(t, rid)], blob)
    relocs.append(dent_off[(t, rid)])

# ---- COFF container
HDR, SH = 20, 40
raw_ptr = HDR + SH
reloc_ptr = raw_ptr + total
sym_ptr = reloc_ptr + 10 * len(relocs)
coff = struct.pack('<HHIIIHH', 0x8664, 1, 0, sym_ptr, 1, 0, 0)                   # AMD64, 1 section, 1 symbol
coff += struct.pack('<8sIIIIIIHHI', b'.rsrc', 0, 0, total, raw_ptr, reloc_ptr, 0, len(relocs), 0, 0x40000040)
coff += bytes(sec)
for va in relocs:
    coff += struct.pack('<IIH', va, 0, 3)                                         # IMAGE_REL_AMD64_ADDR32NB against symbol 0
coff += struct.pack('<8sIhHBB', b'.rsrc', 0, 1, 0, 3, 0)                          # section symbol (static)
coff += struct.pack('<I', 4)                                                      # empty string table
open(OUT, 'wb').write(coff)
print(f'wrote {OUT}: {len(coff)} bytes, {count} icon sizes')
