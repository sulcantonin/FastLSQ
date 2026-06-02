"""Build the ALS-U v21 4-raft + Superbend lattice in pyAT, parsed from
the canonical .m file shipped by ALS Accelerator Physics.

This module reads element-by-element definitions (drift, quadrupole,
sbend, sextupole, marker, rfcavity) from the .m file and assembles the
12-superperiod ring by replicating the parsed Arc_Up + SUP / SUPSB +
Arc_Down structure.  Function-style elements such as `BPM(n)` are
materialised as numbered Markers; sector markers (`SECT(n)`,
`CenOfStr(n)`) similarly.

The pyAT-bundled `load_m` does not understand AT's FAMLIST/function-
handle dialect, so we do the parsing ourselves --- about 200 lines.
"""
from __future__ import annotations

import os
import re
from typing import List, Dict, Tuple
import numpy as np
import at
from at.lattice import Drift, Quadrupole, Sextupole, Dipole, Marker, RFCavity, Lattice

ENERGY = 2.0e9            # 2 GeV, from the .m file

# ---------------------------------------------------------------------
# Tiny .m parser: extract the magnet / drift / marker / cavity definitions
# ---------------------------------------------------------------------

NUM = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"


def _strip_comments(src: str) -> str:
    out = []
    for ln in src.splitlines():
        i = ln.find("%")
        if i >= 0:
            ln = ln[:i]
        out.append(ln)
    return "\n".join(out)


def parse_elements(path: str) -> Dict[str, "at.Element"]:
    """Return {varname -> AT Element} from the .m file."""
    with open(path) as f:
        src = _strip_comments(f.read())
    src = re.sub(r"\.\.\.\s*\n", " ", src)            # join continuations
    elems: Dict[str, "at.Element"] = {}

    def num(s: str) -> float:
        return float(s.strip())

    # drift('NAME', L, 'pass')
    for m in re.finditer(
        r"(\w+)\s*=\s*drift\('([^']+)'\s*,\s*(" + NUM + r")\s*,\s*'[^']+'\)",
        src):
        var, nm, L = m.group(1), m.group(2), num(m.group(3))
        elems[var] = Drift(nm, L)
    # quadrupole('NAME', L, K, 'pass')
    for m in re.finditer(
        r"(\w+)\s*=\s*quadrupole\('([^']+)'\s*,\s*(" + NUM + r")\s*,\s*("
        + NUM + r")\s*,\s*'[^']+'\)", src):
        var, nm, L, k = m.group(1), m.group(2), num(m.group(3)), num(m.group(4))
        elems[var] = Quadrupole(nm, L, k)
    # sextupole('NAME', L, K2, 'pass')
    for m in re.finditer(
        r"(\w+)\s*=\s*sextupole\('([^']+)'\s*,\s*(" + NUM + r")\s*,\s*("
        + NUM + r")\s*,\s*'[^']+'\)", src):
        var, nm, L, k2 = m.group(1), m.group(2), num(m.group(3)), num(m.group(4))
        # AT 'H' parameter is the multipole strength K2 (m^-3).
        elems[var] = Sextupole(nm, L, k2)
    # sbend('NAME', L, angle, e1, e2, K1, 'pass')
    for m in re.finditer(
        r"(\w+)\s*=\s*sbend\('([^']+)'\s*,\s*(" + NUM + r")\s*,\s*("
        + NUM + r")\s*,\s*(" + NUM + r")\s*,\s*(" + NUM + r")\s*,\s*("
        + NUM + r")\s*,\s*'[^']+'\)", src):
        var = m.group(1); nm = m.group(2)
        L, angle = num(m.group(3)), num(m.group(4))
        e1, e2 = num(m.group(5)), num(m.group(6))
        k1 = num(m.group(7))
        elems[var] = Dipole(nm, L, angle, k1,
                            EntranceAngle=e1, ExitAngle=e2)
    # marker('NAME', 'pass')
    for m in re.finditer(
        r"(\w+)\s*=\s*marker\('([^']+)'\s*,\s*'[^']+'\)", src):
        var, nm = m.group(1), m.group(2)
        elems[var] = Marker(nm)
    # rfcavity('NAME', L, V, f, h, 'pass')
    for m in re.finditer(
        r"(\w+)\s*=\s*rfcavity\('([^']+)'\s*,\s*(" + NUM + r")\s*,\s*("
        + NUM + r")\s*,\s*([^,]+),\s*(\w+)\s*,\s*'[^']+'\)", src):
        var, nm = m.group(1), m.group(2)
        L, V = num(m.group(3)), num(m.group(4))
        # f-expression is HarmNumber*C0/L0 and h is HarmNumber.  We
        # compute f from local constants in the .m file: HarmNumber=328,
        # C0=299792458, L0=196.50969188.
        h = 328
        L0 = 196.50969188
        f = h * 299792458.0 / L0
        elems[var] = RFCavity(nm, L, V, f, h, energy=ENERGY)
    return elems


# ---------------------------------------------------------------------
# Build the ring (one super-period at a time)
# ---------------------------------------------------------------------

def _make_marker(name: str) -> "at.Element":
    return Marker(name)


def _seq(*items) -> List["at.Element"]:
    out = []
    for x in items:
        if isinstance(x, list):
            out.extend(x)
        else:
            out.append(x)
    return out


def build_one_superperiod(elems: Dict[str, "at.Element"],
                          variant: str = "SUP") -> List["at.Element"]:
    """Assemble one ALS-U superperiod (Arc_Up + SUP or SUPSB + Arc_Down).

    variant: 'SUP' for standard, 'SUPSB' for superbend variant.
    Per the .m file the BPMs are labelled by index 1..19 plus 115 for
    the SUPSB.  We mark each with a Marker named BPM<n>.
    """
    e = elems
    BPM = lambda n: _make_marker(f"BPM{n}")
    halfstr = [e["D11"]] * 4
    arc_up = _seq(
        e["D11A"], BPM(1), e["DX"], e["SHH"], e["D12"], e["QF1"],
        e["DX"], BPM(2), e["DX"], e["QD1"], e["D12"], e["SHH2"],
        e["DX"], BPM(3), e["DX"], e["BEND1"],
        e["DX"], BPM(4), e["DX"], e["SD"], e["D12"], e["QF2"],
        e["D12"], e["SF"], e["DX"], BPM(5), e["DX"], e["QF3"],
        e["D15"],
    )
    if variant == "SUPSB":
        sup = _seq(
            e["BEND2b"], e["DX"], BPM(6), e["DX"], e["QF4b"],
            e["DX"], BPM(7), e["DX"], e["BEND3b"],
            e["DX"], BPM(8), e["DX"], e["QF5b"], e["D12"],
            e["QD2b"], e["D33B"], e["SB"], e["D33B"], e["QD3b"],
            e["DX"], BPM(9), e["DX"], e["QF6b"],
            e["DX"], BPM(10), e["DX"], e["BEND3b"],
            e["DX"], BPM(11), e["DX"], e["QF6b"], e["D12"],
            e["QD3b"], e["D33B"], e["SB"], e["D33B"], e["QD2b"],
            e["DX"], BPM(13), e["DX"], e["QF5b"],
            e["DX"], BPM(115), e["DX"], e["BEND3b"],
            e["DX"], BPM(14), e["DX"], e["QF4b"], e["D12"],
            e["BEND2b"],
        )
    else:
        sup = _seq(
            e["BEND2"], e["DX"], BPM(6), e["DX"], e["QF4"],
            e["DX"], BPM(7), e["DX"], e["BEND3"],
            e["DX"], BPM(8), e["DX"], e["QF5"], e["D12"],
            e["BEND3"], e["DX"], BPM(9), e["DX"], e["QF6"],
            e["DX"], BPM(10), e["DX"], e["BEND3"],
            e["DX"], BPM(11), e["DX"], e["QF6"],
            e["DX"], BPM(12), e["DX"], e["BEND3"],
            e["DX"], BPM(13), e["DX"], e["QF5"], e["D12"],
            e["BEND3"], e["DX"], BPM(14), e["DX"], e["QF4"], e["D12"],
            e["BEND2"],
        )
    arc_down = _seq(
        e["D15"], e["QF3"], e["D12"], e["SF"], e["DX"], BPM(15), e["DX"],
        e["QF2"], e["D12"], e["SD"], e["DX"], BPM(16), e["DX"],
        e["BEND1"], e["DX"], e["DX"], e["SHH2"],
        e["DX"], BPM(17), e["DX"], e["QD1"], e["DX"], BPM(18), e["DX"],
        e["QF1"], e["D12"], e["SHH"], e["DX"], BPM(19), e["D11A"],
    )
    return _seq(arc_up, sup, arc_down, halfstr,
                _make_marker(f"CenOfStr"), halfstr)


def build_alsu_ring(path: str) -> "at.Lattice":
    """Return a pyAT Lattice approximating the ALS-U v21 4-raft SB
    storage ring.  Twelve superperiods, 4 with SUPSB, 8 with SUP, plus
    one half-straight prefix and one RF cavity in sector 2."""
    elems = parse_elements(path)
    superperiods = []
    SB_INDICES = {4, 8, 12}                # SUPSB sectors per the .m file
    for sect in range(1, 13):
        kind = "SUPSB" if sect in SB_INDICES else "SUP"
        sp = build_one_superperiod(elems, variant=kind)
        superperiods.extend(sp)
        if sect == 2:
            superperiods.append(elems["CAV"])
    # Prepend a HalfStraight that closes the ring before sector 1.
    superperiods = [elems["D11"]] * 4 + superperiods
    ring = at.Lattice(superperiods, energy=ENERGY, periodicity=1,
                      name="ALS-U v21 4-raft SB")
    return ring
