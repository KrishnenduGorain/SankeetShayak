"""ASL Sign Language Handshape Definitions.

Defines realistic 21-MediaPipe-landmark configurations for ASL (American Sign Language)
handshapes: alphabet A–Z, numbers 0–9, and common signs (HELLO, YES, NO, etc.).

Landmark indices (MediaPipe):
  0=wrist, 1=thumb_CMC, 2=thumb_MCP, 3=thumb_IP, 4=thumb_tip,
  5=idx_MCP, 6=idx_PIP, 7=idx_DIP, 8=idx_tip,
  9=mid_MCP, 10=mid_PIP, 11=mid_DIP, 12=mid_tip,
  13=ring_MCP, 14=ring_PIP, 15=ring_DIP, 16=ring_tip,
  17=pinky_MCP, 18=pinky_PIP, 19=pinky_DIP, 20=pinky_tip
"""

# ---- Base resting hand (all fingers slightly curled, spread ~0.15 apart) ----
BASE = [
    [0.00, 0.00, 0.00],      # 0 wrist
    [-0.15, 0.25, -0.05],     # 1 thumb CMC
    [-0.22, 0.15, -0.03],     # 2 thumb MCP
    [-0.12, 0.05, -0.02],     # 3 thumb IP
    [-0.08, 0.00, -0.01],     # 4 thumb tip
    [0.00, 0.20, 0.02],       # 5 idx MCP
    [0.08, 0.10, 0.05],       # 6 idx PIP
    [0.12, 0.05, 0.06],       # 7 idx DIP
    [0.14, 0.00, 0.08],       # 8 idx tip
    [0.00, 0.15, 0.04],       # 9 mid MCP
    [0.06, 0.05, 0.07],       # 10 mid PIP
    [0.10, 0.00, 0.09],       # 11 mid DIP
    [0.12, 0.00, 0.10],       # 12 mid tip
    [0.00, 0.08, 0.03],       # 13 ring MCP
    [0.05, -0.02, 0.05],      # 14 ring PIP
    [0.08, -0.06, 0.07],      # 15 ring DIP
    [0.10, -0.10, 0.09],      # 16 ring tip
    [0.00, 0.05, 0.02],       # 17 pinky MCP
    [0.04, -0.08, 0.04],      # 18 pinky PIP
    [0.06, -0.12, 0.05],      # 19 pinky DIP
    [0.08, -0.14, 0.07],      # 20 pinky tip
]

# Fingertip positions index constants
TIP = [4, 8, 12, 16, 20]
MID = [9, 12, 16]   # middle/ring/pinky tips used as spread reference
THUMB_TIP = 4


def _add(a, b, eps=0.003):
    return [a[i] + b[i] * eps for i in range(3)]


def _build(shape):
    """Build full 21-point trajectory from fingertip dict.

    Each key maps to [x, y, z]; intermediate joints interpolate between palm and fingertips.
    """
    traj = list(BASE)
    idx_per_frame = [[5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19]]
    for j, lp in enumerate(idx_per_frame):
        base_palm = traj[lp[0]]
        for fp in lp:
            fpv = shape[fp]
            if len(fpv) == 3:
                fp = [fpv[0], fpv[1], fpv[2]]
            traj[fp] = [base_palm[i] + fp[i] * 0.6 for i in range(3)]

    # Thumb interpolation (separate path)
    if THUMB_TIP in shape:
        tip_pt = shape[THUMB_TIP]
        if len(tip_pt) == 3:
            tip_pt = [tip_pt[0], tip_pt[1], tip_pt[2]]
        for j in range(1, 5):
            lp = j
            if lp in shape:
                continue
            palm = traj[lp]
            t = [0.35, 0.55, 0.75][j - 1]
            traj[lp] = [palm[i] + tip_pt[i] * t for i in range(3)]

    # Wrist stays at origin; smooth small adjustments
    traj[0] = [round(x, 4) for x in traj[0]]

    return traj


# ====== FINGERTIP LAYOUTS PER LETTER ======

LETTERS = {}

# A — fist, fingertips touching
LETTERS['A'] = {'4': [-0.02, 0.00, 0.07], '8': [-0.01, 0.00, 0.06],
                '12': [-0.01, 0.00, 0.05], '16': [-0.01, 0.00, 0.05],
                '20': [-0.02, 0.00, 0.07]}

# B — fingertips line up vertically
LETTERS['B'] = {'8': [0.15, 0.00, 0.10], '12': [0.15, 0.00, 0.10],
                '16': [0.15, 0.00, 0.10], '20': [0.15, 0.00, 0.10]}

# C — curled, fingertips near palm
LETTERS['C'] = {'8': [-0.03, 0.00, 0.07], '12': [-0.02, 0.00, 0.06],
                '16': [-0.02, 0.00, 0.06], '20': [-0.01, 0.00, 0.07]}

# D — dome of fingertips
LETTERS['D'] = {'8': [0.13, 0.00, 0.12], '12': [0.13, 0.00, 0.11],
                '16': [0.13, 0.00, 0.10], '20': [0.13, 0.00, 0.10]}

# E — point, index extended others curled
LETTERS['E'] = {'8': [0.16, 0.00, 0.10], '12': [-0.01, 0.00, 0.05],
                '16': [-0.02, 0.00, 0.05], '20': [-0.01, 0.00, 0.04]}

# F — spread fingers, thumb beside index
LETTERS['F'] = {'8': [0.16, 0.00, 0.10], '12': [0.14, 0.00, 0.09],
                '16': [0.10, 0.00, 0.08], '20': [0.08, 0.00, 0.06]}

# G — three fingers spread, thumb crossed over middle
LETTERS['G'] = {'8': [0.16, 0.00, 0.10], '12': [0.14, 0.00, 0.09],
                '16': [0.12, 0.00, 0.07], '20': [0.08, 0.00, 0.06],
                '4': [-0.05, 0.02, 0.08]}

# H — horizontal line of four fingertips
LETTERS['H'] = {'8': [0.15, 0.00, 0.10], '12': [0.15, 0.00, 0.10],
                '16': [0.15, 0.00, 0.10], '20': [0.15, 0.00, 0.10]}

# I — only index extended
LETTERS['I'] = {'8': [0.16, 0.00, 0.10], '12': [0.03, 0.00, 0.05],
                '16': [0.02, 0.00, 0.04], '20': [0.01, 0.00, 0.03],
                '4': [-0.04, 0.02, 0.05]}

# J — index extended, thumb touches pinky
LETTERS['J'] = {'8': [0.16, 0.00, 0.10], '20': [0.06, 0.00, 0.09],
                '4': [-0.05, 0.0, 0.07]}

# K — fist, thumb and index extended
LETTERS['K'] = {'8': [0.16, 0.00, 0.10], '4': [-0.07, 0.00, 0.08],
                '12': [-0.01, 0.00, 0.05], '16': [-0.01, 0.00, 0.05],
                '20': [-0.01, 0.00, 0.05]}

# L — thumb+index extended, rest touching
LETTERS['L'] = {'8': [0.16, 0.00, 0.10], '4': [-0.07, 0.01, 0.07],
                '12': [0.03, 0.00, 0.06], '16': [0.03, 0.00, 0.05],
                '20': [0.02, 0.00, 0.04]}

# M — fingertips touching, thumb beside
LETTERS['M'] = {'8': [0.12, 0.00, 0.10], '12': [0.12, 0.00, 0.10],
                '16': [0.12, 0.00, 0.10], '20': [0.12, 0.00, 0.10],
                '4': [-0.05, 0.03, 0.06]}

# N — fingertips together, thumb beside
LETTERS['N'] = {'8': [0.10, 0.00, 0.10], '12': [0.10, 0.00, 0.10],
                '16': [0.10, 0.00, 0.10], '20': [0.10, 0.00, 0.10],
                '4': [-0.04, 0.02, 0.06]}

# O — fist with thumb touching pinky tip
LETTERS['O'] = {'8': [-0.01, 0.00, 0.07], '12': [-0.01, 0.00, 0.06],
                '16': [0.05, 0.00, 0.09], '20': [0.05, 0.00, 0.09],
                '4': [-0.02, 0.0, 0.05]}

# P — thumb & index extended, rest folded
LETTERS['P'] = {'8': [0.16, 0.00, 0.10], '4': [-0.07, 0.01, 0.07],
                '12': [0.02, 0.00, 0.05], '16': [0.02, 0.00, 0.04],
                '20': [0.01, 0.00, 0.03]}

# Q — O-hand with index also extended
LETTERS['Q'] = {'8': [0.14, 0.00, 0.09], '16': [0.05, 0.00, 0.09],
                '20': [0.05, 0.00, 0.09], '4': [-0.02, 0.0, 0.05]}

# R — thumb&middle extended, index touches thumb
LETTERS['R'] = {'12': [0.16, 0.00, 0.10], '8': [0.03, 0.00, 0.06],
                '16': [0.05, 0.00, 0.07], '20': [0.04, 0.00, 0.05],
                '4': [-0.05, 0.02, 0.08]}

# S — curved S, fingertips converge inward
LETTERS['S'] = {'8': [0.10, 0.00, 0.09], '12': [0.07, 0.00, 0.08],
                '16': [0.05, 0.00, 0.07], '20': [0.03, 0.00, 0.06]}

# T — thumb touches index
LETTERS['T'] = {'8': [0.16, 0.00, 0.10], '4': [-0.02, 0.0, 0.07],
                '12': [0.01, 0.00, 0.05], '16': [0.01, 0.00, 0.04],
                '20': [0.01, 0.00, 0.03]}

# U — fist, index & pinky extended
LETTERS['U'] = {'8': [0.16, 0.00, 0.10], '20': [0.07, 0.00, 0.09],
                '4': [-0.03, 0.0, 0.05], '12': [-0.01, 0.00, 0.05],
                '16': [-0.01, 0.00, 0.05]}

# V — V-hand, index & middle spread
LETTERS['V'] = {'8': [0.16, 0.00, 0.10], '12': [0.14, 0.00, 0.08],
                '16': [0.10, 0.00, 0.06], '20': [0.06, 0.00, 0.05],
                '4': [-0.05, 0.0, 0.06]}

# W — pinch between index&pinky, thumb&middle crossing
LETTERS['W'] = {'8': [0.16, 0.00, 0.10], '4': [-0.04, 0.0, 0.08],
                '20': [0.08, 0.00, 0.09], '12': [0.10, 0.00, 0.07],
                '16': [0.06, 0.00, 0.06]}

# X — crossed fingers, form X shape
LETTERS['X'] = {'8': [0.16, 0.00, 0.10], '12': [0.13, 0.00, 0.12],
                '16': [0.13, 0.00, 0.08], '20': [0.06, 0.00, 0.09],
                '4': [-0.04, 0.0, 0.07]}

# Y — Y-finger: thumb, index, pinky extended
LETTERS['Y'] = {'8': [0.16, 0.00, 0.10], '4': [-0.05, 0.0, 0.07],
                '20': [0.07, 0.00, 0.09], '12': [0.08, 0.00, 0.07],
                '16': [0.05, 0.00, 0.05]}

# Z — Z-finger: index spread, pinky curved away
LETTERS['Z'] = {'8': [0.16, 0.00, 0.10], '12': [0.14, 0.00, 0.08],
                '16': [0.10, 0.00, 0.07], '20': [0.06, 0.00, 0.06],
                '4': [-0.04, 0.0, 0.06]}


def _number_shape(n):
    """Build number 0-9 handshape from digit configuration."""
    if n == 0:
        tips = {8: [0.16, 0.0, 0.10], 12: [0.16, 0.0, 0.10], 16: [0.16, 0.0, 0.10], 20: [0.16, 0.0, 0.10], 4: [-0.02, 0.0, 0.05]}
    elif n == 1:
        tips = {8: [0.16, 0.0, 0.10], 12: [-0.02, 0.0, 0.05], 16: [-0.01, 0.0, 0.04], 20: [-0.01, 0.0, 0.03], 4: [-0.04, 0.0, 0.05]}
    elif n == 2:
        tips = {8: [0.16, 0.0, 0.10], 12: [0.05, 0.0, 0.05], 16: [0.13, 0.0, 0.08], 20: [0.04, 0.0, 0.04], 4: [-0.04, 0.0, 0.06]}
    elif n == 3:
        tips = {8: [0.16, 0.0, 0.10], 12: [0.14, 0.0, 0.09], 16: [0.10, 0.0, 0.08], 20: [0.05, 0.0, 0.05], 4: [-0.03, 0.0, 0.05]}
    elif n == 4:
        tips = {8: [0.16, 0.0, 0.10], 12: [0.14, 0.0, 0.09], 16: [0.12, 0.0, 0.08], 20: [0.09, 0.0, 0.07], 4: [-0.03, 0.0, 0.05]}
    elif n == 5:
        tips = {8: [0.16, 0.0, 0.10], 12: [0.14, 0.0, 0.09], 16: [0.12, 0.0, 0.08], 20: [0.06, 0.0, 0.09], 4: [-0.03, 0.0, 0.05]}
    elif n == 6:
        tips = {8: [0.16, 0.0, 0.10], 4: [-0.03, 0.0, 0.07], 12: [0.04, 0.0, 0.05], 16: [0.05, 0.0, 0.05], 20: [0.04, 0.0, 0.05]}
    elif n == 7:
        tips = {8: [0.16, 0.0, 0.10], 12: [0.10, 0.0, 0.08], 16: [0.05, 0.0, 0.06], 20: [0.06, 0.0, 0.06], 4: [-0.03, 0.0, 0.05]}
    elif n == 8:
        tips = {8: [0.16, 0.0, 0.10], 12: [0.14, 0.0, 0.09], 16: [0.14, 0.0, 0.09], 20: [0.14, 0.0, 0.09], 4: [-0.03, 0.0, 0.05]}
    else:  # 9
        tips = {8: [0.16, 0.0, 0.10], 12: [0.12, 0.0, 0.09], 16: [0.05, 0.0, 0.09], 20: [0.07, 0.0, 0.07], 4: [-0.03, 0.0, 0.05]}
    return tips


NUMBERS = {str(n): _number_shape(int(n)) for n in range(10)}


# ====== COMMON SIGN LAYOUTS ======

COMMON_SIGNS = {
    'HELLO': {'8': [0.16, 0.0, 0.10], '12': [0.16, 0.0, 0.10],
              '16': [0.16, 0.0, 0.10], '20': [0.16, 0.0, 0.10],
              '4': [-0.05, 0.0, 0.07]},          # wave-like horizontal line
    'YES': {'8': [0.15, 0.0, 0.08], '12': [0.13, 0.0, 0.08],
            '16': [0.10, 0.0, 0.07], '20': [0.08, 0.0, 0.06],
            '4': [-0.03, 0.0, 0.06]},            # V-shape
    'NO': {'8': [-0.03, 0.0, 0.06], '12': [-0.02, 0.0, 0.06],
           '16': [0.03, 0.0, 0.07], '20': [0.05, 0.0, 0.08],
           '4': [-0.04, 0.0, 0.05]},             # inverted N-shape
    'THANK_YOU': {'8': [0.14, 0.0, 0.10], '12': [0.12, 0.0, 0.08],
                  '16': [0.08, 0.0, 0.06], '20': [0.06, 0.0, 0.05],
                  '4': [-0.04, 0.0, 0.06]},       # flat H, move outward
    'PLEASE': {'8': [0.14, 0.0, 0.09], '12': [0.12, 0.0, 0.08],
               '16': [0.08, 0.0, 0.07], '20': [0.06, 0.0, 0.06],
               '4': [-0.04, 0.0, 0.06]},          # O-hand near chin
    'SORRY': {'8': [0.12, 0.0, 0.09], '12': [0.10, 0.0, 0.09],
              '16': [0.10, 0.0, 0.09], '20': [0.10, 0.0, 0.09],
              '4': [-0.04, 0.0, 0.06]},            # fist, rub sides
    'I_LOVE_YOU': {'8': [0.15, 0.0, 0.09], '12': [0.14, 0.0, 0.07],
                   '16': [0.10, 0.0, 0.06], '20': [0.08, 0.0, 0.05],
                   '4': [-0.03, 0.0, 0.05]},        # 5-pointed star shape
    'HELP': {'8': [0.15, 0.0, 0.08], '12': [0.13, 0.0, 0.07],
             '16': [0.10, 0.0, 0.06], '20': [0.08, 0.0, 0.05],
             '4': [-0.03, 0.0, 0.05]},            # 5-pointed star
    'STOP': {'8': [0.16, 0.0, 0.10], '12': [0.14, 0.0, 0.08],
             '16': [0.10, 0.0, 0.06], '20': [0.08, 0.0, 0.05],
             '4': [-0.03, 0.0, 0.06]},            # flat H, moving across forehead
    'MORE': {'8': [0.16, 0.0, 0.10], '12': [0.13, 0.0, 0.08],
             '16': [0.10, 0.0, 0.06], '20': [0.08, 0.0, 0.05],
             '4': [-0.03, 0.0, 0.06]},             # C-curled to open C
    'EAT': {'8': [0.15, 0.0, 0.08], '12': [0.13, 0.0, 0.06],
            '16': [0.10, 0.0, 0.05], '20': [0.08, 0.0, 0.04],
            '4': [-0.02, 0.0, 0.05]},             # claw hands forward
    'DRINK': {'8': [0.14, 0.0, 0.08], '12': [0.12, 0.0, 0.06],
              '16': [0.08, 0.0, 0.05], '20': [0.06, 0.0, 0.04],
              '4': [-0.02, 0.0, 0.05]},             # flat A, tilt head
}


def get_signs():
    """Return ordered list of all class label names (alphabetical, then numbers, then common signs)."""
    labels = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
    numbers = [str(n) for n in range(10)]
    common = list(COMMON_SIGNS.keys())
    return labels + numbers + common
