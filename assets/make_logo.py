"""Generate the logo, README banners and family icons for one library of the
JaxREAM family (JaxREAM, SMVGEAR, MINT, DepConv, FastJ, EMIS_GEOS, Chem_Grid).

Every library shares one honeycomb: the JaxREAM coupler in the centre and one
hexagon per library around it. The mark is drawn in one colour; a library's own
logo fills only its hexagon and leaves the others as outlines.

Run: python assets/make_logo.py            (writes next to this file)
"""

import math
import pathlib
import sys

LIB = "chem_grid"
OUT = pathlib.Path(__file__).resolve().parent

R, r, D = 34.0, 30.5, 34.0 * math.sqrt(3) * 1.07
MODS = [  # cluster angle (deg, SVG y-down), name, glyph
    (300, "fastj", "sun"),
    (0, "mint", "flow"),
    (60, "depconv", "dep"),
    (120, "emis_geos", "emis"),
    (180, "chem_grid", "grid"),
    (240, "smvgear", "ring"),
]
NAMES = ["jaxream"] + [m[1] for m in MODS]
LIBS = {
    "jaxream": {
        "word": "JaxREAM",
        "tagline": "Modular regional chemical transport model in JAX",
        "keys": ["SMVGEAR", "MINT", "DepConv", "FastJ", "EMIS_GEOS", "Chem_Grid"],
    },
    "smvgear": {
        "word": "SMVGEAR",
        "tagline": "Differentiable Gear and Rosenbrock chemistry solvers in JAX",
        "keys": ["Gear BDF", "ROS3, RODAS3", "sparse LU", "REAM, MCM, GEOS-Chem", "jit, grad"],
    },
    "mint": {
        "word": "MINT",
        "tagline": "Differentiable REAM tracer transport in JAX",
        "keys": [
            "Walcek advection",
            "Kz diffusion",
            "mass-consistent winds",
            "multi-GPU",
        ],
    },
    "depconv": {
        "word": "DepConv",
        "tagline": "REAM deposition and convection operators in JAX",
        "keys": ["dry deposition", "wet deposition", "KF-eta convection", "scavenging"],
    },
    "fastj": {
        "word": "FastJ",
        "tagline": "Differentiable REAM Fast-J photolysis in JAX",
        "keys": ["Feautrier radiative transfer", "clouds and aerosol", "69 J channels"],
    },
    "emis_geos": {
        "word": "EMIS_GEOS",
        "tagline": "Differentiable REAM emissions in JAX",
        "keys": ["anthropogenic", "MEGAN biogenic", "soil NOx", "lightning NOx"],
    },
    "chem_grid": {
        "word": "Chem_Grid",
        "tagline": "Shared grid geometry for the JAX atmospheric-chemistry stack",
        "keys": ["grid", "Lambert conformal", "vertical order", "regridding"],
    },
}
FONT = "'Helvetica Neue',Helvetica,Arial,'Segoe UI',sans-serif"
# ink = the one colour of the mark and the name; line = outlines of the other hexagons; paper = glyph on a filled hexagon
THEME = {
    "light": {
        "ink": "#12263F",
        "sub": "#44546A",
        "line": "#B4C0CF",
        "paper": "#FFFFFF",
    },
    "dark": {"ink": "#E6EDF3", "sub": "#9BA7B4", "line": "#3D4A5C", "paper": "#0D1117"},
    # for a page whose background is not known (README footer, module tables)
    "neutral": {
        "ink": "#3D5A80",
        "sub": "#6B7A90",
        "line": "#8D9BAE",
        "paper": "#FFFFFF",
    },
}


def hexagon(cx, cy, rad, attrs):
    pts = " ".join(
        f"{cx + rad * math.cos(math.radians(a)):.2f},{cy + rad * math.sin(math.radians(a)):.2f}"
        for a in range(-90, 270, 60)
    )
    return f'<polygon points="{pts}" {attrs}/>'


def hexpts(rad):
    return " ".join(
        f"{rad * math.cos(math.radians(a)):.2f},{rad * math.sin(math.radians(a)):.2f}"
        for a in range(-90, 270, 60)
    )


RAYS = "".join(
    f'<line x1="{7.5 * math.cos(t):.2f}" y1="{7.5 * math.sin(t):.2f}" '
    f'x2="{11 * math.cos(t):.2f}" y2="{11 * math.sin(t):.2f}"/>'
    for t in [k * math.pi / 4 for k in range(8)]
)
WAVES = "".join(f'<path d="M-15,{y} q3.75,-5 7.5,0 t7.5,0 t7.5,0 t7.5,0"/>' for y in (-8, 0, 8))
GLYPH = {
    "sun": '<circle r="4.6"/>' + RAYS,
    "flow": '<path d="M-11,-4.5 q4,-4 8,0 t8,0"/><path d="M-11,4.5 H8 M3,-0.5 L8.5,4.5 L3,9.5"/>',
    "dep": '<path d="M0,-10 V5 M-5.5,-0.5 L0,5.5 L5.5,-0.5 M-10,10.5 H10"/>',
    "emis": '<path d="M-10,10.5 H10 M-4.5,6 V-7 M-9,-2.5 L-4.5,-8 L0,-2.5 M5.5,6 V-2 M1,2.5 L5.5,-3 L10,2.5"/>',
    "grid": '<rect x="-9" y="-9" width="18" height="18"/><path d="M-3,-9 V9 M3,-9 V9 M-9,-3 H9 M-9,3 H9"/>',
    "ring": f'<polygon points="{hexpts(9.5)}"/><circle r="4.4"/>',
}
STROKE = 'fill="none" stroke-linecap="round" stroke-linejoin="round"'


def cell(cx, cy, rad, glyph, width, filled, t):
    """One hexagon with its glyph: filled in the ink colour, or an outline."""
    if filled:
        box = hexagon(cx, cy, rad, f'fill="{t["ink"]}" stroke="none"')
        pen = t["paper"]
    else:
        box = hexagon(cx, cy, rad - 0.8, f'fill="none" stroke="{t["line"]}" stroke-width="1.6"')
        pen = t["line"]
    return (
        f"{box}\n"
        f'<g transform="translate({cx:.2f},{cy:.2f})" {STROKE} stroke="{pen}" stroke-width="{width}">{glyph}</g>'
    )


def cluster(focus, t):
    """The honeycomb. `focus` is the library whose hexagon is filled; for "jaxream" that is the centre and the
    six around it are drawn in the ink colour as well.
    """
    whole = focus == "jaxream"
    ring = {**t, "line": t["ink"]} if whole else t
    s = []
    for ang, name, g in MODS:
        cx, cy = D * math.cos(math.radians(ang)), D * math.sin(math.radians(ang))
        s.append(cell(cx, cy, r, GLYPH[g], 2.0, name == focus, ring))
    s.append(cell(0, 0, R, WAVES, 2.6, whole, t))
    return "\n".join(s)


EXT = D + r


def module_icon(name):
    t = THEME["neutral"]
    glyph = WAVES if name == "jaxream" else GLYPH[{m[1]: m[2] for m in MODS}[name]]
    scale, width = (1.0, 2.6) if name == "jaxream" else (1.25, 2.0)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-33 -33 66 66" width="44" height="44">'
        f"<title>{name}</title>"
        + hexagon(0, 0, 32, f'fill="{t["ink"]}" stroke="none"')
        + f'<g transform="scale({scale})" {STROKE} stroke="{t["paper"]}" stroke-width="{width}">{glyph}</g>'
        + "</svg>\n"
    )


def banner(lib, theme, W=960, H=240):
    t, cfg, x0 = THEME[theme], LIBS[lib], 300
    keys = "  ·  ".join(cfg["keys"])
    if x0 + 8.2 * len(keys) > W - 20:
        sys.exit(f"{lib}: the keyword line overflows the banner")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
        f"<title>{cfg['word']}</title>\n"
        f'<g transform="translate(138,{H / 2}) scale(0.98)">{cluster(lib, t)}</g>\n'
        f'<line x1="{x0 - 24}" y1="62" x2="{x0 - 24}" y2="{H - 62}" stroke="{t["line"]}" stroke-width="1"/>\n'
        f'<text x="{x0}" y="112" font-family="{FONT}" font-size="56" font-weight="600" fill="{t["ink"]}" stroke="none">{cfg["word"]}</text>\n'
        f'<text x="{x0 + 2}" y="148" font-family="{FONT}" font-size="20" fill="{t["sub"]}" stroke="none">{cfg["tagline"]}</text>\n'
        f'<text x="{x0 + 2}" y="182" font-family="{FONT}" font-size="14.5" fill="{t["sub"]}" stroke="none" '
        f'xml:space="preserve">{keys}</text>\n'
        "</svg>\n"
    )


def main(lib=LIB):
    half = EXT + 8
    (OUT / "logo.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-half:.1f} {-half:.1f} {2 * half:.1f} {2 * half:.1f}" '
        f'width="256" height="256">\n<title>{lib}</title>\n{cluster(lib, THEME["neutral"])}\n</svg>\n'
    )
    for theme in ("light", "dark"):
        (OUT / f"banner-{theme}.svg").write_text(banner(lib, theme))
    (OUT / "modules").mkdir(exist_ok=True)
    for name in NAMES:
        (OUT / "modules" / f"{name}.svg").write_text(module_icon(name))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else LIB)
