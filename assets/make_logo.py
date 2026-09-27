"""Generate the logo, README banners and family icons for one library of the
JaxREAM family (JaxREAM, SMVGEAR, MINT, DepConv, FastJ, EMIS_GEOS, Chem_Grid).

Every library shares one honeycomb: the JaxREAM coupler hub in the centre and
one hexagon per library around it. A library's own logo colours only its hex.

Run: python assets/make_logo.py            (writes next to this file)
"""
import math
import pathlib
import sys

LIB = "chem_grid"
OUT = pathlib.Path(__file__).resolve().parent

R, r, D = 34.0, 30.5, 34.0 * math.sqrt(3) * 1.07
HUB = ("#818CF8", "#4338CA")
MODS = [  # cluster angle (deg, SVG y-down), name, light, dark, glyph
    (300, "fastj",     "#FDE047", "#F59E0B", "sun"),
    (0,   "mint",      "#67E8F9", "#0891B2", "flow"),
    (60,  "depconv",   "#93C5FD", "#2563EB", "dep"),
    (120, "emis_geos", "#86EFAC", "#16A34A", "emis"),
    (180, "chem_grid", "#D8B4FE", "#9333EA", "grid"),
    (240, "smvgear",   "#FDBA74", "#EA580C", "ring"),
]
COLOR = {m[1]: (m[2], m[3]) for m in MODS} | {"jaxream": HUB}
LIBS = {
    "jaxream": dict(word=("Jax", "REAM"),
                    tagline="Modular regional chemical transport, written in JAX · GPU &amp; CPU",
                    chips=None),
    "smvgear": dict(word="SMVGEAR",
                    tagline="Differentiable SMVGEAR II chemistry solver in JAX · GPU &amp; CPU",
                    chips=["Gear BDF", "sparse LU", "REAM · MCM · GEOS-Chem", "jit + grad"]),
    "mint": dict(word="MINT",
                 tagline="Differentiable REAM tracer transport in JAX · GPU &amp; CPU",
                 chips=["Walcek advection", "Kz diffusion", "ADJUV winds", "multi-GPU"]),
    "depconv": dict(word="DepConv",
                    tagline="REAM deposition and convection operators in JAX · GPU &amp; CPU",
                    chips=["dry dep", "wet dep", "KF-eta convection", "scavenging"]),
    "chem_grid": dict(word="Chem_Grid",
                      tagline="Shared grid geometry for the JAX atmospheric-chemistry stack",
                      chips=["Grid", "Lambert conformal", "vertical order", "regridding"]),
}
FONT = "'Inter','Segoe UI','Helvetica Neue',Helvetica,Arial,sans-serif"
MONO = "'JetBrains Mono','SFMono-Regular',Consolas,'Liberation Mono',Menlo,monospace"


def rounded_hex(cx, cy, rad, fill, k=0.16, extra=""):
    pts = [(cx + rad * math.cos(math.radians(a)), cy + rad * math.sin(math.radians(a)))
           for a in range(-90, 270, 60)]
    d = ""
    for i in range(6):
        p0, p1, p2 = pts[i - 1], pts[i], pts[(i + 1) % 6]
        a = (p1[0] + (p0[0] - p1[0]) * k, p1[1] + (p0[1] - p1[1]) * k)
        b = (p1[0] + (p2[0] - p1[0]) * k, p1[1] + (p2[1] - p1[1]) * k)
        d += (f"M{a[0]:.2f},{a[1]:.2f}" if i == 0 else f"L{a[0]:.2f},{a[1]:.2f}")
        d += f"Q{p1[0]:.2f},{p1[1]:.2f} {b[0]:.2f},{b[1]:.2f}"
    return f'<path d="{d}Z" fill="{fill}"{extra}/>'


def hexpts(rad):
    return " ".join(f"{rad * math.cos(math.radians(a)):.2f},{rad * math.sin(math.radians(a)):.2f}"
                    for a in range(-90, 270, 60))


RAYS = "".join(f'<line x1="{7.5 * math.cos(t):.2f}" y1="{7.5 * math.sin(t):.2f}" '
               f'x2="{11 * math.cos(t):.2f}" y2="{11 * math.sin(t):.2f}"/>'
               for t in [k * math.pi / 4 for k in range(8)])
WAVES = "".join(f'<path d="M-15,{y} q3.75,-5 7.5,0 t7.5,0 t7.5,0 t7.5,0"/>' for y in (-8, 0, 8))
GLYPH = {
    "sun": '<circle r="4.6"/>' + RAYS,
    "flow": '<path d="M-11,-4.5 q4,-4 8,0 t8,0"/><path d="M-11,4.5 H8 M3,-0.5 L8.5,4.5 L3,9.5"/>',
    "dep": '<path d="M0,-10 V5 M-5.5,-0.5 L0,5.5 L5.5,-0.5 M-10,10.5 H10"/>',
    "emis": '<path d="M-10,10.5 H10 M-4.5,6 V-7 M-9,-2.5 L-4.5,-8 L0,-2.5 M5.5,6 V-2 M1,2.5 L5.5,-3 L10,2.5"/>',
    "grid": '<rect x="-9" y="-9" width="18" height="18" rx="2"/><path d="M-3,-9 V9 M3,-9 V9 M-9,-3 H9 M-9,3 H9"/>',
    "ring": f'<polygon points="{hexpts(9.5)}"/><circle r="4.4"/>',
}
STROKE = 'fill="none" stroke-linecap="round" stroke-linejoin="round"'


def stroke(color="#fff"):
    return f'{STROKE} stroke="{color}"'


def grad(gid, lo, hi):
    return (f'<linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
            f'<stop offset="0" stop-color="{lo}"/><stop offset="1" stop-color="{hi}"/></linearGradient>')


def cluster(focus, ghost, prefix):
    """Honeycomb; every hex in colour if focus == 'jaxream', else only `focus`."""
    lit = lambda name: focus == "jaxream" or name == focus
    s = ["<defs>", grad(f"{prefix}hub", *HUB)]
    s += [grad(f"{prefix}{n}", lo, hi) for _, n, lo, hi, _ in MODS]
    s.append("</defs>")
    for ang, name, lo, hi, g in MODS:
        cx, cy = D * math.cos(math.radians(ang)), D * math.sin(math.radians(ang))
        on = lit(name)
        s.append(rounded_hex(cx, cy, r, f"url(#{prefix}{name})" if on else ghost[0]))
        s.append(f'<g transform="translate({cx:.2f},{cy:.2f})" {stroke("#fff" if on else ghost[1])} '
                 f'stroke-width="2.4">{GLYPH[g]}</g>')
    on = focus == "jaxream"
    s.append(rounded_hex(0, 0, R, f"url(#{prefix}hub)" if on else ghost[0]))
    s.append(f'<g {stroke("#fff" if on else ghost[1])} stroke-width="3.2">{WAVES}</g>')
    return "\n".join(s)


GHOST = {"light": ("#E8ECF2", "#FFFFFF"), "dark": ("#1C2230", "#2D3648"),
         "neutral": ("#94A3B833", "#94A3B866")}
EXT = D + r


def module_icon(name):
    lo, hi = COLOR[name]
    glyph = (f'<g {stroke()} stroke-width="3">{WAVES}</g>' if name == "jaxream" else
             f'<g transform="scale(1.25)" {stroke()} stroke-width="2.3">{GLYPH[dict((m[1], m[4]) for m in MODS)[name]]}</g>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="-33 -33 66 66" width="44" height="44">'
            f'<title>{name}</title><defs>{grad("g", lo, hi)}</defs>'
            + rounded_hex(0, 0, 32, "url(#g)") + glyph + "</svg>\n")


def banner(lib, theme, W=960, H=280):
    fg, sub, chip, chipline = (("#0F172A", "#475569", "#F8FAFC", "#E2E8F0") if theme == "light"
                               else ("#E6EDF3", "#9BA7B4", "#161B22", "#30363D"))
    cfg, x0 = LIBS[lib], 300
    if lib == "jaxream":
        word = (f'<tspan fill="url(#word)">{cfg["word"][0]}</tspan>'
                f'<tspan fill="{fg}">{cfg["word"][1]}</tspan>')
        wgrad = ('<linearGradient id="word" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#6366F1"/>'
                 '<stop offset="0.5" stop-color="#0EA5E9"/><stop offset="1" stop-color="#10B981"/></linearGradient>')
        chips = [(n, COLOR[n][1]) for n in ("smvgear", "mint", "depconv", "fastj", "emis_geos", "chem_grid")]
    else:
        lo, hi = COLOR[lib]
        word = f'<tspan fill="url(#word)">{cfg["word"]}</tspan>'
        wgrad = (f'<linearGradient id="word" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{lo if theme == "dark" else hi}"/>'
                 f'<stop offset="1" stop-color="{hi}"/></linearGradient>')
        chips = [(c, hi) for c in cfg["chips"]]
    out, x = [], x0
    for text, dot in chips:
        w = 34 + 9.0 * len(text.replace("&amp;", "&"))
        out.append(f'<rect x="{x:.0f}" y="202" width="{w:.0f}" height="30" rx="15" fill="{chip}" stroke="{chipline}"/>'
                   f'<circle cx="{x + 16:.0f}" cy="217" r="5" fill="{dot}"/>'
                   f'<text x="{x + 27:.0f}" y="222" font-family="{MONO}" font-size="15" fill="{fg}">{text}</text>')
        x += w + 10
    if x - 10 > W - 15:
        sys.exit(f"{lib}: chips overflow the banner ({x - 10:.0f} > {W - 15})")
    title = "".join(cfg["word"]) if isinstance(cfg["word"], tuple) else cfg["word"]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">\n'
            f'<title>{title}</title>\n<defs>{wgrad}</defs>\n'
            f'<g transform="translate(150,{H / 2}) scale(1.08)">{cluster(lib, GHOST[theme], "b")}</g>\n'
            f'<text x="{x0 - 4}" y="128" font-family="{FONT}" font-size="86" font-weight="800" letter-spacing="-2">{word}</text>\n'
            f'<text x="{x0}" y="172" font-family="{FONT}" font-size="23" fill="{sub}">{cfg["tagline"]}</text>\n'
            + "\n".join(out) + "\n</svg>\n")


def main(lib=LIB):
    half = EXT + 8
    (OUT / "logo.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-half:.1f} {-half:.1f} {2 * half:.1f} {2 * half:.1f}" '
        f'width="256" height="256">\n<title>{lib}</title>\n{cluster(lib, GHOST["neutral"], "l")}\n</svg>\n')
    for theme in ("light", "dark"):
        (OUT / f"banner-{theme}.svg").write_text(banner(lib, theme))
    (OUT / "modules").mkdir(exist_ok=True)
    for name in COLOR:
        (OUT / "modules" / f"{name}.svg").write_text(module_icon(name))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else LIB)
