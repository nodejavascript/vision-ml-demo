#!/usr/bin/env python3
"""Generate the icon set for vision-ml-demo.

Written 20 September 2026, when George stated the icon rule for the family — part 15b:

  1 · every app under nodejavascript.com ships its OWN mark, and no two may share one;
  2 · every site ships the favicon set AND the apple icon.

This site already had its mark — four rounded squares in #a78bfa on #08050d, from the
day it was built — but it shipped that mark as `favicon.svg` ONLY. There was no
`favicon.ico` and no `apple-touch-icon.png`, so a bookmark, an older browser and an
iOS home screen each got nothing. **The mark was fine; the set was incomplete.**

Nothing here invents a new drawing. It renders the site's existing mark into the
formats the rule requires, so the icon a visitor sees in a tab and the icon they see
on a home screen are the same mark.

    python3 tools/make-icons.py

Run it from the repository root; it writes into site/.
"""
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(os.path.dirname(HERE), "site")

GROUND = (8, 5, 13)          # #08050d — this site's background
MARK = (167, 139, 250)       # #a78bfa — this site's accent

# Geometry in a 32x32 space, matching site/favicon.svg exactly.
SQUARES = [(5.5, 5.5), (18.0, 5.5), (5.5, 18.0), (18.0, 18.0)]
SIDE = 8.5
RADIUS = 2.0
STROKE = 2.0


def draw_mark(img, size, inset=0.0):
    """The four squares, at whatever pixel size is asked for."""
    draw = ImageDraw.Draw(img)
    pad = inset * size
    scale = (size - 2 * pad) / 32.0
    width = max(1, round(STROKE * scale))
    for x, y in SQUARES:
        draw.rounded_rectangle(
            [pad + x * scale, pad + y * scale,
             pad + (x + SIDE) * scale, pad + (y + SIDE) * scale],
            radius=max(1, round(RADIUS * scale)),
            outline=MARK,
            width=width,
        )


def square(size, background=GROUND, inset=0.0):
    # RGB, never RGBA: the apple icon must carry NO alpha channel, because iOS paints
    # transparency black and the mark would arrive on a home screen with black wedges.
    img = Image.new("RGB", (size, size), background)
    draw_mark(img, size, inset)
    return img


def find_font(bold=True, size=48):
    names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
        "NotoSans-Bold.ttf" if bold else "NotoSans-Regular.ttf",
    ]
    for root in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
                 "/usr/share/fonts/truetype/noto", "/usr/share/fonts"):
        for name in names:
            path = os.path.join(root, name)
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def social_card():
    """1200x630, the size a link preview wants."""
    width, height = 1200, 630
    img = Image.new("RGB", (width, height), GROUND)
    bloom = Image.new("RGB", (width, height), GROUND)
    bd = ImageDraw.Draw(bloom)
    for i in range(60, 0, -1):
        t = i / 60.0
        r = int(120 + 520 * t)
        bd.ellipse([160 - r, 110 - r, 160 + r, 110 + r],
                   fill=(int(8 + 40 * (1 - t)), int(5 + 24 * (1 - t)), int(13 + 70 * (1 - t))))
    img = Image.blend(img, bloom, 0.55)

    mark = Image.new("RGB", (190, 190), GROUND)
    draw_mark(mark, 190)
    img.paste(mark, (72, 72))

    draw = ImageDraw.Draw(img)
    draw.text((72, 320), "Watch a model read", font=find_font(True, 62), fill=(240, 236, 250))
    draw.text((72, 396), "an image", font=find_font(True, 62), fill=(240, 236, 250))
    draw.text((72, 486),
              "A small vision model running in your browser. Nothing uploaded.",
              font=find_font(False, 30), fill=(178, 166, 206))
    draw.text((72, 546), "vision-ml-demo.nodejavascript.com",
              font=find_font(True, 26), fill=MARK)
    return img


def main():
    os.makedirs(SITE, exist_ok=True)
    # The SVG is the site's own and is NOT generated here — it is the source of the
    # geometry above, and overwriting it from a raster would lose the vector.
    if not os.path.exists(os.path.join(SITE, "favicon.svg")):
        raise SystemExit("site/favicon.svg is missing — it is the source of this mark")

    master = square(1024)
    for size, name in [(512, "android-chrome-512x512.png"),
                       (192, "android-chrome-192x192.png"),
                       (32, "favicon-32.png")]:
        master.resize((size, size), Image.LANCZOS).save(os.path.join(SITE, name))

    # Apple: 180x180, RGB, opaque. Inset so the mark clears iOS's rounded corners.
    square(180, inset=0.16).save(os.path.join(SITE, "apple-touch-icon.png"))

    # .ico with the three sizes the rule requires
    square(256).save(os.path.join(SITE, "favicon.ico"), sizes=[(16, 16), (32, 32), (48, 48)])

    social_card().save(os.path.join(SITE, "og.png"))

    for name in ["favicon-32.png", "favicon.ico", "apple-touch-icon.png",
                 "android-chrome-192x192.png", "android-chrome-512x512.png", "og.png"]:
        path = os.path.join(SITE, name)
        print(f"{name:32} {os.path.getsize(path):>8,} bytes")


if __name__ == "__main__":
    main()
