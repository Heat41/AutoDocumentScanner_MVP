"""Non-destructive precision loupe for dragging document corners on Tk Canvas."""
from PIL import Image, ImageDraw, ImageTk


def draw_corner_magnifier(
    canvas,
    original,
    image_point,
    canvas_point,
    *,
    tag="corner-magnifier",
    zoom=4,
    size=160,
    crop_radius=24,
):
    """Draw magnified pixels from the ORIGINAL image; never modify source/corners.

    Returns a PhotoImage reference which the caller must retain while visible.
    The loupe is automatically repositioned to remain inside the canvas.
    """
    canvas.delete(tag)
    if original is None or image_point is None or canvas_point is None:
        return None

    if original.mode != "RGB":
        original = original.convert("RGB")
    width, height = original.size
    x = min(max(int(round(image_point[0])), 0), width - 1)
    y = min(max(int(round(image_point[1])), 0), height - 1)

    # Crop with padding to ensure the crosshair always marks the exact point.
    radius = max(2, int(crop_radius))
    padded = Image.new("RGB", (radius * 2, radius * 2), "#ECEFF4")
    left = max(0, x - radius)
    top = max(0, y - radius)
    right = min(width, x + radius)
    bottom = min(height, y + radius)
    if right > left and bottom > top:
        padded.paste(
            original.crop((left, top, right, bottom)),
            (left - (x - radius), top - (y - radius)),
        )

    # Nearest-neighbor preserves sharp pixel boundaries for alignment.
    loupe = padded.resize((size, size), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(loupe)
    mid = size // 2
    draw.line((mid, 0, mid, size), fill="#FFFFFF", width=3)
    draw.line((0, mid, size, mid), fill="#FFFFFF", width=3)
    draw.line((mid, 0, mid, size), fill="#E11D48", width=1)
    draw.line((0, mid, size, mid), fill="#E11D48", width=1)
    draw.ellipse((mid-5, mid-5, mid+5, mid+5), outline="#E11D48", width=2)

    cw = max(canvas.winfo_width(), size + 12)
    ch = max(canvas.winfo_height(), size + 12)
    cx, cy = canvas_point
    # Prefer to the right and above the finger/mouse, flip when near edge.
    bx = cx + 30
    if bx + size + 6 > cw:
        bx = cx - 30 - size
    by = cy - 30 - size
    if by < 6:
        by = cy + 30
    bx = min(max(6, bx), max(6, cw - size - 6))
    by = min(max(6, by), max(6, ch - size - 6))

    photo = ImageTk.PhotoImage(loupe, master=canvas)
    canvas.create_rectangle(
        bx - 3, by - 3, bx + size + 3, by + size + 3,
        fill="#FFFFFF", outline="#334155", width=2, tags=(tag,),
    )
    canvas.create_image(bx, by, image=photo, anchor="nw", tags=(tag,))
    canvas.tag_raise(tag)
    return photo
