def _positive_pair(values, name):
    first, second = values
    first = float(first)
    second = float(second)

    if first <= 0 or second <= 0:
        raise ValueError(
            f"{name} harus lebih besar dari nol."
        )

    return first, second


def fit_image_size(
    image_width,
    image_height,
    canvas_width,
    canvas_height,
    padding=24,
):
    image_width, image_height = (
        _positive_pair(
            (image_width, image_height),
            "Ukuran gambar",
        )
    )
    canvas_width, canvas_height = (
        _positive_pair(
            (canvas_width, canvas_height),
            "Ukuran canvas",
        )
    )
    padding = max(
        float(padding),
        0.0,
    )

    usable_width = max(
        canvas_width - 2.0 * padding,
        1.0,
    )
    usable_height = max(
        canvas_height - 2.0 * padding,
        1.0,
    )

    scale = min(
        usable_width / image_width,
        usable_height / image_height,
    )

    display_width = max(
        1,
        int(round(
            image_width * scale
        )),
    )
    display_height = max(
        1,
        int(round(
            image_height * scale
        )),
    )

    return (
        display_width,
        display_height,
    )


def image_to_canvas(
    point,
    image_size,
    display_size,
    offset,
):
    image_width, image_height = (
        _positive_pair(
            image_size,
            "Ukuran gambar",
        )
    )
    display_width, display_height = (
        _positive_pair(
            display_size,
            "Ukuran display",
        )
    )

    x, y = [
        float(value)
        for value in point
    ]
    offset_x, offset_y = [
        float(value)
        for value in offset
    ]

    return (
        offset_x
        + x * display_width / image_width,
        offset_y
        + y * display_height / image_height,
    )


def canvas_to_image(
    point,
    image_size,
    display_size,
    offset,
):
    image_width, image_height = (
        _positive_pair(
            image_size,
            "Ukuran gambar",
        )
    )
    display_width, display_height = (
        _positive_pair(
            display_size,
            "Ukuran display",
        )
    )

    x, y = [
        float(value)
        for value in point
    ]
    offset_x, offset_y = [
        float(value)
        for value in offset
    ]

    return (
        (x - offset_x)
        * image_width
        / display_width,
        (y - offset_y)
        * image_height
        / display_height,
    )
