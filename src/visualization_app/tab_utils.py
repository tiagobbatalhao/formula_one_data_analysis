import colorsys


class ColorConverter:
    @classmethod
    def txt2rgb(cls, value):
        return [int(value[i : i + 2], base=16) for i in [0, 2, 4]]

    @classmethod
    def rgb2txt(cls, value):
        return "".join([hex(int(round(x)))[2:].upper().zfill(2) for x in value])

    @classmethod
    def rgb2hls(cls, value):
        hlsval = colorsys.rgb_to_hls(*[x / 255 for x in value])
        return hlsval

    @classmethod
    def hls2rgb(cls, value):
        rgbval = colorsys.hls_to_rgb(*value)
        return [x * 255 for x in rgbval]

    @classmethod
    def txt2hsl(cls, value):
        color_rgb = cls.txt2rgb(value)
        color_hsl = cls.rgb2hls(color_rgb)
        return color_hsl

    @classmethod
    def hsl2txt(cls, value):
        color_rgb = cls.hls2rgb(value)
        color_txt = cls.rgb2txt(color_rgb)
        return color_txt


def change_hue(color_txt, additive_factor):
    color_hsl = ColorConverter.txt2hsl(color_txt)
    new_h = additive_factor + color_hsl[0]
    while new_h > 1:
        new_h -= 1
    while new_h < 0:
        new_h += 1
    return ColorConverter.hsl2txt([new_h, color_hsl[1], color_hsl[2]])


def change_saturation(color_txt, multiply_factor):
    color_hsl = ColorConverter.txt2hsl(color_txt)
    new_s = multiply_factor * color_hsl[1]
    new_s = max(0, min(1, new_s))
    return ColorConverter.hsl2txt([color_hsl[0], new_s, color_hsl[2]])


def change_lightness(color_txt, multiply_factor):
    color_hsl = ColorConverter.txt2hsl(color_txt)
    new_l = multiply_factor * color_hsl[2]
    new_l = max(0, min(1, new_l))
    return ColorConverter.hsl2txt([color_hsl[0], color_hsl[1], new_l])


def choose_driver_colors(df_):
    df_["driver_index_in_team"] = df_.groupby(["session_id", "team_color"])[
        "year"
    ].transform(lambda x: range(len(x)))
    df_["driver_color"] = df_.apply(
        lambda r: (
            change_hue(
                r["team_color"], 0.05 if r["driver_index_in_team"] == 0 else -0.05
            )
            if isinstance(r["team_color"], str)
            else None
        ),
        axis=1,
    )
    return df_
