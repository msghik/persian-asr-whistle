"""Import-only stub of `ctcdecode` for CAT. See third_party/cat_ext_stubs/README.md."""

IS_STUB = True


def _fail(*args, **kwargs):
    raise RuntimeError("`ctcdecode` is an import-only stub here (third_party/cat_ext_stubs). "
                       "This code path is not supported in the persian_asr pipeline.")


def __getattr__(name):
    return _fail


class CTCBeamDecoder:
    def __init__(self, *args, **kwargs):
        _fail()
