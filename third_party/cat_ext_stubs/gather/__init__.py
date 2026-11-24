"""Import-only stub of `gather` for CAT. See third_party/cat_ext_stubs/README.md."""

IS_STUB = True


def _fail(*args, **kwargs):
    raise RuntimeError("`gather` is an import-only stub here (third_party/cat_ext_stubs). "
                       "This code path is not supported in the persian_asr pipeline.")


def __getattr__(name):
    return _fail
