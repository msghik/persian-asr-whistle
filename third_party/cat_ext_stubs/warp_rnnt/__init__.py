"""Import-only stub of `warp_rnnt` for CAT. See third_party/cat_ext_stubs/README.md."""

IS_STUB = True


def _fail(*args, **kwargs):
    raise RuntimeError("`warp_rnnt` is an import-only stub here (third_party/cat_ext_stubs). "
                       "This code path is not supported in the persian_asr pipeline.")


def __getattr__(name):
    return _fail


rnnt_loss = _fail
