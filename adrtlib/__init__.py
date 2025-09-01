__version__ = "0.0.2"

try:
    from ._adrtlib import (
        ids_recursive as ids_recursive,
        ids_non_recursive as ids_non_recursive,
        idt_recursive as idt_recursive,
        idt_non_recursive as idt_non_recursive,
        ds_recursive as ds_recursive,
        ds_non_recursive as ds_non_recursive,
        dt_recursive as dt_recursive,
        dt_non_recursive as dt_non_recursive,
        round05 as round05,
    )

    # aliases
    fht2ids_recursive = ids_recursive
    fht2ids_non_recursive = ids_non_recursive
    fht2idt_recursive = idt_recursive
    fht2idt_non_recursive = idt_non_recursive
    fht2ds_recursive = ds_recursive
    fht2ds_non_recursive = ds_non_recursive
    fht2dt_recursive = dt_recursive
    fht2dt_non_recursive = dt_non_recursive
except ImportError:
    pass  # fine, c++ version failed to compile
