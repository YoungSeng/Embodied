"""Source-tree compatibility wrapper.

Formal checkpoints receive the self-contained implementation from
``eaglevl/model/locany/relation_modules.py`` during export.  Keeping this
wrapper avoids maintaining two training-time copies of the same module.
"""

from eaglevl.model.locany.relation_modules import *  # noqa: F401,F403

