"""
Version and aliases

"""
__version__ = '0.1.3'

__all__ = [
    'nest',
    'nest_list',
    'fold',
    'fold_list',
    'rem',
    'exponential',
    'cosine',
    'kaiser',
    'frequency',
    'fma',
    'gali',
    'hsvd',
    'fli',
    'iterate',
    'prime',
    'chain',
    'monodromy',
    'unique',
    'combine',
    'classify',
    'manifold'
]

from tohubohu.functional import nest
from tohubohu.functional import nest_list
from tohubohu.functional import fold
from tohubohu.functional import fold_list

from tohubohu.rem import rem

from tohubohu.filter import exponential
from tohubohu.filter import cosine
from tohubohu.filter import kaiser

from tohubohu.frequency import frequency

from tohubohu.fma import fma

from tohubohu.gali import gali

from tohubohu.hsvd import hsvd

from tohubohu.fli import fli

from tohubohu.fp import iterate
from tohubohu.fp import prime
from tohubohu.fp import chain
from tohubohu.fp import monodromy
from tohubohu.fp import unique
from tohubohu.fp import combine
from tohubohu.fp import classify
from tohubohu.fp import manifold