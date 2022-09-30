policy_factory = dict()
def none_policy():
    return None

from crowd_nav.policy.orca import ORCA
from crowd_nav.policy.pas_rnn import PASRNN

policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
policy_factory['pas_rnn'] = PASRNN

