from events.events_synchronization import events_synchronization
import numpy as np


def test_event_synchronization():
    events_i = [0, 5, 10, 15]
    events_j = [1, 16]
    es = events_synchronization(events_i, events_j)
    es2 = events_synchronization(events_j, events_i)
    assert es == es2
    assert np.isclose(es, 2 / np.sqrt(8))
    assert events_synchronization(events_i, events_i) == 1
    assert events_synchronization(events_i, []) == 0
    assert events_synchronization(events_i, events_j, tau=0.5) == 0
    assert events_synchronization(events_i, events_j, tau=6) == 2 / np.sqrt(8)
    assert events_synchronization([0, 1], [0, 1], tau=0) == 1
    assert events_synchronization([0, 2], [1], tau=2) == 1 / np.sqrt(2)
    assert events_synchronization([0, 2, 3], [0], tau=2) == 1 / np.sqrt(3)
    assert events_synchronization([0, 2], [1, 2], tau=2) == 1
    assert events_synchronization([0, 2], [1, 4], tau=5) == 1
    assert events_synchronization([0, 2], [1, 3], tau=1) == 1
    assert events_synchronization([0, 2], [1, 4], tau=1) == 1 / 2
