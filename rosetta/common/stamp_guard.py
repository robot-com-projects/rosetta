#!/usr/bin/env python3
# Copyright 2026 robot.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Monotonicity guard for the stamps a bag's streams are resampled on."""

from __future__ import annotations


class StampMonotonicityGuard:
    """Rejects a stream whose message stamps step backwards.

    Bag receive time is monotonic by construction, so only header stamped
    streams can regress; a regression breaks the nearest stamp resampling
    (an old sample wins a later tick) and shows up as video jumping back.
    """

    def __init__(self, max_backward_jump_s: float = 0.0) -> None:
        """Initialize the guard.

        Args:
            max_backward_jump_s: Largest tolerated backward step, in seconds.
                ``0`` rejects any regression while still allowing repeated
                stamps; a negative value disables the guard.
        """
        self._enabled = max_backward_jump_s >= 0.0
        self._max_backward_jump_ns = int(max(0.0, max_backward_jump_s) * 1e9)
        self._last_ns: dict[str, int] = {}

    def check(self, topic: str, key: str, ts_ns: int) -> None:
        """Record *ts_ns* for *topic* and reject a backward step.

        Args:
            topic: Bag topic the message came from (one clock per topic).
            key: Output feature key, for the error message only.
            ts_ns: Stamp the message will be resampled on, in nanoseconds.

        Raises:
            ValueError: If the stamp regressed by more than the tolerance.
        """
        if not self._enabled:
            return
        last_ns = self._last_ns.get(topic)
        self._last_ns[topic] = ts_ns
        if last_ns is None:
            return
        backward_ns = last_ns - ts_ns
        if backward_ns > self._max_backward_jump_ns:
            raise ValueError(
                f'Non-monotonic stamp on {topic} ({key}): went backwards '
                f'{backward_ns / 1e9:.6f} s ({last_ns / 1e9:.6f} -> '
                f'{ts_ns / 1e9:.6f}); tolerance '
                f'{self._max_backward_jump_ns / 1e9:.6f} s'
            )
