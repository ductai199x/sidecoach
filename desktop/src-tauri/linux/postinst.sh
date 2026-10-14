#!/bin/sh
# The .deb's udev rule (70-sidecoach-input.rules) applied to the keyboards and mice already
# plugged in, so the push-to-talk key works in every window from the first run, as on Windows.
set -e
if command -v udevadm >/dev/null 2>&1; then
  udevadm control --reload || true
  udevadm trigger --subsystem-match=input --action=change || true
fi
exit 0
