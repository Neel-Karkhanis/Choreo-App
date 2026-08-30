"""Terms-of-service text and version, checked at signup.

TOS_VERSION is bumped whenever TOS_TEXT changes in a way that needs fresh
acceptance. Signup validates the client's tos_accepted flag server-side
(see server.py's /api/auth/signup) — a client that skips or bypasses the
frontend's checkbox still cannot make an account without explicitly
claiming acceptance in the request itself, so users.tos_accepted_at stays
a real paper trail rather than cosmetic UI.

Not legal advice, not a compliance program — just enough of a record that
the operator asked, and the user agreed, before anything was uploaded.
"""

TOS_VERSION = "1"

TOS_TEXT = """
Choreo — Terms of Use (v1)

By creating an account you agree that:

1. You own the rights to any audio or video you upload, or you have
   permission from the rights holder to use it with this service.
2. You will not upload content that is illegal to possess or distribute.
3. The operator may remove content or terminate accounts at their
   discretion, without notice.
4. This service is provided as-is. There is no guarantee of uptime, and
   no guarantee that uploaded content or data derived from it (tapped
   grids, separated stems) will be preserved or backed up.

To request removal of your account and everything associated with it,
contact the operator.
""".strip()
