
"""
nfl: a package to analyze NFL seasons with data

Typical use:

from nfl import NFL
nfl = NFL().update()             # load instance and update data from API

nfl('MIN')                       # standings and partial schedule for vikings
nfl('MIN').schedule              # full schedule
nfl.NFC                          # NFC and AFC are available as properties
nfl.wlt(['MIN', 'GB'])           # win/loss/tie info
nfl.tiebreakers(['DAL', 'PHI'])  # tiebreaker calculations
nfl.NFC.playoffs()               # playoff picture, if season ended today

nfl.scoreboard                   # live data for the current week

And lots more!

"""

from .nfl import (
	NFL,
  NFLDataFrame,
	NFLTeam,
	NFLDivision,
	NFLConference,
	NFLScoreboard,
  NFLRoster,
  NFLPlayer
)

# include these objects in the help docs
__all__ = [
  "NFL",
  "NFLDataFrame",
  "NFLTeam",
  "NFLDivision",
  "NFLConference",
  "NFLScoreboard",
  "NFLRoster",
  "NFLScenarioMaker"
]

from .profootballref import NFLSourceProFootballRef
from .espn import NFLSourceESPN
