
import pandas as pd
from .utils import ivmap

class NFLTeamMatrix(pd.DataFrame):
	'''This class specifies a team matrix with potential outcomes.
	   You can use this in calls to NFL.set to analyze potential
	   end-of-season scenarios.

	   You initialize the object with list of teams define both
	   the columns and index of a dataframe. The generic code '_'
	   is added as the last element in both directions to signify
	   an otherwise unspecified team. Note that team order is important,
	   as some class methods assume that teams are in rank order
	   with "front-runners" first
	'''

	def __init__(self, teams):
		'''Initialize the matrix with a list of teams. Note that division and conference
		   codes are *not* supported, so you must pass explicit list-likes of team codes
		   in the order you want them defined
		'''

		t = list(teams) + ['_']
		super().__init__(0, columns=t, index=t, dtype='int')
		self.columns.name = 'opp'
		self.index.name = 'name'

	def set_score(self, team, score):
		'''Set team to the specified score

		   team is a team code, or '_' for the wildcard

		   Pass an integer for score to set the score for all teams without specifying an opponent
		   Pass 'win', 'loss' or 'tie' to also set the opponent to the corresponding outcome
		'''

		if team not in self.index:
			raise ValueError('{} is not in the matrix'.format(team))

		if score in ['win','loss','tie']:
			score = ivmap(score)
			self.loc[team] = max(score,0)
			self[team] = 0 if score < 0 else 1-score

	def score(self, team, opp):
		'''Return a tuple consisting of the prescribed outcome (scores) for the two
		   specified teams. If either of the teams is not in the matrix then the generic
		   '_' outcome is returned. If neither of the teams is specified, or team
		   and opp are the same, then an empty tuple is returned
		'''
		if team not in self.index:
			team = '_'

		if opp not in self.index:
			opp = '_'

		if team == opp:
			return ()

		return (self.loc[team,opp],self.loc[opp,team])


	def clean(self):
		'''Clean the matrix, i.e. set the diagonal to 0 - purely cosmetic
		'''

		for t in self.index:
			self.loc[t, t] = 0

	def best(self, team):
		'''Set matrix to the "best possible world" scenario for the specified team.
		   This assumes that teams in the matrix are in rank order; teams at the top
		   will be handicapped to the advantage of the one specified.

		   The "best possible world" scenario provides that:

		   1) the specified team wins against all opponents
		   3) teams ranked higher than the one specified score 0 points against each other,
		      resulting in tie games
		   3) Teams ranked lower than the one specified win against higher ranked ones
		   4) All teams but the one specified lose to unspecified teams
		'''

		self[:] = 0
		self.set_score('_', 'win')
		self.set_score(team, 'win')
		a = self.index.get_loc(team)
		b = len(self.index)-1

		# teams ranked below beat teams ranked above
		self.iloc[a+1:b, :a] = 1
		self.iloc[:a, a+1:b] = 0
		self.clean()

	@property
	def teams(self):
		'''Return matrix teams as a set
		'''

		return set(self.columns[:-1])

