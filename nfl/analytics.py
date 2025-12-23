
import pandas as pd
from .utils import ivmap, is_listlike
from .sequence import reorder, expand, expand_len, expand_wid
import math

class NFLScenario(pd.Series):
    '''A series of game outcomes created by NFLScenarioMaker. Values will
       be one of 'win', 'loss' or 'tie', indexed by values in the *master*
       games database
    '''

    def __init__(self, host, data, index, name=None):
        super().__init__(data, index=index, name=name, dtype=str)
        self.host = host

    def weeks(self, team):
        '''Return a Series of outcomes for the specified team, one row per week
        '''

        x = self.host.games.join(self.rename('outcome'))
        return x[(x['ht']==team) | (x['at']==team)]['outcome']

    def st(self, team, outcome='win', op='count'):
        '''Test summarized outcomes for a given team.

           team     team code

           outcome  outcome to count

           op       operation - one of:

                    count: returns count of outcome
                    all:   True if all weeks equal outcome, else False
                    any:   True if any week equals outcome, else False

           The result will in all cases be a scalar
        '''

        z = self.weeks(team) == vmap(outcome)
        if op == 'all':
            return z.all()
        elif op == 'any':
            return z.any()

        return z.sum()

    def to_frame(self):
        '''Converts an NFLScenario to week,team outcomes to a Series indexed by week,team.
           The returned Series will have the same index structure as columns in an NFLScenarioFrame.
           For example:

           with NFLScenarioMaker(nfl, 'NFC-North', -1) as gen:
                outcomes = gen.frame(['outcome'])
                for scenario in gen:
                    x = len(outcomes)
                    outcomes.loc[x] = scenario.to_frame()
        '''

        aresults = {'win': 'loss', 'loss': 'win', 'tie': 'tie'}
        s = pd.Series('', index=self.host.incomplete, dtype=str)

        # this object must have the same index as self.host.games
        for (k,row) in self.host.games.iterrows():
            result = self.get(k, '')
            if (row['wk'],row['ht']) in s.index:
                s[(row['wk'],row['ht'])] = result

            if (row['wk'],row['at']) in s.index:
                s[(row['wk'],row['at'])] = aresults.get(result, '')

        return s

class NFLScenarioFrame(pd.DataFrame):

    _metadata = ['extra', 'nTeams']

    @property
    def I(self):
        '''Return the dataframe with the week/team levels inverted
        '''
        s = len(self.extra)*self.nTeams * -1

        z = self.iloc[:, :s].reorder_levels([1,0],axis=1).sort_index(axis=1).join(self.iloc[:, s:].reorder_levels([1,0], axis=1))
        
        # join doesn't carry over metadata so we have to do it ourselves
        for elem in self._metadata:
            z.__setattr__(elem, self.__getattr__(elem))

        return z


    def weeks(self, team):
        '''Return weeks for the specified team
        '''

        # team/week columns could be inverted
        level = 0 if self.columns.names[0] == 'team' else 1
        return self.drop(columns=self.extra, level=1-level).xs(team, level=level, axis=1)

    def test(self, team, *values):
        '''Apply tests by week to a given team

           team        Team code
           
           values      Test values for each week for that team ('win'|1,'loss'|-,'tie'|-1). Use
                       None to ignore a given week

           The return is a DataFrame with the same index as the object and columns for each week
        '''

        z = self.weeks(team)

        c = list(z.columns)

        # a terse way to pad with None and trim to length
        v = (list(values) + ([None] * (len(c)-len(values))))[:len(c)]

        # now eliminate values and columns that are None
        (c,v) = zip(*filter(lambda x: x[1] is not None, zip(c,v)))

        c = list(c)
        v = list(map(lambda x: vmap(x), v))

        return z[c].eq(v).all(axis=1)

    def st(self, team, outcome='win', op='count'):
        '''Test summarized outcomes for a given team

           team     team code

           outcome  outcome to count

           op       operation - one of:

                    count: returns count of outcome
                    all:   True if all weeks equal outcome, else False
                    any:   True if any week equals outcome, else False

           The return is a Series of either counts or Booleans with the same index as the object
        '''

        z = self.weeks(team) == vmap(outcome)
        if op == 'all':
            return z.all(axis=1)
        elif op == 'any':
            return z.any(axis=1)

        return z.sum(axis=1)


class NFLScenarioMaker():
    '''Facilitates generating and testing different win/lose/tie scenarios,
       typically to analyze the outcome on the playoff picture.

       If impelemented in a "with" context the object saves and restores
       the game state.

       Typical usage:

       weeks = [17, 18]
       teams = ['LAC', 'DEN', 'CIN', 'IND', 'MIA']
       with NFLScenarioMaker(nfl, weeks, teams, ties=False) as s:
          df = s.frame(['playoffs'])
          for option in s:
             z = len(df)

       NFLScenarioMaker is an iterable that you can use in wrappers like this:

       from tqdm import tqdm     # progress bar wrapper
       with NFLScenarioMaker(nfl, 'NFC-North', -1) as gen, tqdm(gen) as tgen:
          for scenario in tgen:
            ...
    '''

    def __init__(self, nfl, teams, weeks, ties=True):
        self.nfl = nfl
        self.weeks = list(nfl._weeks(weeks))
        self.teams = list(nfl._list(teams))
        self.ties = ties
        self.stash = None
        self.games = None
        self.completed = None
        self.incomplete = None


        # all resulting DataFrames and Series should be sorted by team and week
        self.teams.sort()
        self.weeks.sort()

    def __enter__(self):
        self.stash = self.nfl.stash()

        # get scope of games in this run
        gm = self.nfl.games_
        gm = gm[(gm.index.get_level_values(0)==self.nfl.season) & gm['wk'].isin(self.weeks) & \
                    (gm['ht'].isin(self.teams) | gm['at'].isin(self.teams)) & (gm['p']==False)]

        # The games index defines the structure for each scenario
        self.games = gm[['wk','at','ht']]

        # complete and incomplete are transformations of games.index, structured by [week,team]
        sch = self.nfl.schedule(self.teams, self.weeks, by='team')
        self.completed = sch.dropna().index
        self.incomplete = sch.index.drop(self.completed)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.nfl.restore(self.stash)

    def __len__(self):
        types = 2
        if self.ties:
            types += 1

        return types ** len(self.games)

    def __iter__(self):
        '''Iterate over possible scenarios
        '''

        # subroutine based on this:
        # https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays/1235363#1235363
        # 12/24: new implementation using the "yield from" construct, which returns a generator
        # instead of a potentally very large double array
        def outcomes(arrays, z=[]):

            x = arrays[0]
            if len(arrays) > 1:
                for elem in x:
                    yield from outcomes(arrays[1:], z+[elem])
            else:
                for elem in x:
                    yield z + [elem]

        values = ('win','loss','tie')
        if not self.ties:
            values = values[:-1]

        for row in outcomes([values] * len(self.games)):
            yield NFLScenario(self, row, index=self.games.index)

    def frame(self, extra=[]):
        '''Return a DataFrame structured to hold the set of scenarios. Typically
           you call this in advance of iterating over the object.

           extra specifiies additional level-0 column names.

           The resulting DataFrame will have a MultiIndex column that is the product
           of weeks and teams, plus an extra set of team columns for each 1st-level
           name in extra. For example:

           >>> with NFLScenarioMaker(nfl, ['NE', 'BUF'], [17,18]) as gen:
           ...   df = gen.frame(['divchamp'])
           ...   df.loc[0] = np.nan # each row defines a unique scenario and outcome
           ...   print(df)
           ... 
           week  17      18     divchamp    
           team  NE BUF  NE BUF       NE BUF
           0    NaN NaN NaN NaN      NaN NaN

           Typical example:

           teams = {'DEN', 'MIA', 'LAC'}
           weeks = [17, 18]

           with nfl.scenarios(weeks, teams) as s:
               df = s.frame(['playoff'])

               # NB: computationally intense: up to 729 possibilities, could take 10m or more
               for option in s:
                  at = len(df)
                  df.loc[at] = option.to_frame()
                  nfl.loc[at, 'playoff'] = ''
                  p = nfl.AFC.playoffs()
                  z = [('playoff', i) for i in set(teams) & set(p.index)]
                  nfl.loc[at, z] = 'x'
        '''

        df = NFLScenarioFrame(columns=pd.MultiIndex.from_product([self.weeks+extra, self.teams], names=['week', 'team']))
        df.extra = extra
        df.nTeams = len(self.teams)
        return df.drop(self.completed, axis=1)


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


class NFLSequenceMaker():

    def __init__(self, source, n=1):
        self.source = source
        self.n = n

    def __len__(self):

        return expand_len(self.source, self.n) * math.factorial(expand_width(self.surce, self.n))

    def __iter__(self):

        for elems in self.expand():
            yield from self.reorder(elems)

    def reorder(self, elems):
        yield from reorder(elems)


    def expand(self):
        yield from expand(self.source, self.n)
